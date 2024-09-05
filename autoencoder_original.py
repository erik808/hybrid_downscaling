import os
import dill

from datetime import datetime
import time
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import keras
import keras_tuner
from keras import layers
from keras import ops
from keras.models import Model

import data_manager as dm
reload(dm)
import ae_model
reload(ae_model)
from ae_model import AutoEncoder
from ae_model import TriggerESN
from ae_model import CustomValidation

import esn_interface
reload(esn_interface)
from esn_interface import ESN_embedded

import plot_utils
reload(plot_utils)
from plot_utils import PlotMachine

#-------------------------------------------------------
#-------------------------------------------------------
# Experiment settings

# If True, train and predict residuals: R such that X_LR + R = X_HR
residual_mode = False

use_feedthrough = True
feedthrough_only = False

overwrite_existing_model = False
select_existing_model = False

# truncate
# history = 'all'
history = 15000
# history = 4000
future = 400

# 
epochs = [10,30]
batch_size = 4
esn_train_in_epochs=[0,2,4,8]

compute_data=False

data, params, scalers, _  = \
    dm.create_training_data(compute_data=compute_data,
                            residual_mode=residual_mode,
                            coarsen_in_time=False,
                            detide=False)

# CNN_modes:
#  'snapshots' : train an instanteous model
#  'timesteps' : train a time-stepping model
CNN_modes = ['timesteps', 'timesteps']

loading = [False, True]
for itr, (CNN_mode, load_existing_model) in enumerate(zip(CNN_modes, loading)):
    use_embedded_ESN = False

    #-------------------------------------------------------
    if select_existing_model:
        # 20240828_144827_snapshot_model/results/history_20240828_145013.png
        # 20240829_090516_snapshot_model/models/aencodr_20240829_090516.ker
        folder_id = '20240904_152647'
        add_id    = '_snapshot_model'
        model_id  = '20240904_152647'
    elif not load_existing_model:
        folder_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = folder_id
        add_id = '_feedthrough_only' if feedthrough_only else ''
        add_id = '_snapshot_model' if CNN_mode == 'snapshots' else add_id
        add_id = '_timestep_model' if CNN_mode == 'timesteps' else add_id

    # setup new or existing directories
    dirs, files = dm.setup_directories(folder_id, add_id)
    models_dir = dirs['models']

    if history == 'all':
         # use all data we have
        history = data['train']['HR'].shape[0]

    if residual_mode:
        # input training data
        train_data_inp = data['train']['R'][:-1,][-history:,]
        # output training data, shifted by 1
        train_data_otp = data['train']['R'][1:,][-history:,]
        # feedthrough data, shifted by 1
        train_data_ft  = data['train']['LR'][1:,][-history:,]

        test_data      = data['test']['R'][:future,]
        test_data_ft   = data['test']['LR'][:future,]
        test_time      = data['test']['time'][:future,]
    else:
        train_data_inp = data['train']['HR'][:-1,][-history:,]
        train_data_otp = data['train']['HR'][1:,][-history:,]
        train_data_ft  = data['train']['LR'][1:,][-history:,]
        test_data      = data['test']['HR'][:future,]
        test_data_ft   = data['test']['LR'][:future,]
        test_time      = data['test']['time'][:future,]

    mask = params['mask']
    Nt   = params['Nt']
    Nlon = params['Nlon']
    Nlat = params['Nlat']
    num_channels = params['num_channels']

    ## Build an autoencoder with Keras using the functional API
    keras.utils.clear_session(free_memory=True)

    esn_params = esn_interface.hyperparams

    plot_prediction = True
    if CNN_mode == 'snapshots':
        # Disable ESN
        use_embedded_ESN = False

        # Get rid of time shift in training data, train with the '1:'
        # range.
        # train_data_inp = train_data_otp
        # snapshot mode does not give any predictions:
        plot_prediction = False        

    if feedthrough_only: use_embedded_ESN = False

    if load_existing_model:
        load_path_autoencoder = f'{models_dir}/aencodr_{model_id}.keras'
        load_path_encoder     = f'{models_dir}/encoder_{model_id}.keras'
        load_path_decoder     = f'{models_dir}/decoder_{model_id}.keras'

        autoencoder = keras.models.load_model(load_path_autoencoder)
        encoder = keras.models.load_model(load_path_encoder)
        decoder = keras.models.load_model(load_path_decoder)
        if use_embedded_ESN:
            esn = autoencoder.get_layer('esn_embedded')
            # overwrite parameters
            
            num_samples = train_data_inp.shape[0]
            timeids = np.arange(num_samples)
            timetns = np.expand_dims(timeids, axis=[1,2,3])
            print('create training data for embedded ESN')
            esn.setPars(esn_params, num_samples=num_samples)
            values  = esn.pixel_shuffle(encoder.predict([train_data_inp, timetns]))
            control = esn.pixel_shuffle(encoder.predict([train_data_ft, timetns]))
            esn.initialize(values, control)
            esn.populate_storage(values, timeids, control)

    else:

        esn_params['external']['bypass_mode'] = not use_embedded_ESN
        esn = ESN_embedded(esn_params=esn_params)

        ae = AutoEncoder(test_vec=train_data_inp[0,:,:,:],
                         mask=mask,
                         log_file=files['log'],
                         esn=esn)

        autoencoder, encoder, decoder = \
        ae.build_model(use_feedthrough=use_feedthrough,
                       feedthrough_only=feedthrough_only,
                       feedthrough_type='multiply')

    autoencoder.summary()

    print('----------------------------------------------------------')
    print(f'experiment: {folder_id}{add_id}, model: {model_id}')
    print('---------------------------------------------------------')

    checkpoints_dir = dirs['checkpoints']
    checkpoint_filepath = f'{checkpoints_dir}/checkpoint.model.keras'

    # callback to create checkpoints
    mdl_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # callback for extra output to tensorboard
    tb_callback = keras.callbacks.TensorBoard(
        log_dir=models_dir,
        histogram_freq=1,
        write_graph=False,
        write_images=True,
        write_steps_per_second=True,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    shuffle = True
    tic = time.time()

    # really necessary to expand to 4 dims?
    T_train = np.expand_dims(np.arange(train_data_inp.shape[0]), axis=[1,2,3])
    T_test  = np.expand_dims(np.arange(train_data_inp.shape[0],
                                       train_data_inp.shape[0] +
                                       test_data.shape[0]),
                             axis=[1,2,3])

    if feedthrough_only:
        X_train = [train_data_ft]
    elif use_feedthrough:
        X_train = [train_data_inp, T_train, train_data_ft]
    else:
        X_train = [train_data_inp, T_train]

    Y_train = train_data_otp

    esn_callback = TriggerESN(esn,
                              train_in_epochs=esn_train_in_epochs,
                              num_samples=X_train[0].shape[0])

    if CNN_mode == 'timesteps':
        # we create a custom validation using a callback at every epoch
        # end
        initial_xk   = np.expand_dims(data['train']['HR'][-1,:,:,:], axis=0)
        initial_xkm1 = np.expand_dims(data['train']['HR'][-2,:,:,:], axis=0)
        plotmachine = PlotMachine(results_dir=dirs['results'])
        if residual_mode: test_data = data['test']['HR'][:future,]
        if residual_mode: test_data_ft = data['test']['LR'][:future,]

        validation_callback = \
            CustomValidation(test_data=(test_data, T_test, test_data_ft),
                             initial_xk=(initial_xk, initial_xkm1),
                             plotmachine=plotmachine,
                             pars = {'feedthrough_only': feedthrough_only,
                                     'use_feedthrough': use_feedthrough,
                                     'residual_mode': residual_mode},
                             scalers = scalers)

        callbacks = [esn_callback, validation_callback]

    elif CNN_mode == 'snapshots':
        callbacks = None

    # TRAINING --------------------------------------------
    hist = autoencoder.fit(x=X_train,
                           y=Y_train,
                           epochs=epochs[itr],
                           batch_size=batch_size,
                           shuffle=shuffle,
                           validation_data=None,
                           callbacks=callbacks
                           )
    toc = time.time()
    print(f'total training time: {(toc-tic)/60}m')

    # SAVING -----------------------------------------------
    # save model and metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mdata_file = f'{models_dir}/mdata_{timestamp}.dill'
    container = {'hist' : hist,
                 'epochs' : epochs[itr],
                 'batch_size' : batch_size,
                 'encoder' : encoder,
                 'decoder' : decoder,
                 'autoencoder' : autoencoder}
    with open(mdata_file, 'wb') as file:
        dill.dump(container, file)

    # save models
    if (load_existing_model and
        not overwrite_existing_model):
        save_path_autoencoder = f'{models_dir}/aencodr_{timestamp}.keras'
        save_path_encoder     = f'{models_dir}/encoder_{timestamp}.keras'
        save_path_decoder     = f'{models_dir}/decoder_{timestamp}.keras'
    else:
        save_path_autoencoder = f'{models_dir}/aencodr_{model_id}.keras'
        save_path_encoder     = f'{models_dir}/encoder_{model_id}.keras'
        save_path_decoder     = f'{models_dir}/decoder_{model_id}.keras'

    print(f'saving autoencoder to {save_path_autoencoder}')
    print(f'saving encoder to {save_path_encoder}')
    print(f'saving decoder to {save_path_decoder}')
    autoencoder.save(save_path_autoencoder)
    encoder.save(save_path_encoder)
    decoder.save(save_path_decoder)

    # PLOTTING --------------------------------------------
    if plot_prediction:
        print('create predictions')

        predictions = validation_callback.predictions

        # Create dictionary for output visualization
        xr_HR_true_fun = lambda i : \
            scalers['HR'].inverse_transform(test_data[i,:,:,:]\
                                            .reshape(1,-1))\
                         .reshape(Nlat, Nlon, num_channels)

        # instant kinetic energy
        Kt_HR_true_fun = lambda i : \
            np.sqrt(np.square(xr_HR_true_fun(i)).sum(axis=2))

        xr_HR_pred_fun = lambda i : \
            scalers['HR'].inverse_transform(predictions[i,:,:,:]\
                                            .reshape(1,-1))\
                         .reshape(Nlat, Nlon, num_channels)

        # instant kinetic energy
        Kt_HR_pred_fun = lambda i : \
            np.sqrt(np.square(xr_HR_pred_fun(i)).sum(axis=2))

        # Create dictionary for output visualization
        xr_LR_true_fun = lambda i : \
            scalers['HR'].inverse_transform(test_data_ft[i,:,:,:]\
                                            .reshape(1,-1))\
                         .reshape(Nlat, Nlon, num_channels)

        # instant kinetic energy
        Kt_LR_true_fun = lambda i : \
            np.sqrt(np.square(xr_LR_true_fun(i)).sum(axis=2))


        xr_HR_diff_fun = lambda i : xr_HR_true_fun(i) - xr_HR_pred_fun(i)

        Kt_HR_diff_fun = lambda i : Kt_HR_true_fun(i) - Kt_HR_pred_fun(i)

        Rs_true_fun = lambda i : test_data[i,:,:,0] - test_data_ft[i,:,:,0]
        Rs_pred_fun = lambda i : predictions[i,:,:,0] - test_data_ft[i,:,:,0]
        Rs_diff_fun = lambda i : Rs_true_fun(i) - Rs_pred_fun(i)

        vmax = Kt_HR_true_fun(0).max()
        vmin_diff = Kt_HR_diff_fun(0).min()
        vmax_diff = Kt_HR_diff_fun(0).max()

        output_dict = {'Kt_HR true' : {'values' : Kt_HR_true_fun,
                                       'vmin' : 0,
                                       'vmax' : vmax,
                                       'cmap' : 'viridis'},

                       'Kt_HR pred' : {'values' : Kt_HR_pred_fun,
                                       'vmin' : 0,
                                       'vmax' : vmax,
                                       'cmap' : 'viridis'},

                       'Kt_LR true' : {'values' : Kt_LR_true_fun,
                                       'vmin' : 0,
                                       'vmax' : vmax,
                                       'cmap' : 'viridis'},

                       'Kt_HR diff' : {'values' : Kt_HR_diff_fun,
                                       'vmin' : vmin_diff,
                                       'vmax' : vmax_diff,
                                       'cmap' : 'RdBu'},

                       'res true' : {'values' : Rs_true_fun,
                                     'vmin' : vmin_diff,
                                     'vmax' : vmax_diff,
                                     'cmap' : 'RdBu'},

                       'res pred' : {'values' : Rs_pred_fun,
                                     'vmin' : vmin_diff,
                                     'vmax' : vmax_diff,
                                     'cmap' : 'RdBu'},
                       }

        plotmachine = PlotMachine(output_dict=output_dict,
                                  results_dir=dirs['results'],
                                  movie_dir=dirs['movies'],
                                  time_array=test_time)

        plotmachine.plot_prediction_error(test_data, predictions, test_data_ft)
        plotmachine.plot_single_frame(100)
        plotmachine.create_movie()
        plotmachine.plot_history(hist)

    else: # only pot the history
        import plot_utils
        reload(plot_utils)
        from plot_utils import PlotMachine

        print('print history')
        plotmachine = PlotMachine(results_dir=dirs['results'])
        plotmachine.plot_history(hist)
