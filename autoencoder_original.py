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
residual_mode = True ### TODO maybe in data_manager, or here, or ....

# CNN_modes:
#  'snapshots' : train an instanteous model
#  'timesteps' : train a time-stepping model
# CNN_mode = 'timesteps'
CNN_mode = 'snapshots'

# enable or disable embedded ESN,
# disabled by default in snapshots mode
use_embedded_ESN = True

use_feedthrough = True
feedthrough_only = False

# Save/load settings
load_existing_model = False
overwrite_existing_model = False

# Visualization settings
plot_prediction = True

#-------------------------------------------------------
if load_existing_model:
    # 20240828_144827_snapshot_model/results/history_20240828_145013.png
    # 20240829_090516_snapshot_model/models/aencodr_20240829_090516.ker
    folder_id = '20240830_103023'
    add_id    = '_snapshot_model'
    model_id  = '20240830_103023'
else:
    folder_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    add_id = '_feedthrough_only' if feedthrough_only else '_testing'
    model_id = folder_id

# setup new or existing directories
dirs, files = dm.setup_directories(folder_id, add_id)
models_dir = dirs['models']

data, params, scalers, _  = \
    dm.create_training_data(compute_data=False,
                            residual_mode=residual_mode,
                            coarsen_in_time=True,
                            detide=True)
# truncate
# history = data['train']['HR'].shape[0] # use all data we have
history = 10000
# history = 1000
future = 400

if residual_mode:
    # input training data
    # output training data, shifted by 1
    # feedthrough data, shifted by 1
    train_data_inp = data['train']['R'][:-1,][-history:,]
    train_data_otp = data['train']['R'][1:,][-history:,]
    train_data_ft  = data['train']['FT'][1:,][-history:,]
    test_data      = data['test']['R'][:future,]
    test_data_ft   = data['test']['FT'][:future,]
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

if CNN_mode == 'snapshots':
    # Disable ESN
    use_embedded_ESN = False

    # Get rid of time shift in training data, train with the '1:'
    # range.
    train_data_inp = train_data_otp
    # snapshot mode does not give any predictions:
    plot_prediction = False

if feedthrough_only: use_embedded_ESN = False

if load_existing_model:
    load_path_autoencoder = f'{models_dir}/aencodr_{model_id}.keras'
    load_path_encoder     = f'{models_dir}/encoder_{model_id}.keras'
    load_path_decoder     = f'{models_dir}/decoder_{model_id}.keras'

    autoencoder = keras.models.load_model(load_path_autoencoder)
    esn = autoencoder.get_layer('esn_embedded')
    # overwrite parameters

    encoder = keras.models.load_model(load_path_encoder)
    num_samples = train_data_inp.shape[0]
    timeids = np.arange(num_samples)
    timetns = np.expand_dims(timeids, axis=[1,2,3])
    print('create training data for embedded ESN')
    values  = encoder.predict([train_data_inp, timetns])
    control = encoder.predict([train_data_inp, timetns])
    esn.setPars(esn_params, num_samples=num_samples)
    esn.initialize(values, control)
    esn.populate_storage(values, timeids, control)
    decoder = keras.models.load_model(load_path_decoder)

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

epochs = 50
batch_size = 4
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
                          # train_every=2,
                          train_in_epochs=[5,10,20,30],
                          num_samples=X_train[0].shape[0])

if CNN_mode == 'timesteps':
    # normal validation is not valid for timestepping mode
    validation_data=None

    # we create a custom validation using a callback at every epoch
    # end
    initial_xk   = np.expand_dims(train_data_otp[-1,:,:,:], axis=0)
    initial_xkm1 = np.expand_dims(train_data_otp[-2,:,:,:], axis=0)
    plotmachine = PlotMachine(results_dir=dirs['results'])
    if residual_mode: test_data = data['test']['HR'][:future,]
    if residual_mode: test_data_ft = data['test']['LR'][:future,]
    validation_callback = \
        CustomValidation(test_data=(test_data, T_test, test_data_ft),
                         initial_xk=(initial_xk, initial_xkm1),
                         plotmachine=plotmachine,
                         pars = {'feedthrough_only': feedthrough_only,
                                 'use_feedthrough': use_feedthrough,
                                 'residual_mode': residual_mode})

    callbacks = [esn_callback, validation_callback]

elif CNN_mode == 'snapshots':
    X_test = [test_data, T_test, test_data_ft]
    Y_test = test_data
    validation_data = (X_test, Y_test)
    callbacks = None

# TRAINING --------------------------------------------
hist = autoencoder.fit(x=X_train,
                       y=Y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks
                       )
toc = time.time()
print(f'total training time: {(toc-tic)/60}m')

# SAVING -----------------------------------------------
# save model and metadata
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
mdata_file = f'{models_dir}/mdata_{timestamp}.dill'
container = {'hist' : hist,
             'epochs' : epochs,
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

    breakpoint()
    out = validation_callback.on_epoch_end(epochs+1)

    predictions = np.zeros_like(test_data)
    xk = np.expand_dims(train_data_otp[-1,:,:,:], axis=0)
    N_steps=T_test.shape[0]
    pb_i = keras.utils.Progbar(N_steps)
    for i in range(N_steps):
        Pxk = np.expand_dims(test_data_ft[i,:,:,:], axis=0)
        tid = np.expand_dims(T_test[i,:,:,:], axis=0)
        if feedthrough_only:
            xk = autoencoder.predict([Pxk], verbose=0)
        elif use_feedthrough:
            xk = autoencoder.predict([xk, tid, Pxk], verbose=0)
        else:
            xk = autoencoder.predict([xk, tid], verbose=0)

        predictions[i,:,:,:] = xk
        pb_i.add(1)

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
    print('print history')
    plotmachine = PlotMachine(results_dir=dirs['results'])
    plotmachine.plot_history(hist)
