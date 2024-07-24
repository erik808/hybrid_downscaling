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

import esn_interface
reload(esn_interface)
from esn_interface import ESN_embedded

import plot_utils
reload(plot_utils)
from plot_utils import PlotMachine

# setup config
new_experiment=True
training_mode='normal'
do_prediction = True
use_feedthrough = True

if new_experiment:
    load_models_from_file=False
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    add_id = '_with_feedthrough'

    # experiment_id = 'tuning'
    # add_id = ''
else:
    load_models_from_file=True
    add_id = ''
    experiment_id = '20240718_153705_optimized'

dirs, files = dm.setup_directories(experiment_id, add_id)

models_dir = dirs['models']
tuning_dir = dirs['tuning']
results_dir = dirs['results']
movie_dir = dirs['movies']
checkpoints_dir = dirs['checkpoints']
log_file = files['log']

data, params, scalers, _  = dm.create_training_data(False)
train_data    = data['train']['HR'][:1007,]
train_data_ft = data['train']['LR'][:1007,]
train_time    = data['train']['time'][:1007,]
test_data     = data['test']['HR'][:1007,]
test_data_ft  = data['test']['LR'][:1007,]
test_time     = data['test']['time'][:1007,]
mask = params['mask']
Nt = params['Nt']
Nlon = params['Nlon']
Nlat = params['Nlat']
num_channels = params['num_channels']

## Build an autoencoder with Keras using the functional API
keras.utils.clear_session(free_memory=True)

model_path_autoencoder = f'{models_dir}/autoencoder_res.keras'
model_path_encoder = f'{models_dir}/encoder_res.keras'
model_path_decoder = f'{models_dir}/decoder_res.keras'


esn_params = esn_interface.hyperparams
esn = ESN_embedded(esn_params=esn_params,
                   total_num_samples=train_data.shape[0])

ae = AutoEncoder(test_vec=train_data[0,:,:,:],
                 mask=mask,
                 log_file=log_file,
                 esn=esn)

autoencoder, encoder, decoder = ae.build_model(use_feedthrough=use_feedthrough,
                                               feedthrough_type='multiply')

if load_models_from_file:
    autoencoder = keras.models.load_model(model_path_autoencoder)
    encoder = keras.models.load_model(model_path_encoder)
    decoder = keras.models.load_model(model_path_decoder)

print('--------------------------------------')
print(f'experiment: {experiment_id}{add_id}')
print('--------------------------------------')

if training_mode == 'normal':
    checkpoint_filepath = f'{checkpoints_dir}/checkpoint.model.keras'

    mdl_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

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

    epochs = 2
    batch_size = 4
    shuffle = True
    tic = time.time()

    # really necessary to expland to 4 dims?    
    T_train = np.expand_dims(np.arange(train_data.shape[0]), axis=[1,2,3])
    T_test = np.expand_dims(np.arange(train_data.shape[0],
                                      train_data.shape[0] + test_data.shape[0]),
                            axis=[1,2,3])
    if ae.use_feedthrough:
        X_train = [train_data, T_train, train_data_ft]
        X_test = [test_data, T_test, test_data_ft]
    else:
        X_train = [train_data, T_train]
        X_test = [test_data, T_train]

    Y_train = train_data
    Y_test = test_data

    hist = autoencoder.fit(x=X_train,
                           y=Y_train,
                           epochs=epochs,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           # validation_data=(X_test, Y_test),
                           validation_data=None, # validation does not
                                                 # work with embedded
                                                 # ESN
                           callbacks=[mdl_callback, tb_callback]
                           )
    toc = time.time()
    print(f'total training time: {(toc-tic)/60}m')

    # save models
    autoencoder.save(model_path_autoencoder)
    encoder.save(model_path_encoder)
    decoder.save(model_path_decoder)

elif training_mode == 'tuning':
    do_prediction = False
    ae_tuning = AutoEncoder(test_vec=train_data[0,:,:,:],
                            mask=mask, log_file=log_file)
    epochs = 2
    batch_size = 2
    shuffle = True

    tuner = keras_tuner.Hyperband(
        hypermodel=AutoEncoder(test_vec=train_data[0,:,:,:],
                               mask=mask, log_file=log_file),
        objective="val_loss",
#        max_trials=200,
        max_epochs=200,
#        executions_per_trial=2,
#        epochs=2,
        overwrite=True,
        directory=tuning_dir,
        project_name="ae_tuning",
    )

    tuner.search_space_summary()

    hist = tuner.search(train_data,
                        train_data,
                        epochs=epochs,
                        #batch_size=batch_size,
                        shuffle=shuffle,
                        validation_data=(test_data, test_data))

    tuner.results_summary()
else:
    print('-- Skipping training --')
    pass

# save modeldata
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

if do_prediction:

    print('create predictions')
    predictions = autoencoder.predict(X_test)
    encoded_xr_HR_true = encoder.predict(train_data)

    # Create dictionary for output visualization
    xr_HR_true_fun = lambda i : scalers['HR'].inverse_transform(test_data[i,:,:,:]\
                                                                .reshape(1,-1))\
                                             .reshape(Nlat, Nlon, num_channels)

    # total kinetic energy
    Kt_HR_true_fun = lambda i : np.sqrt(np.square(xr_HR_true_fun(i)).sum(axis=2))

    xr_HR_pred_fun = lambda i : scalers['HR'].inverse_transform(predictions[i,:,:,:]\
                                                                .reshape(1,-1))\
                                             .reshape(Nlat, Nlon, num_channels)

    Kt_HR_pred_fun = lambda i : np.sqrt(np.square(xr_HR_pred_fun(i)).sum(axis=2))

    xr_HR_diff_fun = lambda i : xr_HR_true_fun(i) - xr_HR_pred_fun(i)

    Kt_HR_diff_fun = lambda i : Kt_HR_true_fun(i) - Kt_HR_pred_fun(i)

    Rs_true_fun = lambda i : test_data[i,:,:,0] - test_data_ft[i,:,:,0]
    Rs_pred_fun = lambda i : predictions[i,:,:,0] - test_data_ft[i,:,:,0]
    Rs_diff_fun = lambda i : Rs_true_fun(i) - Rs_pred_fun(i)

    enc_xr_HR_k_fun = lambda i,k : (encoded_xr_HR_true[i,:,:,k])
    enc_vmax = lambda i : np.max(encoded_xr_HR_true[:,:,:,i])
    enc_vmin = lambda i : np.min(encoded_xr_HR_true[:,:,:,i])

    output_dict = {'Kt_HR true' : {'values' : Kt_HR_true_fun,
                                   'vmin' : 0,
                                   'vmax' : .8,
                                   'cmap' : 'viridis'},
                   'Kt_HR pred' : {'values' : Kt_HR_pred_fun,
                                   'vmin' : 0,
                                   'vmax' : .8,
                                   'cmap' : 'viridis'},
                   'Kt_HR diff' : {'values' : Kt_HR_diff_fun,
                                   'vmin' : -0.1,
                                   'vmax' : 0.1,
                                   'cmap' : 'RdBu'},

                   'res true' : {'values' : Rs_true_fun,
                                 'vmin' : -0.1,
                                 'vmax' : 0.1,
                                 'cmap' : 'RdBu'},
                   'res pred' : {'values' : Rs_pred_fun,
                                 'vmin' : -0.1,
                                 'vmax' : 0.1,
                                 'cmap' : 'RdBu'},
                   'res dif' : {'values' : Rs_diff_fun,
                                'vmin' : -0.05,
                                'vmax' : 0.05,
                                'cmap' : 'RdBu'},

                   'enc xr_HR ch:0' : {'values' : lambda i : enc_xr_HR_k_fun(i,0),
                                       'vmin' : enc_vmin(0),
                                       'vmax' : enc_vmax(0),
                                       'cmap' : 'viridis'},
                   'enc xr_HR ch:1' : {'values' : lambda i : enc_xr_HR_k_fun(i,1),
                                       'vmin' : enc_vmin(1),
                                       'vmax' : enc_vmax(1),
                                       'cmap' : 'viridis'},
                   'enc xr_HR ch:2' : {'values' : lambda i : enc_xr_HR_k_fun(i,2),
                                       'vmin' : enc_vmin(2),
                                       'vmax' : enc_vmax(2),
                                       'cmap' : 'viridis'},
                   }

    plotmachine = PlotMachine(output_dict=output_dict,
                              results_dir=results_dir,
                              movie_dir=movie_dir,
                              time_array=test_time)

    plotmachine.plot_single_frame(100)
    plotmachine.create_movie()
    plotmachine.plot_history(hist)
