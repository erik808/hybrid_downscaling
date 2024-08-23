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

# setup config
new_model=False

# CNN_modes:
#  'snapshots' : train an instanteous model
#  'timesteps' : train a time-stepping model
CNN_mode = 'timesteps'
# CNN_mode = 'snapshots'

do_prediction = True
use_feedthrough = True
feedthrough_only = False

if new_model:
    load_models_from_file=False
    # model_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    # add_id = '_testing'
    model_id = 'testing'
    add_id = ''
else:
    load_models_from_file=True
    add_id = ''
    model_id = '20240823_133518_testing'

dirs, files = dm.setup_directories(model_id, add_id)


data, params, scalers, _  = dm.create_training_data(False)

# truncate
history = data['train']['HR'].shape[0] # use all data we have
# history = 10001
history = 1000
future = 1000

# input training data
train_data_inp = data['train']['HR'][:-1,][-history:,]
# output training data, shifted by 1
train_data_otp = data['train']['HR'][1:,][-history:,]
# feedthrough data, shifted by 1
train_data_ft = data['train']['LR'][1:,][-history:,]

test_data     = data['test']['HR'][:future,]
test_data_ft  = data['test']['LR'][:future,]
test_time     = data['test']['time'][:future,]
mask = params['mask']
Nt = params['Nt']
Nlon = params['Nlon']
Nlat = params['Nlat']
num_channels = params['num_channels']

## Build an autoencoder with Keras using the functional API
keras.utils.clear_session(free_memory=True)

models_dir = dirs['models']
model_path_autoencoder = f'{models_dir}/autoencoder_res.keras'
model_path_encoder = f'{models_dir}/encoder_res.keras'
model_path_decoder = f'{models_dir}/decoder_res.keras'

esn_params = esn_interface.hyperparams

if CNN_mode == 'snapshots':
    # Disable ESN
    esn_params['external']['bypass_mode'] = True
    # Get rid of time shift in training data, train with the '1:'
    # range.
    train_data_inp = train_data_otp
    # snapshot mode does not give any predictions:
    do_prediction = False

elif CNN_mode == 'timesteps':
    # make sure the ESN is not bypassed in this mode
    esn_params['external']['bypass_mode'] = False

esn = ESN_embedded(esn_params=esn_params)

log_file = files['log']
ae = AutoEncoder(test_vec=train_data_inp[0,:,:,:],
                 mask=mask,
                 log_file=log_file,
                 esn=esn)

autoencoder, encoder, decoder = \
    ae.build_model(use_feedthrough=use_feedthrough,
                   feedthrough_only=feedthrough_only,
                   feedthrough_type='multiply')


if load_models_from_file:
    autoencoder = keras.models.load_model(model_path_autoencoder)
    esn = autoencoder.get_layer('esn_embedded')
    # overwrite parameters

    encoder = keras.models.load_model(model_path_encoder)
    num_samples = train_data_inp.shape[0]
    timeids = np.arange(num_samples)
    timetns = np.expand_dims(timeids, axis=[1,2,3])
    print('create training data for embedded ESN')
    values  = encoder.predict([train_data_inp, timetns])
    control = encoder.predict([train_data_inp, timetns])
    esn.setPars(esn_params, num_samples=num_samples)
    esn.initialize(values, control)
    esn.populate_storage(values, timeids, control)
    decoder = keras.models.load_model(model_path_decoder)

print('--------------------------------------')
print(f'experiment: {model_id}{add_id}')
print('--------------------------------------')

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

# callback to trigger ESN training
epochs = 10
batch_size = 4
shuffle = True
tic = time.time()

# really necessary to expand to 4 dims?
T_train = np.expand_dims(np.arange(train_data_inp.shape[0]), axis=[1,2,3])
T_test  = np.expand_dims(np.arange(train_data_inp.shape[0],
                                   train_data_inp.shape[0] + test_data.shape[0]),
                         axis=[1,2,3])
if feedthrough_only:
    X_train = [train_data_ft]
elif use_feedthrough:
    X_train = [train_data_inp, T_train, train_data_ft]
else:
    X_train = [train_data_inp, T_train]

Y_train = train_data_otp

esn_callback = TriggerESN(esn,
                          train_in_epochs=[0],
                          num_samples=X_train[0].shape[0])


if CNN_mode == 'timesteps':
    # normal validation does not work with embedded ESN
    validation_data=None

    # we create a custom validation using a callback at every epoch
    # end
    initial_xk = np.expand_dims(train_data_otp[-1,:,:,:], axis=0)
    plotmachine = PlotMachine(results_dir=dirs['results'])
    validation_callback = \
        CustomValidation(test_data=(test_data, T_test, test_data_ft),
                         initial_xk=initial_xk,
                         plotmachine=plotmachine)

    callbacks=[esn_callback, validation_callback]
    
elif CNN_mode == 'snapshots':
    X_test = [test_data, T_test, test_data_ft]
    Y_test = test_data
    validation_data=(X_test, Y_test)
    callbacks=None

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

# save models
autoencoder.save(model_path_autoencoder)
encoder.save(model_path_encoder)
decoder.save(model_path_decoder)


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

    predictions = np.zeros_like(test_data)
    xk = np.expand_dims(train_data_otp[-1,:,:,:], axis=0)
    for i in range(T_test.shape[0]):
        print(f'{i} / {T_test.shape[0]}')
        Pxk = np.expand_dims(test_data_ft[i,:,:,:], axis=0)
        tid = np.expand_dims(T_test[i,:,:,:], axis=0)
        if feedthrough_only:
            xk = autoencoder.predict([Pxk], verbose=0)
        elif use_feedthrough:
            xk = autoencoder.predict([xk, tid, Pxk], verbose=0)
        else:
            xk = autoencoder.predict([xk, tid], verbose=0)

        predictions[i,:,:,:] = xk

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

    output_dict = {'Kt_HR true' : {'values' : Kt_HR_true_fun,
                                   'vmin' : 0,
                                   'vmax' : .8,
                                   'cmap' : 'viridis'},

                   'Kt_HR pred' : {'values' : Kt_HR_pred_fun,
                                   'vmin' : 0,
                                   'vmax' : .8,
                                   'cmap' : 'viridis'},

                   'Kt_LR true' : {'values' : Kt_LR_true_fun,
                                   'vmin' : 0,
                                   'vmax' : .8,
                                   'cmap' : 'viridis'},

                   'Kt_HR diff' : {'values' : Kt_HR_diff_fun,
                                   'vmin' : -0.2,
                                   'vmax' : 0.2,
                                   'cmap' : 'RdBu'},

                   'res true' : {'values' : Rs_true_fun,
                                 'vmin' : -0.2,
                                 'vmax' : 0.2,
                                 'cmap' : 'RdBu'},

                   'res pred' : {'values' : Rs_pred_fun,
                                 'vmin' : -0.2,
                                 'vmax' : 0.2,
                                 'cmap' : 'RdBu'},

                   # 'res dif' : {'values' : Rs_diff_fun,
                   #              'vmin' : -0.05,
                   #              'vmax' : 0.05,
                   #              'cmap' : 'RdBu'},
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
