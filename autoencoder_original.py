import os
import sys
os.system('export MKL_NUM_THREADS=12')
os.system('export OMP_NUM_THREADS=12')

from datetime import datetime
import time
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
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

import plot_utils
reload(plot_utils)
from plot_utils import PlotMachine

# setup config
new_experiment=True
training_mode='normal'
do_prediction = True
if new_experiment:
    load_models_from_file=False
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    add_id = '_tf_multiply'
    
    # experiment_id = 'tuning'
    # add_id = ''
else:
    load_models_from_file=True
    add_id = ''
    experiment_id = '20240718_153705_optimized'

models_dir = f'experiments/{experiment_id}{add_id}/models'
tuning_dir = f'experiments/{experiment_id}{add_id}/tuning'
results_dir = f'experiments/{experiment_id}{add_id}/results'
movie_dir = f'experiments/{experiment_id}{add_id}/movies'
checkpoints_dir = f'experiments/{experiment_id}{add_id}/checkpoints'
log_file = f'{models_dir}/log.txt'

os.system(f'mkdir -p {models_dir}')
os.system(f'mkdir -p {tuning_dir}')
os.system(f'mkdir -p {movie_dir}')
os.system(f'mkdir -p {results_dir}')
os.system(f'mkdir -p {checkpoints_dir}')

# assume everything has this shape
da_HR, da_LR, da_mask = dm.load_uv_data()

# do the assembling into channels here
data_HR_stacked = np.stack([da_HR['uo'].values,
                            da_HR['vo'].values], axis=3)
data_LR_stacked = np.stack([da_LR['uo'].values,
                            da_LR['vo'].values], axis=3)

Nt, Nlat, Nlon, num_channels = data_HR_stacked.shape

scaled_range = (0,1)

# StandardScaler doesnt work that well
scaler_HR = MinMaxScaler(feature_range=scaled_range)
data_HR = scaler_HR.fit_transform(data_HR_stacked.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon, num_channels)
data_LR = scaler_HR.transform(data_LR_stacked.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon, num_channels)

plt.close('all')

split = int(Nt*4/5)
train_range = range(0, split)
test_range = range(split, Nt)

train_data = data_HR[train_range,:,:,:]
test_data = data_HR[test_range,:,:,:]
# train and test data used for feedthrough connection in AE
train_data_ft = data_LR[train_range,:,:,:]
test_data_ft = data_LR[test_range,:,:,:]

test_LR = test_data_ft
train_time = da_LR['uo'].time.values[train_range]
test_time  = da_LR['uo'].time.values[test_range]

# create mask to be used in network
mask = torch.tensor(da_mask.values)[None,:,:,None]

# clean memory
del data_HR, data_LR, dm, da_HR, da_LR

## Build an autoencoder with Keras using the functional API
keras.utils.clear_session(free_memory=True)

model_path_autoencoder = f'{models_dir}/autoencoder_res.keras'
model_path_encoder = f'{models_dir}/encoder_res.keras'
model_path_decoder = f'{models_dir}/decoder_res.keras'

ae = AutoEncoder(test_vec=train_data[0,:,:,:],
                 mask=mask, log_file=log_file)
autoencoder, encoder, decoder = ae.build_model(use_feedthrough=True,
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

    epochs = 50
    batch_size = 2
    shuffle = True
    tic = time.time()

    if ae.use_feedthrough:
        X_train = [train_data, train_data_ft]
        X_test = [test_data, test_data_ft]
    else:
        X_train = train_data
        X_test = test_data

    Y_train = train_data
    Y_test = test_data
    
    hist = autoencoder.fit(x=X_train,
                           y=Y_train,
                           epochs=epochs,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           validation_data=(X_test, Y_test),
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

if do_prediction:

    print('create predictions')
    predictions = autoencoder.predict(X_test)
    encoded_xr_HR_true = encoder.predict(train_data)

    # Create dictionary for output visualization
    xr_HR_true_fun = lambda i : scaler_HR.inverse_transform(test_data[i,:,:,:]\
                                                            .reshape(1,-1))\
                                         .reshape(Nlat, Nlon, num_channels)

    # total kinetic energy
    Kt_HR_true_fun = lambda i : np.sqrt(np.square(xr_HR_true_fun(i)).sum(axis=2))

    xr_HR_pred_fun = lambda i : scaler_HR.inverse_transform(predictions[i,:,:,:]\
                                                            .reshape(1,-1))\
                                         .reshape(Nlat, Nlon, num_channels)

    Kt_HR_pred_fun = lambda i : np.sqrt(np.square(xr_HR_pred_fun(i)).sum(axis=2))

    xr_HR_diff_fun = lambda i : xr_HR_true_fun(i) - xr_HR_pred_fun(i)

    Kt_HR_diff_fun = lambda i : Kt_HR_true_fun(i) - Kt_HR_pred_fun(i)

    Rs_true_fun = lambda i : test_data[i,:,:,0] - test_LR[i,:,:,0]
    Rs_pred_fun = lambda i : predictions[i,:,:,0] - test_LR[i,:,:,0]
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
