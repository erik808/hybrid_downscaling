import os
import sys
os.system('export MKL_NUM_THREADS=12')
os.system('export OMP_NUM_THREADS=12')

from datetime import datetime
import time
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.multiprocessing
import keras
from keras import layers
from keras import ops
from keras.models import Model

import data_manager as dm
experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')

experiment_id = '20240712_171926'

models_dir = f'experiments/{experiment_id}/models'
results_dir = f'experiments/{experiment_id}/results'
checkpoints_dir = f'experiments/{experiment_id}/checkpoints'

log_file = f'{models_dir}/log.txt'

os.system(f'mkdir -p {models_dir}')
os.system(f'mkdir -p {results_dir}')
os.system(f'mkdir -p {checkpoints_dir}')

# assume everything has this shape
Nt, Nlat, Nlon = dm.da_HR.shape

scaled_range = (0,1)

# StandardScaler doesnt work that well
scaler_HR = MinMaxScaler(feature_range=scaled_range)
scaler_LR = MinMaxScaler(feature_range=scaled_range)
scaler_Rs = MinMaxScaler(feature_range=scaled_range)

da_Rs = dm.da_HR - dm.da_LR

data_HR = scaler_HR.fit_transform(dm.da_HR.values.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon,1)

data_HR = scaler_HR.fit_transform(dm.da_HR.values.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon,1)

data_LR = scaler_LR.fit_transform(dm.da_LR.values.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon,1)
data_Rs = scaler_Rs.fit_transform(da_Rs.values.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon,1)


plt.close('all')

Nt, Nlat, Nlon, Nchannels = data_HR.shape
split = int(Nt*4/5)
train_range = range(0, split)
test_range = range(split, Nt)

stacked_data = np.concatenate([data_Rs, data_LR], axis=3)

train_data = stacked_data[train_range,:,:,:]
test_data = stacked_data[test_range,:,:,:]

# clean memory
del stacked_data, data_Rs, data_LR, dm

## Build an autoencoder with Keras using the functional API
keras.utils.clear_session(free_memory=True)

model_path_autoencoder = f'{models_dir}/autoencoder_res.keras'
model_path_encoder = f'{models_dir}/encoder_res.keras'
model_path_decoder = f'{models_dir}/decoder_res.keras'

# create custom masking class

create_model_from_scratch=False
if create_model_from_scratch:
    num_filters = 32
    num_channels = train_data.shape[-1]
    state_input = layers.Input(shape=(Nlat, Nlon, num_channels),
                               name="full_state_input")
    # Encoding layers
    e1 = layers.Conv2D(num_filters, (3,3), activation="relu",
                       padding="same")(state_input)
    e2 = layers.MaxPooling2D((2,2),
                             padding="same")(e1)
    e3 = layers.Conv2D(num_filters, (3,3), activation="relu",
                       padding="same")(e2)
    encoded = layers.MaxPooling2D((2,2),
                                  padding="same")(e3)

    encoder = Model(state_input, encoded, name="encoder")
    encoder.summary(60)

    # Decoder
    d1 = layers.Conv2DTranspose(num_filters, (3,3), strides=2, activation="relu",
                                padding="same")(encoded)
    d2 = layers.Conv2DTranspose(num_filters, (3,3), strides=2, activation="relu",
                                padding="same")(d1)
    d3 = layers.Conv2D(num_channels, (3,3), activation="sigmoid",
                       padding="same")(d2)
    decoded = layers.Cropping2D(cropping=((2,1),(2,1)))(d3)

    decoder = Model(encoded, decoded, name="decoder")
    decoder.summary(60)

    autoencoder = Model(state_input, decoded, name="autoencoder")
    autoencoder.summary(60)

elif isinstance(model_path_encoder, str):
    encoder = keras.saving.load_model(model_path_encoder)
    decoder = keras.saving.load_model(model_path_decoder)
    autoencoder = keras.saving.load_model(model_path_autoencoder)
else:
    raise Exception('nope')

loss = keras.losses.MeanSquaredError(
        reduction="sum_over_batch_size",
        name="mean_squared_error"
    )
autoencoder.compile(optimizer='adam',
                    loss=loss)

train_model=True
if train_model:
    checkpoint_filepath = f'{checkpoints_dir}/checkpoint.model.keras'

    mdl_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # tb_callback = keras.callbacks.TensorBoard(
    #     log_dir=models_dir,
    #     histogram_freq=1,
    #     write_graph=False,
    #     write_images=True,
    #     write_steps_per_second=True,
    #     update_freq="epoch",
    #     profile_batch=0,
    #     embeddings_freq=0,
    #     embeddings_metadata=None,
    # )

    epochs = 1
    batch_size = 50
    shuffle = True
    tic = time.time()
    hist = autoencoder.fit(x=train_data,
                           y=train_data,
                           epochs=epochs,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           validation_data=(test_data, test_data),
                           callbacks=[mdl_callback]
                           )
    toc = time.time()
    print(f'total training time: {(toc-tic)/60}m')
    
    # save models
    autoencoder.save(model_path_autoencoder)
    encoder.save(model_path_encoder)
    decoder.save(model_path_decoder)

print('create predictions')
predictions = autoencoder.predict(test_data)
encoded_data = encoder.predict(test_data)
decoded_data = decoder.predict(encoded_data)

original = sys.stdout
with open(log_file, 'w') as f:
    sys.stdout = f
    print(autoencoder.summary(100))
    sys.stdout = original

# FACTORIZE THIS:
# Analysis

id = 100
plt.close('all')
fig = plt.figure(figsize=(13, 15))

flow_vmin = -1
flow_vmax = 1
plt.subplot(4,3,1)
resid = scaler_Rs.inverse_transform(test_data[id,:,:,0]\
                                    .reshape(1,-1))\
                 .reshape(Nlat, Nlon)
xr_LR = scaler_LR.inverse_transform(test_data[id,:,:,1]\
                                    .reshape(1,-1))\
                 .reshape(Nlat, Nlon)
xr_HR_true = xr_LR + resid

plt.imshow(xr_HR_true, vmin=flow_vmin, vmax = flow_vmax, cmap='RdBu')
plt.gca().invert_yaxis()

plt.subplot(4,3,2)
resid = scaler_Rs.inverse_transform(predictions[id,:,:,0]\
                                    .reshape(1,-1))\
                 .reshape(Nlat, Nlon)
xr_LR = scaler_LR.inverse_transform(predictions[id,:,:,1]\
                                    .reshape(1,-1))\
                 .reshape(Nlat, Nlon)
xr_HR_pred = xr_LR + resid
plt.imshow(xr_HR_pred,vmin=flow_vmin,vmax=flow_vmax,cmap='RdBu')
plt.gca().invert_yaxis()

plt.subplot(4,3,3)
diff=xr_HR_true-xr_HR_pred
h = plt.imshow(diff,cmap='RdBu')
plt.gca().invert_yaxis()
plt.gca().set_title(f'{np.linalg.norm(diff.reshape(-1))}')
plt.pause(1)

plt.subplot(4,3,4)
plt.imshow(test_data[id,:,:,0],)
plt.gca().invert_yaxis()

plt.subplot(4,3,5)
plt.imshow(predictions[id,:,:,0])
plt.gca().invert_yaxis()

plt.subplot(4,3,6)
diff = test_data[id,:,:,0]-predictions[id,:,:,0]
plt.imshow(diff)
plt.gca().set_title(f'{np.linalg.norm(diff.reshape(-1))}')
plt.gca().invert_yaxis()

plt.subplot(4,3,7)
plt.imshow(test_data[id,:,:,1],)
plt.gca().invert_yaxis()

plt.subplot(4,3,8)
plt.imshow(predictions[id,:,:,1])
plt.gca().invert_yaxis()

plt.subplot(4,3,9)
diff = test_data[id,:,:,1]-predictions[id,:,:,1]
plt.imshow(diff)
plt.gca().set_title(f'{np.linalg.norm(diff.reshape(-1))}')
plt.gca().invert_yaxis()

plt.subplot(4,3,10)
plt.imshow(encoded_data[id,:,:,0],)
plt.gca().invert_yaxis()

plt.subplot(4,3,11)
plt.imshow(encoded_data[id,:,:,1])
plt.gca().invert_yaxis()

plt.tight_layout()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
fig_name = f'{results_dir}/results_autoencoder_{timestamp}.png'
print(fig_name)
plt.savefig(fig_name)

fig_name = f'{results_dir}/history_{timestamp}.png'
plt.close('all')
plt.plot(hist.history['loss'],'.-', label='loss')
plt.plot(hist.history['val_loss'],'.-', label='validation loss')
plt.legend()
plt.gca().set_xlabel('epoch')
print(fig_name)
plt.savefig(fig_name)
