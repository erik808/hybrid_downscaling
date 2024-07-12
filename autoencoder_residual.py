from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import keras
from keras import layers
from keras import ops
from keras.models import Model

import data_manager as dm

# assume everything has this shape
Nt, Nlat, Nlon = dm.da_HR.shape

scaled_range = (0,1)

scaler_HR = MinMaxScaler(feature_range=scaled_range)
scaler_LR = MinMaxScaler(feature_range=scaled_range)
scaler_Rs = MinMaxScaler(feature_range=scaled_range)
# scaler_HR = StandardScaler()
# scaler_LR = StandardScaler()
# scaler_Rs = StandardScaler()

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

# plt.subplot(1,3,1)
# plt.imshow(data_HR[0,:,:]);
# plt.gca().invert_yaxis();
# plt.subplot(1,3,2)
# plt.imshow(data_LR[0,:,:]);
# plt.gca().invert_yaxis();
# plt.subplot(1,3,3)
# plt.imshow(data_Rs[0,:,:]);
# plt.gca().invert_yaxis();
# plt.tight_layout()
# plt.pause(1)

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
create_model=True
model_path_autoencoder = 'models/autoencoder_res.keras'
model_path_encoder = 'models/encoder_res.keras'
model_path_decoder = 'models/decoder_res.keras'
if create_model:
    num_channels = train_data.shape[-1]
    state_input = layers.Input(shape=(Nlat, Nlon, num_channels),
                               name="full_state_input")
    # Encoding layers
    e1 = layers.Conv2D(32, (3,3), activation="relu",
                       padding="same")(state_input)
    e2 = layers.MaxPooling2D((2,2),
                             padding="same")(e1)
    e3 = layers.Conv2D(32, (3,3), activation="relu",
                       padding="same")(e2)
    encoded = layers.MaxPooling2D((2,2),
                                  padding="same")(e3)

    encoder = Model(state_input, encoded, name="encoder")
    encoder.summary(60)

    # Decoder
    d1 = layers.Conv2DTranspose(32, (3,3), strides=2, activation="relu",
                                padding="same")(encoded)
    d2 = layers.Conv2DTranspose(32, (3,3), strides=2, activation="relu",
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
    epochs = 10
    batch_size = 50
    shuffle = True
    autoencoder.fit(
        x=train_data,
        y=train_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_data=(test_data, test_data)
        )

    # save models
    autoencoder.save(model_path_autoencoder)
    encoder.save(model_path_encoder)
    decoder.save(model_path_decoder)

print('create predictions')
predictions = autoencoder.predict(test_data)
encoded_data = encoder.predict(test_data)
decoded_data = decoder.predict(encoded_data)

id = 100
flow_vmin = -1
flow_vmax = 1
plt.close()
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
plt.pause(1)
print('\a')
breakpoint()
