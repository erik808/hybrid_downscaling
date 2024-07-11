from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


from sklearn.preprocessing import MinMaxScaler

import keras
from keras import layers
from keras import ops
from keras.models import Model

import data_manager as dm

Nt, Nlat, Nlon = dm.da_HR.shape

scaled_range = (0,1)

scaler = MinMaxScaler(feature_range=scaled_range)
data = scaler.fit_transform(dm.da_HR.values.reshape(Nt, -1))\
             .reshape(Nt, Nlat, Nlon,1)

# plt.close('all')
# plt.imshow(data[0,:,:]);
# plt.gca().invert_yaxis();
# plt.pause(1)

Nt, Nlat, Nlon, Nchannels = data.shape
split = int(Nt*4/5)
train_range = range(0, split)
test_range = range(split, Nt)

train_data = data[train_range,:,:]
test_data = data[test_range,:,:]

## Build an autoencoder with Keras using the functional API
keras.utils.clear_session(free_memory=True)
create_model=False
model_path_autoencoder = 'models/autoencoder.keras'
model_path_encoder = 'models/encoder.keras'
model_path_decoder = 'models/decoder.keras'
if create_model:

    state_input = layers.Input(shape=(Nlat, Nlon, 1),
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
    d3 = layers.Conv2D(1, (3,3), activation="sigmoid",
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

train_model = False
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

print('predict')    
predictions = autoencoder.predict(test_data)
encoded_data = encoder.predict(test_data)
decoded_data = decoder.predict(encoded_data)

id = 100
plt.close()
plt.subplot(2,2,1)
plt.imshow(test_data[id,:,:,0])
plt.gca().invert_yaxis()

plt.subplot(2,2,2)
plt.imshow(predictions[id,:,:,0])
plt.gca().invert_yaxis()

plt.subplot(2,2,3)
plt.imshow(encoded_data[id,:,:,0])
plt.gca().invert_yaxis()

plt.subplot(2,2,4)
plt.imshow(decoded_data[id,:,:,0])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.pause(1)
print('\a')
breakpoint()
