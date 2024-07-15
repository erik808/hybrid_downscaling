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
reload(dm)

new_experiment=False
if new_experiment:
    create_model_from_scratch=True
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    # experiment_id = 'testing'
else:
    create_model_from_scratch=False
    experiment_id = '20240715_165252'

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

data_HR = scaler_HR.fit_transform(dm.da_HR.values.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon,1)

data_LR = scaler_LR.fit_transform(dm.da_LR.values.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon,1)

plt.close('all')

Nt, Nlat, Nlon, Nchannels = data_HR.shape
split = int(Nt*4/5)
train_range = range(0, split)
test_range = range(split, Nt)

# stacked_data = np.concatenate([data_HR, data_LR], axis=3)
stacked_data = data_HR

train_data = stacked_data[train_range,:,:,:]
test_data = stacked_data[test_range,:,:,:]

# create mask to be used in network
mask = torch.tensor(dm.mask.values)[None,:,:,None]

# clean memory
del stacked_data, data_HR, data_LR, dm

## Build an autoencoder with Keras using the functional API
keras.utils.clear_session(free_memory=True)

model_path_autoencoder = f'{models_dir}/autoencoder_res.keras'
model_path_encoder = f'{models_dir}/encoder_res.keras'
model_path_decoder = f'{models_dir}/decoder_res.keras'

# create custom masking class
@keras.saving.register_keras_serializable(name="custom_masking")
class Masking(layers.Layer):
    def __init__(self, mask, **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.mask = mask

    def get_config(self):
        config = super(Masking, self).get_config()
        config.update({
            'mask' : keras.saving.serialize_keras_object(self.mask)})
        return config

    @classmethod
    def from_config(cls, config):
        mask_config = config.pop("mask")
        mask = keras.saving.deserialize_keras_object(mask_config)
        return cls(mask, **config)

    def call(self, inputs):
        return ops.multiply(inputs, self.mask)

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
    cropped = layers.Cropping2D(cropping=((2,1),(2,1)))(d3)
    masking_layer = Masking(mask)
    decoded = masking_layer(cropped)

    decoder = Model(encoded, decoded, name="decoder")
    decoder.summary(60)

    autoencoder = Model(state_input, decoded, name="autoencoder")
    autoencoder.summary(60)

elif isinstance(model_path_encoder, str):
    encoder = keras.models.load_model(model_path_encoder)
    decoder = keras.models.load_model(model_path_decoder)
    autoencoder = keras.models.load_model(model_path_autoencoder)
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

    epochs = 100
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

# write log
original = sys.stdout
with open(log_file, 'w') as f:
    sys.stdout = f
    print(autoencoder.summary(100))
    sys.stdout = original




# FACTORIZE THIS:
# Analysis

id = 100
plt.close('all')
fig = plt.figure(figsize=(15, 13))
shrinkf=0.6

plt.subplot(3,3,1)
xr_HR_true = scaler_HR.inverse_transform(test_data[id,:,:,0]\
                                             .reshape(1,-1))\
                          .reshape(Nlat, Nlon)

h = plt.imshow(xr_HR_true, cmap='RdBu')
plt.colorbar(h, shrink=shrinkf)
plt.gca().invert_yaxis()

plt.subplot(3,3,2)
xr_HR_pred = scaler_HR.inverse_transform(predictions[id,:,:,0]\
                                         .reshape(1,-1))\
                      .reshape(Nlat, Nlon)
h = plt.imshow(xr_HR_pred, cmap='RdBu')
plt.colorbar(h, shrink=shrinkf)
plt.gca().invert_yaxis()

plt.subplot(3,3,3)
diff=xr_HR_true-xr_HR_pred
h = plt.imshow(diff,cmap='RdBu')
plt.colorbar(h, shrink=shrinkf)
plt.gca().invert_yaxis()
plt.gca().set_title(f'{np.linalg.norm(diff.reshape(-1))}')
plt.pause(1)

plt.subplot(3,3,4)
h = plt.imshow(test_data[id,:,:,0],)
plt.colorbar(h, shrink=shrinkf)
plt.gca().invert_yaxis()

plt.subplot(3,3,5)
h = plt.imshow(predictions[id,:,:,0])
plt.colorbar(h, shrink=shrinkf)
plt.gca().invert_yaxis()

plt.subplot(3,3,6)
diff = test_data[id,:,:,0]-predictions[id,:,:,0]
h = plt.imshow(diff)
plt.colorbar(h, shrink=shrinkf)
plt.gca().set_title(f'{np.linalg.norm(diff.reshape(-1))}')
plt.gca().invert_yaxis()

plt.subplot(3,3,7)
h = plt.imshow(encoded_data[id,:,:,0],)
plt.colorbar(h, shrink=shrinkf)
plt.gca().invert_yaxis()


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
fig_name = f'{results_dir}/results_autoencoder_{timestamp}.png'
print(fig_name)

plt.tight_layout()
plt.savefig(fig_name)

fig_name = f'{results_dir}/history_{timestamp}.png'
plt.close('all')
plt.plot(hist.history['loss'],'.-', label='loss')
plt.plot(hist.history['val_loss'],'.-', label='validation loss')
plt.grid()
plt.legend()
plt.gca().set_xlabel('epoch')
print(fig_name)

plt.tight_layout()
plt.savefig(fig_name)
