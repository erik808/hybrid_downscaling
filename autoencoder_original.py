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

from multiprocess import Pool

import data_manager as dm
reload(dm)

new_experiment=True
if new_experiment:
    create_model_from_scratch=True
    experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    # experiment_id = 'testing'
else:
    create_model_from_scratch=False
    experiment_id = '20240716_104034'

models_dir = f'experiments/{experiment_id}/models'
results_dir = f'experiments/{experiment_id}/results'
movie_dir = f'experiments/{experiment_id}/movies'
checkpoints_dir = f'experiments/{experiment_id}/checkpoints'

log_file = f'{models_dir}/log.txt'

os.system(f'mkdir -p {models_dir}')
os.system(f'mkdir -p {movie_dir}')
os.system(f'mkdir -p {results_dir}')
os.system(f'mkdir -p {checkpoints_dir}')

# assume everything has this shape
Nt, Nlat, Nlon = dm.da_HR.shape

scaled_range = (0,1)

# StandardScaler doesnt work that well
scaler_HR = MinMaxScaler(feature_range=scaled_range)
data_HR = scaler_HR.fit_transform(dm.da_HR.values.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon,1)
data_LR = scaler_HR.transform(dm.da_LR.values.reshape(Nt, -1))\
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

test_LR = data_LR[test_range,:,:,:]
train_time = dm.da_LR.time.values[train_range]
test_time  = dm.da_LR.time.values[test_range]

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
    masking_layer1 = Masking(mask, name="masking_layer1")
    masking_layer2 = Masking(mask, name="masking_layer2")
    state_input = layers.Input(shape=(Nlat, Nlon, num_channels),
                               name="full_state_input")
    # Encoding layers
    e1 = layers.Conv2D(num_filters, (5,5), strides = (2,2),
                       activation="relu",
                       padding="same")(state_input)
    # e2 = masking_layer1(e1)
    # e3 = layers.MaxPooling2D((2,2),
    #                          padding="same")(e2)
    encoded = layers.Conv2D(num_filters, (5,5), strides = (2,2),
                       activation="relu",
                       padding="same")(e1)

    # encoded = layers.MaxPooling2D((2,2),
    #                               padding="same")(e4)

    encoder = Model(state_input, encoded, name="encoder")
    encoder.summary(60)

    # Decoder
    d1 = layers.Conv2DTranspose(num_filters, (5,5), strides=(2,2), activation="relu",
                                padding="same")(encoded)
    d2 = layers.Conv2DTranspose(num_filters, (5,5), strides=(2,2), activation="relu",
                                padding="same")(d1)
    d3 = layers.Conv2D(num_channels, (3,3), activation="sigmoid",
                       padding="same")(d2)
    cropped = layers.Cropping2D(cropping=((2,1),(2,1)))(d3)

    decoded = masking_layer2(cropped)

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

    epochs = 10
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
encoded_xr_HR_true = encoder.predict(test_data)
encoded_xr_HR_pred = encoder.predict(predictions)
encoded_xr_LR_true = encoder.predict(test_LR)

# write log
original = sys.stdout
with open(log_file, 'w') as f:
    sys.stdout = f
    print(autoencoder.summary(100))
    sys.stdout = original

# Create dictionary for output visualization
xr_HR_true_fun = lambda i : scaler_HR.inverse_transform(test_data[i,:,:,0]\
                                                   .reshape(1,-1))\
                                .reshape(Nlat, Nlon)

xr_HR_pred_fun = lambda i : scaler_HR.inverse_transform(predictions[i,:,:,0]\
                                                        .reshape(1,-1))\
                                     .reshape(Nlat, Nlon)

xr_HR_diff_fun = lambda i : xr_HR_true_fun(i) - xr_HR_pred_fun(i)

Rs_true_fun = lambda i : test_data[i,:,:,0] - test_LR[i,:,:,0]
Rs_pred_fun = lambda i : predictions[i,:,:,0] - test_LR[i,:,:,0]
Rs_diff_fun = lambda i : Rs_true_fun(i) - Rs_pred_fun(i)

enc_xr_HR_true_fun = lambda i : (encoded_xr_HR_true[i,:,:,0])
enc_xr_LR_true_fun = lambda i : (encoded_xr_LR_true[i,:,:,0])
enc_xr_HR_pred_fun = lambda i : (encoded_xr_HR_pred[i,:,:,0])

output_dict = {'xr_HR true' : {'values' : xr_HR_true_fun,
                               'vmin' : -.8,
                               'vmax' : .8, 'cmap' : 'RdBu'},
               'xr_HR pred' : {'values' : xr_HR_pred_fun,
                               'vmin' : -.8,
                               'vmax' : .8, 'cmap' : 'RdBu'},
               'xr_HR diff' : {'values' : xr_HR_diff_fun,
                               'vmin' : -0.2,
                               'vmax' : 0.2, 'cmap' : 'RdBu'},

               'res true' : {'values' : Rs_true_fun,
                             'vmin' : -0.2,
                             'vmax' : 0.2, 'cmap' : 'RdBu'},
               'res pred' : {'values' : Rs_pred_fun,
                             'vmin' : -0.2,
                             'vmax' : 0.2, 'cmap' : 'RdBu'},
               'res dif' : {'values' : Rs_diff_fun,
                            'vmin' : -0.1,
                            'vmax' : 0.1, 'cmap' : 'RdBu'},

               'enc xr_HR true' : {'values' : enc_xr_HR_true_fun,
                                   'vmin' : 0,
                                   'vmax' : .5, 'cmap' : 'viridis'},
               'enc xr_HR pred' : {'values' : enc_xr_HR_pred_fun,
                                   'vmin' : 0,
                                   'vmax' : .5, 'cmap' : 'viridis'},
               'enc xr_LR true' : {'values' : enc_xr_LR_true_fun,
                                   'vmin' : 0,
                                   'vmax' : .5, 'cmap' : 'viridis'},
               }

# FACTORIZE THIS further:
# Analysis
def plot_frame(id, fig_name=None):
    shrinkf=0.5
    plt.clf()
    if fig_name == None:
        fig_name = f'{movie_dir}/frame-{id:06d}.png'

    for f, (key, item) in enumerate(output_dict.items()):
        plt.subplot(3,3,f+1)
        h = plt.imshow(item['values'](id),
                       cmap=item['cmap'],
                       vmin=item['vmin'],
                       vmax=item['vmax'])
        plt.colorbar(h, shrink=shrinkf)
        plt.gca().set_title(key)
        plt.gca().invert_yaxis()

    plt.suptitle(f"date: {np.datetime64(test_time[id], 'h')}")
    print(fig_name)
    plt.savefig(fig_name)

fig = plt.figure(figsize=(15, 13))
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
fig_name = f'{results_dir}/results_autoencoder_{timestamp}.png'
plot_frame(100, fig_name)

fig = plt.figure(figsize=(15, 8))
with Pool(8) as p:
    p.map(plot_frame, range(0,len(test_time),4))

movie_name = f'movie_{timestamp}.mov'
framerate = 24
sys_cmd = ( f"ffmpeg -r {framerate} -f image2 -pattern_type glob -i "
            f"'{movie_dir}/frame-*.png' "
            f"-vcodec libx264 -crf 25  -pix_fmt yuv420p -y "
            f"{movie_dir}/{movie_name}" )

print(sys_cmd)
os.system(sys_cmd)
sys_cmd = ( f"rm {movie_dir}/frame-*.png")
print(sys_cmd)
os.system(sys_cmd)

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
