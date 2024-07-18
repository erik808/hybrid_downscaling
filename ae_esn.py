import os
import sys
from datetime import datetime
import time
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt

import keras

import data_manager as dm
reload(dm)

# base experiment
base_exp = '20240718_164427_tf_multiply'
models_dir = f'experiments/{base_exp}/models'
ae_esn_dir = f'experiments/{base_exp}/ae_esn_experiments'
os.system(f'mkdir -p {ae_esn_dir}')

model_path_encoder = f'{models_dir}/encoder_res.keras'
model_path_decoder = f'{models_dir}/decoder_res.keras'
# load trained encoder and decoder
encoder = keras.models.load_model(model_path_encoder)
decoder = keras.models.load_model(model_path_decoder)

# load uv data
da_HR, da_LR, da_mask = dm.load_uv_data()

# FACTORIZE training data creation:
# data, time, mask = dm.create_training_data()
# train_data_HR = data['train']['HR']
# test_data_HR = data['test']['HR']
# train_data_LR = data['train']['LR']
# test_data_LR = data['test']['LR']
# train_time = time['train']
# test_time = time['test']

# do the assembling into channels here
data_HR_stacked = np.stack([da_HR['uo'].values,
                            da_HR['vo'].values], axis=3)
data_LR_stacked = np.stack([da_LR['uo'].values,
                            da_LR['vo'].values], axis=3)

Nt, Nlat, Nlon, num_channels = data_HR_stacked.shape

# StandardScaler doesnt work that well
scaler_HR = MinMaxScaler(feature_range=scaled_range)
data_HR = scaler_HR.fit_transform(data_HR_stacked.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon, num_channels)
data_LR = scaler_HR.transform(data_LR_stacked.reshape(Nt, -1))\
                   .reshape(Nt, Nlat, Nlon, num_channels)



