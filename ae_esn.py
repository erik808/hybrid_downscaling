import os
import sys
from datetime import datetime
from tabulate import tabulate

import time
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import asciichartpy as acp

import keras
import optuna

import data_manager as dm
reload(dm)

import esn_interface
reload(esn_interface)
from esn_interface import ESN_interface

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

# get training data and metadata
compute_training_data=False
orig_data, params, scaler, enc_data = \
    dm.create_training_data(compute_training_data, encoder)


def train_and_test_wrapper(orig_data, enc_data, hyperparams,
                           encoder=None, decoder=None):
    RMSE_list = []
    corr_list = []
    RSE_list = []

    repetitions = hyperparams['external']['repetitions']
    for rep in range(repetitions):
        print(f'repetition {rep} / {repetitions-1}')
        ESNint = ESN_interface(orig_data, enc_data, hyperparams,
                               encoder, decoder)
        ESNint.train()
        Y,X,_ = ESNint.create_predictions()

        # compute correlation
        Xs = np.sum(np.square(X),axis=(1,2,3))
        Ys = np.sum(np.square(Y),axis=(1,2,3))
        corr_list.append(1-np.corrcoef(Xs,Ys)[1,0])

        # compute RMSE
        SE = np.sum(np.square(X-Y),axis=(1,2,3)).tolist()
        # scaling puts focus on initial errors
        scaling = 2-np.arange(len(SE))*1/(len(SE)-1)
        # scaling = 1./np.sqrt(np.arange(1,len(SE)+1)/4/24)
        # scaling = scaling / np.max(scaling)
        SE_scaled = SE * scaling
        RSE_list.append(np.sqrt(SE[:70]).tolist())
        RMSE_list.append(np.sqrt(np.mean(SE_scaled)))

    return Y, X, RMSE_list, corr_list, RSE_list

def log_and_plot(trial, Y, X, RMSE_list, corr_list, RSE_list):

    Xs = np.sum(np.square(X),axis=(1,2,3))
    Ys = np.sum(np.square(Y),axis=(1,2,3))
    SE = np.sum(np.square(X-Y),axis=(1,2,3)).tolist()

    with open(trial_dump, "a") as file:
        print('\n', file=file)
        print(f'Trial {trial.number}', file=file)
        print(f'Params: {trial.params}', file=file)

    if np.max(np.sqrt(RSE_list)) < 20:
        print(acp.plot(RSE_list), {'height':12})
        with open(trial_dump, "a") as file:
            print(acp.plot(RSE_list, {'height':12}), file=file)
            print(acp.plot([Xs[:70].tolist(),
                            Ys[:70].tolist()], {'height':12}), file=file)

    inputs = [Y[-2:,], orig_data['test']['LR'][Y.shape[0]-2:Y.shape[0],]]
    print('decoding final prediction')
    D_Y = decoder.predict(inputs)[-1,]

    inputs = [X[-2:,], orig_data['test']['LR'][X.shape[0]-2:X.shape[0],]]
    print('decoding final prediction')
    D_X = decoder.predict(inputs)[-1,]

    fignm = f'{tuningplots_dir}/trial_{trial.number}.png'
    #
    create_plots = True
    if create_plots:
        figsize=(11,7)
        fig = plt.figure(figsize=figsize)
        plt.subplot(2,2,1)
        h=plt.imshow(D_X[:,:,0], cmap='viridis')
        plt.colorbar(h)
        plt.gca().invert_yaxis()
        plt.subplot(2,2,2)
        h=plt.imshow(D_Y[:,:,0], cmap='viridis')
        plt.colorbar(h)
        plt.gca().invert_yaxis()
        plt.subplot(2,2,3)
        h=plt.imshow(D_X[:,:,0]-D_Y[:,:,0], cmap='viridis')
        plt.colorbar(h)
        plt.gca().invert_yaxis()
        plt.subplot(2,2,4)
        plt.plot(np.sqrt(SE))
        plt.plot(np.asarray(RSE_list).T)
        plt.grid()
        plt.gca().set_ylim([0,20])
        plt.tight_layout()
        print(fignm)
        plt.savefig(fignm)

# A collection of parameters, divided between parameters external to
# the ESN and ESN internals.
hyperparams = esn_interface.hyperparams

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tuning_id = 'Tikhonov_2'
reload_tuning = True
do_gridsearch = False
study_name = tuning_id
tuning_dir = f'{ae_esn_dir}/{tuning_id}/'

storage = f'sqlite:///{tuning_dir}/storage.db'
timeout = 60*60*2 # in seconds
trial_dump = f'{tuning_dir}/optuna_{timestamp}.dump'
tuningplots_dir = f'{tuning_dir}/plots/'

os.system(f'mkdir -p {tuning_dir}')
os.system(f'mkdir -p {tuningplots_dir}')

with open(trial_dump, "w") as file:
    print('\n', file=file)
    print('All hyperparams: ', file=file)
    print(hyperparams, file=file)

def objective(trial):

    # hyperparams['internal']['Nr'] = \
    #     trial.suggest_int('Nr', 1000, 30000)
    hyperparams['internal']['tikhonov_lambda'] = \
        trial.suggest_float('tikhonov_lambda', 1e-6, 1e6, log=True)
    hyperparams['internal']['rhoMax'] = \
        trial.suggest_float('rhoMax', 1e-2, 2, log=False)

    Y, X, RMSE_list, corr_list, RSE_list = \
        train_and_test_wrapper(orig_data, enc_data, hyperparams,
                               encoder, decoder)
    log_and_plot(trial, Y, X, RMSE_list, corr_list, RSE_list)

    mn_RMSE = np.mean(RMSE_list)
    mn_corr = np.mean(corr_list)

    print(f'mean RMSE: {mn_RMSE}, mean correlation: {1-mn_corr}')
    with open(trial_dump, "a") as file:
        print(f'mean RMSE: {mn_RMSE}, mean correlation: {1-mn_corr}',
              file=file)

    return mn_RMSE

if do_gridsearch:
    search_space = {
        # 'decode_pred' : [False, True],
        'Nr' : [10e3, 12e3, 15e3, 18e3],
        # 'avgDegree' : [5, 10, 50, 100, 500, 1000, 5000]
        # "training_length": [1e3, 5e3, 10e3, 15e3, 20e3, 25e3],
        # "alpha" : [1.3, 1.5, 1.8],
        "tikhonov_lambda" : [1e-2, 1e-1, 1, 1e1, 1e2],
        # "noiseAmplitude": [0.0, 0.1, 0.4],
        # "rhoMax" : [0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    }
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(sampler=sampler,
                                direction="minimize",
                                storage=storage,
                                load_if_exists=reload_tuning,
                                study_name=study_name)

    study.optimize(objective, timeout=timeout)
else:
    study = optuna.create_study(direction="minimize",
                                storage=storage,
                                study_name=study_name,
                                load_if_exists=reload_tuning)

    study.optimize(objective, timeout=timeout)

    
study_log = f'{tuning_dir}/optuna_{timestamp}.log'
print(f'writing log to {study_log}')
with open(study_log, "w") as file:
    print(tabulate(study.trials_dataframe(),
                   headers='keys',
                   tablefmt='orgtbl'), file=file)
    print('best trials:', file=file)
    print(study.best_trials, file=file)

print('best trials:')
print(study.best_trials)



# inputs = [Y, orig_data['test']['LR'][:self.T_test,]]
# print('decoding predictions')
# D = decoder.predict(inputs)

# plt.close('all')
# figsize=(11,9)
# fig = plt.figure(figsize=figsize)
# plt.subplot(3,2,1)
# tid = 300
# chn = 3
# t_mse = self.T_test
# h=plt.imshow(X[tid,:,:,chn], cmap='binary')
# plt.colorbar(h)
# plt.gca().invert_yaxis()
# plt.subplot(3,2,3)
# h=plt.imshow(Y[tid,:,:,chn], cmap='binary')
# plt.colorbar(h)
# plt.gca().invert_yaxis()
# plt.subplot(3,2,5)
# # h=plt.imshow(X[tid,:,:,chn]-Y[tid,:,:,chn], cmap='binary')
# # plt.colorbar(h)
# # plt.gca().invert_yaxis()
# MSE = np.sqrt(np.sum(np.square(X-Y),axis=(1,2,3)))
# plt.plot(MSE[:t_mse])
# plt.gca().set_ylim([0,8])
# plt.grid()

# plt.subplot(3,2,2)
# Z = orig_data['test']['HR'][:self.T_test,]
# h=plt.imshow(Z[tid,:,:,0], cmap='viridis')
# plt.colorbar(h)
# plt.gca().invert_yaxis()
# plt.subplot(3,2,4)
# h=plt.imshow(D[tid,:,:,0], cmap='viridis')
# plt.colorbar(h)
# plt.gca().invert_yaxis()
# plt.subplot(3,2,6)
# # h=plt.imshow(Z[tid,:,:,0]-D[tid,:,:,0], cmap='viridis')
# # plt.colorbar(h)
# # plt.gca().invert_yaxis()
# MSE = np.sqrt(np.sum(np.square(Z-D),axis=(1,2,3)))
# plt.plot(MSE[:t_mse])
# plt.gca().set_ylim([0,7])

# plt.grid()

# plt.tight_layout()
# figname = f'{ae_esn_dir}/results_{timestamp}.png'
# print(figname)
# plt.savefig(figname)
# plt.pause(1)
