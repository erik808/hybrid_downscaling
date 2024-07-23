import os
os.system('export MKL_NUM_THREADS=12')
os.system('export OMP_NUM_THREADS=12')
import sys
from datetime import datetime
from tabulate import tabulate

import time
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import asciichartpy as acp
import dill

import keras
import optuna

import data_manager as dm
reload(dm)

from ESN.ESN import ESN

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
reload_data=True
if reload_data:
    print('Create training data')
    orig_data, params, scalers  = dm.create_training_data()

    print('Create encoded train and test data...')
    enc_data = {}
    for period in ['train', 'test']:
        enc_data[period] = {}
        for resolution in ['HR', 'LR']:
            print(f'{period}-{resolution}')
            enc_data[period][resolution] = \
                encoder.predict(orig_data[period][resolution])

def ESN_train_and_test(orig_data, enc_data, hyperparams):

    reshape_order = hyperparams['external']['reshape_order']
    test_length = hyperparams['external']['test_length']

    T_train = len(orig_data['train']['time'])
    T_test = np.min([len(orig_data['test']['time']), test_length])

    # Reshape train and test data
    # !! reshape_order: 'C' make most sense as it clusters spatial
    # !! information from the different channels
    xHR_train = enc_data['train']['HR']\
        .reshape(T_train, -1, order=reshape_order)
    xLR_train = enc_data['train']['LR']\
        .reshape(T_train, -1, order=reshape_order)
    xHR_test = enc_data['test']['HR'][:T_test,]\
        .reshape(T_test, -1, order=reshape_order)
    xLR_test = enc_data['test']['LR'][:T_test,]\
        .reshape(T_test, -1, order=reshape_order)

    N_feats_orig = xHR_train.shape[1]

    # Remove zero columns
    nonzero_ids = np.where(np.sum(xHR_train, axis=0)!=0)[0]
    xHR_train = xHR_train[:,nonzero_ids]
    xLR_train = xLR_train[:,nonzero_ids]
    xLR_test = xLR_test[:,nonzero_ids]
    N_feats = xHR_train.shape[1]

    # !! TODO factorize this somewhere
    model_type = hyperparams['external']['model_type']

    history = hyperparams['external']['training_length']
    control_amp = hyperparams['external']['control_amp']

    if (model_type == 'DMDc' or
        model_type == 'ESNc'):
        # !! another hyperparameter
        trainU = np.hstack((xHR_train[-history:-1,] ,
                            xLR_train[-history+1:,] * control_amp))

    elif (model_type == 'DMD' or
          model_type == 'ESN'):
        trainU = xHR_train[-history:-1,]

    elif model_type == 'corr_only':
        raise Exception('not implemented')
         # trainU = X_LR[train_range_p,:]

    trainY = xHR_train[-history+1:,]

    if (model_type == 'DMD' or
        model_type == 'DMDc' or
        model_type == 'corr_only'):
        hyperparams['internal']['dmdMode'] = True
    else:
        hyperparams['internal']['dmdMode'] = False

    if (model_type == 'DMD' or
        model_type == 'DMDc' or
        model_type == 'corr_only' or
        model_type == 'ESNc'):
        hyperparams['internal']['feedThrough'] = True
    else:
        hyperparams['internal']['feedThrough'] = False

    if model_type == 'ESNc':
        hyperparams['internal']['ftRange'] = range(N_feats,
                                                   2*N_feats)

    esn = ESN(hyperparams['internal']['Nr'],
              trainU.shape[1],
              trainY.shape[1])

    esn.setPars(hyperparams['internal'])
    esn.initialize()
    esn.train(trainU, trainY)

    # -------------------------------------------------------
    # CREATE PREDICTIONS
    # -------------------------------------------------------
    predY = np.zeros((T_test, N_feats_orig))
    esn_state = esn.X[-1,:].copy()
    print(np.linalg.norm(esn_state))

    # initialization:
    xk = xHR_train[-1,]

    verbosity = 400
    for i in range(T_test):

        # from data:
        Pxk = xLR_test[i,]

        if (model_type == 'DMDc' or
            model_type == 'ESNc' ):
            u_in = np.append(xk.squeeze(),
                             Pxk.squeeze() * control_amp)
        elif (model_type == 'DMD' or
              model_type == 'ESN'):
            u_in = xk.squeeze()

        elif model_type == 'corr_only':
            raise Exception('not implemented')
            # u_in = Pyk.squeeze()

        u_in       = np.expand_dims(u_in, axis=0)
        u_in       = esn.scaleInput(u_in)
        esn_state  = esn.update(esn_state, u_in)
        u_out      = esn.apply(esn_state, u_in)
        u_out      = np.expand_dims(u_out, axis=0)
        yk         = esn.unscaleOutput(u_out)

        xk = yk
        predY[i,nonzero_ids] = yk
        if not i % verbosity:
            print(f'{i} / {T_test}')

    _, enclat, enclon, filters = enc_data['test']['LR'].shape
    Y = predY.reshape(T_test, enclat, enclon,
                      filters, order=reshape_order)
    X = xHR_test.reshape(T_test, enclat, enclon,
                         filters, order=reshape_order)

    MSE = np.mean(np.sum(np.square(X-Y),axis=(1,2,3)))
    RMSE = np.sqrt(MSE)
    return Y, X, RMSE


# A collection of parameters, divided between parameters external to
# the ESN and ESN internals.
hyperparams = { 'external' : {'model_type'      : 'ESN',
                              'training_length' : 8000,
                              'repetitions'     : 2,
                              'test_length'     : 4*24*5, # 10 days
                              'reshape_order'   : 'C',
                              'control_amp'     : 1 },

                'internal' : { 'Nr'                 : 1000,
                               'scalingType'        : 'none',
                               'rhoMax'             : 0.4,
                               'alpha'              : 0.1,
                               'avgDegree'          : 100,
                               'entriesPerRow'      : 50,
                               'noiseAmplitude'     : 0.4,
                               'tikhonov_lambda'    : 1e-6,
                               'squaredStates'      : 'even',
                               'reservoirStateInit' : 'zero',
                               'inputMatrixType'    : 'balancedSparse',
                               'fCutoff'            : 0.0,
                               'Wconstruction'      : 'entriesPerRow'} }


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tuning_dir = f'{ae_esn_dir}/optuna_{timestamp}/'
tuningplots_dir = f'{ae_esn_dir}/optuna_{timestamp}/plots/'
os.system(f'mkdir -p {tuning_dir}')
os.system(f'mkdir -p {tuningplots_dir}')

def objective(trial):

    # hyperparams['internal']['tikhonov_lambda'] = \
    #     trial.suggest_float('tikhonov_lambda', 1e-7, 1e-2, log=True)

    # hyperparams['internal']['fCutoff'] = \
    #     trial.suggest_float('fCutoff', 1e-12, 1e-1, log=True)

    hyperparams['external']['model_type'] = \
        trial.suggest_categorical('model_type', ['ESN', 'ESNc'])

    hyperparams['internal']['scalingType'] = \
        trial.suggest_categorical('scalingType',
                                  ['none', 'minMax1', 'standardize'])

    # hyperparams['internal']['inputMatrixType'] = \
    #     trial.suggest_categorical('inputMatrixType',
    #                               ['balancedSparse',
    #                                'sparse',
    #                                'full',
    #                                'identity'])

    # hyperparams['internal']['reservoirStateInit'] = \
    #     trial.suggest_categorical('reservoirStateInit',
    #                               ['zero', 'random'])

    hyperparams['internal']['rhoMax'] = \
        trial.suggest_float('rhoMax', 0.01, 4, step=0.1)

    # hyperparams['internal']['noiseAmplitude'] = \
    #     trial.suggest_float('noiseAmplitude', 0.0, 1.0, step=0.1)

    # hyperparams['internal']['alpha'] = \
    #     trial.suggest_float('alpha', 0.01, 2, step=0.1)

    # hyperparams['internal']['avgDegree'] = \
    #     trial.suggest_int('avgDegree', 2, 200, step=10)


    RMSE_list = []
    RSE_list = []
    repetitions = hyperparams['external']['repetitions']
    for rep in range(repetitions):
        print(f'repetition {rep} / {repetitions-1}')
        Y,X,_ = ESN_train_and_test(orig_data, enc_data, hyperparams)
        SE = np.sum(np.square(X-Y),axis=(1,2,3)).tolist()
        RSE_list.append(np.sqrt(SE[:70]).tolist())
        RMSE_list.append(np.sqrt(np.mean(SE)))

    print('Errors:')
    if np.max(np.sqrt(RSE_list)) < 20:
        print(acp.plot(RSE_list))

    # Visualize (FaCTORIZE!!)
    inputs = [Y[-2:,], orig_data['test']['LR'][Y.shape[0]-2:Y.shape[0],]]
    print('decoding final prediction')
    D_Y = decoder.predict(inputs)[-1,]

    inputs = [X[-2:,], orig_data['test']['LR'][X.shape[0]-2:X.shape[0],]]
    print('decoding final prediction')
    D_X = decoder.predict(inputs)[-1,]

    SE = np.sum(np.square(X-Y),axis=(1,2,3))

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

    mn_RMSE = np.mean(RMSE_list)

    print(f'mean RMSE: {mn_RMSE}')
    return mn_RMSE

study = optuna.create_study(direction="minimize",
                            study_name="ESN tuning")
study.optimize(objective, n_trials=100)

study_log = f'{tuning_dir}/optuna_{timestamp}.log'
print(f'writing log to {study_log}')
with open(study_log, "w") as file:
    print(tabulate(study.trials_dataframe(),
                   headers='keys',
                   tablefmt='orgtbl'), file=file)
    print('best params:', file=file)
    print(study.best_params, file=file)

print('best params:')
print(study.best_params)

# Y,X,RMSE = ESN_train_and_test(orig_data, enc_data, hyperparams)

# breakpoint()

# inputs = [Y, orig_data['test']['LR'][:T_test,]]
# print('decoding predictions')
# D = decoder.predict(inputs)

# plt.close('all')
# figsize=(11,9)
# fig = plt.figure(figsize=figsize)
# plt.subplot(3,2,1)
# tid = 300
# chn = 3
# t_mse = T_test
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
# Z = orig_data['test']['HR'][:T_test,]
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
