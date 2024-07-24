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
compute_training_data=False
orig_data, params, scaler, enc_data = \
    dm.create_training_data(compute_training_data, encoder)


class ESN_interface():

    def __init__(self, orig_data, enc_data, hyperparams,
                 encoder=None, decoder=None):

        _, self.enclat, self.enclon, self.filters = \
            enc_data['test']['LR'].shape

        self.orig_test_data = orig_data['test']['LR']
        
        self.encoder = encoder
        self.decoder = decoder

        self.reshape_order = hyperparams['external']['reshape_order']
        test_length = hyperparams['external']['test_length']

        self.T_train = len(orig_data['train']['time'])
        self.T_test = np.min([len(orig_data['test']['time']), test_length])

        # Reshape train and test data
        # !! reshape_order: 'C' make most sense as it clusters spatial
        # !! information from the different channels
        self.xHR_train = enc_data['train']['HR']\
            .reshape(self.T_train, -1, order=self.reshape_order)
        self.xLR_train = enc_data['train']['LR']\
            .reshape(self.T_train, -1, order=self.reshape_order)
        self.xHR_test = enc_data['test']['HR'][:self.T_test,]\
            .reshape(self.T_test, -1, order=self.reshape_order)
        self.xLR_test = enc_data['test']['LR'][:self.T_test,]\
            .reshape(self.T_test, -1, order=self.reshape_order)

        self.N_feats_orig = self.xHR_train.shape[1]

        # Remove zero columns
        self.nonzero_ids = np.where(np.sum(self.xHR_train, axis=0)!=0)[0]
        self.xHR_train = self.xHR_train[:,self.nonzero_ids]
        self.xLR_train = self.xLR_train[:,self.nonzero_ids]
        self.xLR_test = self.xLR_test[:,self.nonzero_ids]        
        
        self.model_type = hyperparams['external']['model_type']

        self.history = hyperparams['external']['training_length']
        self.control_amp = hyperparams['external']['control_amp']

        if (self.model_type == 'DMDc' or
            self.model_type == 'ESNc'):
            # !! another hyperparameter
            self.trainU = np.hstack((self.xHR_train[-self.history:-1,] ,
                                self.xLR_train[-self.history+1:,] * self.control_amp))

        elif (self.model_type == 'DMD' or
              self.model_type == 'ESN'):
            self.trainU = self.xHR_train[-self.history:-1,]

        elif self.model_type == 'corr_only':
            raise Exception('not implemented')
             # self.trainU = X_LR[train_range_p,:]

        self.trainY = self.xHR_train[-self.history+1:,]

        if (self.model_type == 'DMD' or
            self.model_type == 'DMDc' or
            self.model_type == 'corr_only'):
            hyperparams['internal']['dmdMode'] = True
        else:
            hyperparams['internal']['dmdMode'] = False

        if (self.model_type == 'DMD' or
            self.model_type == 'DMDc' or
            self.model_type == 'corr_only' or
            self.model_type == 'ESNc'):
            hyperparams['internal']['feedThrough'] = True
        else:
            hyperparams['internal']['feedThrough'] = False

        if self.model_type == 'ESNc':
            N_feats = self.xHR_train.shape[1]
            hyperparams['internal']['ftRange'] = range(N_feats,
                                                       2*N_feats)

        self.esn = ESN(hyperparams['internal']['Nr'],
                       self.trainU.shape[1],
                       self.trainY.shape[1])
        self.esn.setPars(hyperparams['internal'])
        self.esn.initialize()
        self.hyperparams = hyperparams
        # initialization is done

    def train(self):
        self.esn.train(self.trainU,
                       self.trainY)
        self.esn_state = self.esn.X[-1,:].copy()

    def create_predictions(self):

        predY = np.zeros((self.T_test, self.N_feats_orig))

        # esn state
        sk = self.esn.X[-1,:].copy()

        # initialization:
        xk = self.xHR_train[-1,]

        dec_pred = self.hyperparams['external']['decode_pred']

        verbosity = 400
        for i in range(self.T_test):
            # from data:
            Pxk = self.xLR_test[i,]
            if dec_pred:
                Pxk_dec = np.expand_dims(self.orig_test_data[i,], axis=0)
            else:
                Pxk_dec = None
                
            xk, sk, yk = self.step(xk, Pxk, sk, Pxk_dec)
            predY[i,self.nonzero_ids] = yk

            if not i % verbosity:
                print(f'{i} / {self.T_test}, decoding predictions: {dec_pred}')


        Y = predY.reshape(self.T_test, self.enclat, self.enclon,
                          self.filters, order=self.reshape_order)
        X = self.xHR_test.reshape(self.T_test, self.enclat, self.enclon,
                             self.filters, order=self.reshape_order)

        MSE = np.mean(np.sum(np.square(X-Y),axis=(1,2,3)))
        RMSE = np.sqrt(MSE)
        return Y, X, RMSE

    def step(self, xk, Pxk, sk, Pxk_dec=None):
        if (self.model_type == 'DMDc' or
            self.model_type == 'ESNc' ):
            u_in = np.append(xk.squeeze(),
                             Pxk.squeeze() * self.control_amp)
        elif (self.model_type == 'DMD' or
              self.model_type == 'ESN'):
            u_in = xk.squeeze()

        elif self.model_type == 'corr_only':
            raise Exception('not implemented')
            # u_in = Pyk.squeeze()

        u_in  = np.expand_dims(u_in, axis=0)
        u_in  = self.esn.scaleInput(u_in)
        sk    = self.esn.update(sk, u_in)
        u_out = self.esn.apply(sk, u_in)
        u_out = np.expand_dims(u_out, axis=0)
        yk    = self.esn.unscaleOutput(u_out)

        xk = yk.copy()
        pY = np.zeros((1, self.N_feats_orig))
        pY[0,self.nonzero_ids] = yk

        dec_pred = self.hyperparams['external']['decode_pred']
        if dec_pred:
            assert self.decoder != None, \
                "give a decoder when decode_pred = True"
            assert self.encoder != None, \
                "give an encoder when decode_pred = True"
            assert len(Pxk_dec) > 0, \
                "give a full predictor state"

            full_yk = pY[0,:].reshape(1,self.enclat,
                                      self.enclon,self.filters)            
            inputs = [full_yk, Pxk_dec]
            yk_dec = self.decoder.predict(inputs, verbose=0)
            xk_enc = self.encoder.predict(yk_dec, verbose=0)\
                                 .reshape(1, -1, order=self.reshape_order)
            xk[:] = xk_enc[0,self.nonzero_ids]

        return xk, sk, yk


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
hyperparams = { 'external' : {'model_type'      : 'ESNc',
                              'training_length' : 25000,
                              'repetitions'     : 2,
                              'test_length'     : 4*24*10,
                              'reshape_order'   : 'C',
                              'decode_pred'     : True,
                              'control_amp'     : 1 },

                'internal' : { 'Nr'                 : 15000,
                               'scalingType'        : 'none',
                               'rhoMax'             : 1.2,
                               'alpha'              : 0.7,
                               'avgDegree'          : 100,
                               'entriesPerRow'      : 100,
                               'noiseAmplitude'     : 0.1,
                               'tikhonov_lambda'    : 10,
                               'squaredStates'      : 'even',
                               'reservoirStateInit' : 'zero',
                               'inputMatrixType'    : 'balancedSparse',
                               'fCutoff'            : 0.0,
                               'Wconstruction'      : 'avgDegree'} }


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tuning_id = 'Nr_tuning'
reload_tuning = True
do_gridsearch = True
study_name = tuning_id
tuning_dir = f'{ae_esn_dir}/{tuning_id}/'

storage = f'sqlite:///{tuning_dir}/storage.db'
trial_dump = f'{tuning_dir}/optuna_{timestamp}.dump'
tuningplots_dir = f'{tuning_dir}/plots/'

os.system(f'mkdir -p {tuning_dir}')
os.system(f'mkdir -p {tuningplots_dir}')

with open(trial_dump, "w") as file:
    print('\n', file=file)
    print('All hyperparams: ', file=file)
    print(hyperparams, file=file)

def objective(trial):

    hyperparams['internal']['Nr'] = \
        trial.suggest_int('Nr', 1000, 30000)

    Y, X, RMSE_list, corr_list, RSE_list = \
        train_and_test_wrapper(orig_data, enc_data, hyperparams,
                               encoder, decoder)
    log_and_plot(trial, Y, X, RMSE_list, corr_list, RSE_list)

    mn_RMSE = np.mean(RMSE_list)
    mn_corr = np.mean(corr_list)

    print(f'mean RMSE: {mn_RMSE}, mean correlation: {1-mn_corr}')
    with open(trial_dump, "a") as file:
        print(f'mean RMSE: {mn_RMSE}, mean correlation: {1-mn_corr}', file=file)

    return mn_RMSE

if do_gridsearch:
    search_space = {
        # 'decode_pred' : [False, True],
        # 'Nr' : [10e3, 12e3, 15e3, 18e3]
        'Nr' : [2e3, 4e3, 8e3],
        # 'avgDegree' : [5, 10, 50, 100, 500, 1000, 5000]
        # "training_length": [1e3, 5e3, 10e3, 15e3, 20e3, 25e3],
        # "alpha" : [1.3, 1.5, 1.8],
        # "tikhonov_lambda" : [1e-2, 1e-1, 1, 1e1, 1e2]
        # "noiseAmplitude": [0.0, 0.1, 0.4],
        # "rhoMax" : [0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    }
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space),
                                direction="minimize",
                                storage=storage,
                                load_if_exists=reload_tuning,
                                study_name=study_name)

    study.optimize(objective, timeout=60*20)
else:
    study = optuna.create_study(direction="minimize",
                                storage=storage,
                                study_name=study_name,
                                load_if_exists=reload_tuning)

    study.optimize(objective, n_trials=100)

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
