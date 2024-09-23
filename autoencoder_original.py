from importlib import reload

import os
import sys
import dill

from datetime import datetime
from tabulate import tabulate
import time

import numpy as np
import keras

import optuna

import plot_utils
reload(plot_utils)
import data_manager as dm
reload(dm)
import ae_model
reload(ae_model)
import esn_interface
reload(esn_interface)

from ae_model import AutoEncoder
from ae_model import TriggerESN
from ae_model import CustomValidation
from plot_utils import PlotMachine
from esn_interface import ESN_embedded

#-------------------------------------------------------
#-------------------------------------------------------

class AE_Experiment():

    def __init__(self, existing_model=None, exp_name=None,
                 tuning_config=None, detide=False, compute_data=False):

        self.init_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = exp_name
        self.tuning_config = tuning_config
        self.do_gridsearch = True

        self.folder_id = self.init_timestamp \
            if self.exp_name == None else self.exp_name
        self.folder_id = self.exp_name
        self.folder_postfix = ''\
            if self.tuning_config == None else f'-{self.tuning_config}'

        # setup new or existing directories
        self.dirs, self.files = \
            dm.setup_directories(self.folder_id, self.folder_postfix)

        if existing_model == None:
            self.load_existing_model = False
        else:
            self.load_existing_model = True
            self.load_model_folder = existing_model['folder']
            self.load_model_postfix = existing_model['postfix']

        if self.load_existing_model:
            mdir = self.load_model_folder
            self.load_path_autoencoder = \
                f'{mdir}/autoencoder_{self.load_model_postfix}.keras'
            self.load_path_encoder     = \
                f'{mdir}/encoder_{self.load_model_postfix}.keras'
            self.load_path_decoder     = \
                f'{mdir}/decoder_{self.load_model_postfix}.keras'

        # -------------------------------------------------------
        # Load or compute data
        self.data, self.params, self.scalers, _ = \
            dm.create_training_data(compute_data=compute_data,
                                    detide=detide)
        # -------------------------------------------------------
        self.trial_id = None

        # default hyperparams
        self.hyper_params = {
            'history' : 'all',
            'use_skip_connections' : False,
            'conv_layers_per_block' : 2,
            'future' : 'all',
            #'future' : 400,
            'noise_stddev' : 0.04,
            'dropout_rate' : 0.0,
            'optimizer' : 'adam',
            'L2_lambda' : 0.0,
            'epochs' : 4,
            'batch_size' : 4,
            'learning_rate' : 0.002,
            'num_filters' : 32,
            'num_filters_exp' : 32,
            'num_filters_red' : 9,
            'inner_stride' : 1,
        }

        ## maybe read from ini or xml instead?
        self.tuning_config_dict = {
            #-------------------------------------------------------
            'default' : {
                'epochs' : {
                    'type' : 'int',
                    'args' : {'name':'epochs',
                              'low': 1,
                              'high':50},
                    'search_space' : [4] },
                'layers_per_block' : {
                    'type' : 'int',
                    'args' : {'name' : 'conv_layers_per_block',
                              'low'  : 1,
                              'high' : 6},
                    'search_space' : [2] } },

            #-------------------------------------------------------
            'regularization' : {
                'L2_lambda' : {
                    'type' : 'float',
                    'args' : {'name' : 'L2_lambda',
                              'low'  : 0,
                              'high' : 1e2},
                    'search_space' : [0, 1e-8, 1e-7, 1e-6, 1e-5] } },
            #-------------------------------------------------------
            'training_pars' : {
                'L2_lambda' : {
                    'type' : 'float',
                    'args' : {'name' : 'L2_lambda',
                              'low'  : 0,
                              'high' : 1e2},
                    'search_space' : [0, 1e-10, 1e-8, 1e-6] },
                'learning_rate' : {
                    'type' : 'float',
                    'args' : {'name':'learning_rate',
                              'low':1e-4,
                              'high':1e-2},
                    'search_space' : [2e-3, 4e-3, 6e-3, 8e-3, 1e-2] },
                'batch_size' : {
                    'type' : 'int',
                    'args' : {'name':'batch_size',
                              'low':1,
                              'high':100},
                    'search_space' : [1,2,4,8] } },
            #-------------------------------------------------------
            'filters' : {}, #TODO
            'dropout' : {}, #TODO
            'skip_connections' : {}, # TODO
            'layers_per_block' : {}, # TODO
            #-------------------------------------------------------
        }


    def run_optuna_study(self):
        self.init_log()

        tuning_dir = self.dirs['tuning']
        storage = f'sqlite:///{tuning_dir}/storage.db'
        reload_tuning=True
        timeout=60*60*4 # 6h

        self.setup_search_space()

        if self.do_gridsearch:
            sampler = self.gridSampler
        else:
            sampler = optuna.samplers.TPESampler()

        self.study = \
            optuna.create_study(sampler=sampler,
                                direction="minimize",
                                storage=storage,
                                study_name=f'{self.exp_name}_{self.tuning_config}',
                                load_if_exists=reload_tuning)

        self.study.optimize(self.objective, timeout=timeout)

    def objective(self, trial):
        self.trial_id = trial._trial_id-1
        self.setup_search_space(trial)
        self.log(trial)
        err = self.build_and_run_model()
        return err

    def hyper_param_helper(self, vartype: str, suggest_args={},
                           search_space=[], trial=None):

        var = suggest_args['name']
        trial_mode = True if trial != None else False
        suggest_fun = getattr(trial, f'suggest_{vartype}') \
            if trial_mode else []

        if trial_mode:
            self.hyper_params[var] = suggest_fun(**suggest_args)
        self.search_space[var] = search_space


    def setup_search_space(self, trial=None):

        assert self.tuning_config in self.tuning_config_dict, \
            f"invalid tuning config: {self.tuning_config}"

        # initialize search space
        self.search_space = {}
        tuning_config = self.tuning_config_dict[self.tuning_config]
        for key, item in tuning_config.items():
            self.hyper_param_helper(
                item['type'],
                item['args'],
                search_space=item['search_space'],
                trial=trial)

        self.gridSampler = \
            optuna.samplers.GridSampler(self.search_space)


    def init_log(self):
        tuning_dir = self.dirs['tuning']
        self.study_log = f'{tuning_dir}/optuna_{self.init_timestamp}.log'
        with open(self.study_log, "w") as file:
            print('study log ______________________________' ,
                  file=file)


    def log(self, trial):
        print(f'writing log to {self.study_log}')
        with open(self.study_log, "a") as file:

            for out in [file, sys.stdout]:

                print(f'trial_id: {self.trial_id}', file=out)
                print(f'trial_params: {trial.params}', file=out)
                print(f'best_trials: {self.study.best_trials}', file=out)

                print(tabulate(self.study.trials_dataframe(),
                               headers='keys',
                               tablefmt='orgtbl'), file=out)

    def build_and_run_model(self, predict_only=False):
        # AE-MODEL CONFIG
        use_feedthrough = True
        feedthrough_only = False
        use_embedded_ESN = False

        # DATA CONFIG
        self.history = self.hyper_params['history']
        self.future = self.hyper_params['future']

        # TRAINING CONFIG
        epochs = self.hyper_params['epochs']
        batch_size = self.hyper_params['batch_size']
        esn_train_in_epochs = [0,2,4,8]
        shuffle = True

        if self.history == 'all':  # use all data we have
            self.history = self.data['train']['HR'].shape[0]
        if self.future == 'all':  # use all data we have
            self.future = self.data['test']['HR'].shape[0]

        # input data
        train_data_inp = self.data['train']['HR'][:-1,][-self.history:,]
        # output data
        train_data_otp = self.data['train']['HR'][1:,][-self.history:,]
        # control/feedthrough data
        train_data_ft  = self.data['train']['LR'][1:,][-self.history:,]

        # HR test data
        test_data      = self.data['test']['HR'][:self.future,]
        # LR/control/feedthrough test data
        test_data_ft   = self.data['test']['LR'][:self.future,]
        test_time      = self.data['test']['time'][:self.future,]

        mask = self.params['mask']
        Nt   = self.params['Nt']
        Nlon = self.params['Nlon']
        Nlat = self.params['Nlat']
        num_channels = self.params['num_channels']

        esn_params = esn_interface.hyperparams

        if feedthrough_only: use_embedded_ESN = False

        mdir = self.dirs['models']
        postfix, timestamp = self.create_postfix()
        if self.load_existing_model:

            autoencoder = keras.models.load_model(self.load_path_autoencoder)
            encoder = keras.models.load_model(self.load_path_encoder)
            decoder = keras.models.load_model(self.load_path_decoder)

            if use_embedded_ESN:
                esn = autoencoder.get_layer('esn_embedded')
                # overwrite parameters

                num_samples = train_data_inp.shape[0]
                timeids = np.arange(num_samples)
                timetns = np.expand_dims(timeids, axis=[1,2,3])
                print('create training data for embedded ESN')
                esn.setPars(esn_params, num_samples=num_samples)
                values  = esn.pixel_shuffle(encoder.predict([train_data_inp, timetns]))
                control = esn.pixel_shuffle(encoder.predict([train_data_ft, timetns]))
                esn.initialize(values, control)
                esn.populate_storage(values, timeids, control)
            else:
                esn_params['external']['bypass_mode'] = True
                esn = ESN_embedded(esn_params=esn_params)
        else:

            esn_params['external']['bypass_mode'] = not use_embedded_ESN
            esn = ESN_embedded(esn_params=esn_params)

            ae = AutoEncoder(test_vec=train_data_inp[0,:,:,:],
                             mask=mask,
                             log_file=self.files['log'] + f'{postfix}',
                             esn=esn)

            autoencoder, encoder, decoder = \
                ae.build_model(
                    use_feedthrough=use_feedthrough,
                    feedthrough_only=feedthrough_only,
                    feedthrough_type='multiply',
                    learning_rate=self.hyper_params['learning_rate'],
                    conv_layers_per_block=self.hyper_params['conv_layers_per_block'],
                    use_skip_connections=self.hyper_params['use_skip_connections'],
                    optimizer=self.hyper_params['optimizer'],
                    L2_lambda=self.hyper_params['L2_lambda'],
                    dropout_rate=self.hyper_params['dropout_rate'],
                    noise_stddev=self.hyper_params['noise_stddev'],
                    num_filters=self.hyper_params['num_filters'],
                    num_filters_red=self.hyper_params['num_filters_red'],
                    num_filters_exp=self.hyper_params['num_filters_exp'],
                    inner_stride=self.hyper_params['inner_stride'],
                )

        # print a summary
        autoencoder.summary()

        print('----------------------------------------------------------')
        print(f'experiment: {self.folder_id}{self.folder_postfix},       ')
        print(f'model: {postfix}                                         ')
        print('--------------------------------------------------------- ')

        tic = time.time()

        # really necessary to expand to 4 dims?
        T_train = np.expand_dims(np.arange(train_data_inp.shape[0]), axis=[1,2,3])
        T_test  = np.expand_dims(np.arange(train_data_inp.shape[0],
                                           train_data_inp.shape[0] +
                                           test_data.shape[0]),
                                 axis=[1,2,3])

        if feedthrough_only:
            X_train = [train_data_ft]
        elif use_feedthrough:
            X_train = [train_data_inp, T_train, train_data_ft]
        else:
            X_train = [train_data_inp, T_train]

        Y_train = train_data_otp

        esn_callback = TriggerESN(esn,
                                  train_in_epochs=esn_train_in_epochs,
                                  num_samples=X_train[0].shape[0])

        # we create a custom validation using a callback at every
        # epoch end
        initial_xk   = np.expand_dims(self.data['train']['HR'][-1,:,:,:], axis=0)
        initial_xkm1 = np.expand_dims(self.data['train']['HR'][-2,:,:,:], axis=0)

        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  trial_id=self.trial_id)

        self.validation_callback = \
            CustomValidation(test_data=(test_data, T_test, test_data_ft),
                             initial_xk=(initial_xk, initial_xkm1),
                             plotmachine=plotmachine,
                             pars = {'feedthrough_only': feedthrough_only,
                                     'use_feedthrough': use_feedthrough,
                                     'predict_only' : predict_only},
                             scalers = self.scalers)

        callbacks = [esn_callback, self.validation_callback]

        # TRAINING --------------------------------------------
        self.hist = autoencoder.fit(x=X_train,
                                    y=Y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    callbacks=callbacks
                                    )
        toc = time.time()
        print(f'total training time: {(toc-tic)/60}m')

        # SAVING -----------------------------------------------
        # save model and metadata
        mdata_file = f'{mdir}/mdata{postfix}.dill'
        container = {'hist' : self.hist,
                     'epochs' : epochs,
                     'batch_size' : batch_size,
                     'shuffle' : shuffle
                     }

        with open(mdata_file, 'wb') as file:
            dill.dump(container, file)

        # save models
        save_path_autoencoder = f'{mdir}/autoencoder{postfix}.keras'
        save_path_encoder     = f'{mdir}/encoder{postfix}.keras'
        save_path_decoder     = f'{mdir}/decoder{postfix}.keras'

        print(f'saving autoencoder to {save_path_autoencoder}')
        print(f'saving encoder to {save_path_encoder}')
        print(f'saving decoder to {save_path_decoder}')
        autoencoder.save(save_path_autoencoder)
        encoder.save(save_path_encoder)
        decoder.save(save_path_decoder)

        self.plot_history()
        self.plot_spectra()

        print(f'final error: {self.validation_callback.final_error}')
        return self.validation_callback.final_error


    def plot_spectra(self):

        test_time  = self.data['test']['time'][:self.future,]
        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  movie_dir=self.dirs['movies'],
                                  time_array=test_time,
                                  trial_id=self.trial_id)

        data_dict = {
            'truth'  : self.data['test']['HR'][:self.future,],
            'lowres' : self.data['test']['LR'][:self.future,],
            'pred'   : self.validation_callback.predictions,
            'scaler' : self.scalers['HR'],
        }

        self.spec_along = plotmachine.plot_spectrum(transect_name='along_flow',
                                                    data=data_dict)
        self.spec_across = plotmachine.plot_spectrum(transect_name='across_flow',
                                                     data=data_dict)

    def plot_history(self):
        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  movie_dir=self.dirs['movies'],
                                  time_array=self.data['test']['time'][:self.future,],
                                  trial_id=self.trial_id)
        plotmachine.plot_history(self.hist)


    def create_movie(self):

        Nlon = self.params['Nlon']
        Nlat = self.params['Nlat']
        num_channels = self.params['num_channels']

        # HR test data
        test_data      = self.data['test']['HR'][:self.future,]
        # LR/control/feedthrough test data
        test_data_ft   = self.data['test']['LR'][:self.future,]
        test_time      = self.data['test']['time'][:self.future,]

        predictions = self.validation_callback.predictions

        # Create dictionary for output visualization
        xr_HR_true_fun = lambda i : \
            self.scalers['HR'].inverse_transform(test_data[i,:,:,:]\
                                                 .reshape(1,-1))\
                              .reshape(Nlat, Nlon, num_channels)

        # instant kinetic energy
        Kt_HR_true_fun = lambda i : \
            np.sqrt(np.square(xr_HR_true_fun(i)).sum(axis=2))

        xr_HR_pred_fun = lambda i : \
            self.scalers['HR'].inverse_transform(predictions[i,:,:,:]\
                                                 .reshape(1,-1))\
                              .reshape(Nlat, Nlon, num_channels)

        # instant kinetic energy
        Kt_HR_pred_fun = lambda i : \
            np.sqrt(np.square(xr_HR_pred_fun(i)).sum(axis=2))

        # Create dictionary for output visualization
        xr_LR_true_fun = lambda i : \
            self.scalers['HR'].inverse_transform(test_data_ft[i,:,:,:]\
                                                 .reshape(1,-1))\
                              .reshape(Nlat, Nlon, num_channels)

        # instant kinetic energy
        Kt_LR_true_fun = lambda i : \
            np.sqrt(np.square(xr_LR_true_fun(i)).sum(axis=2))

        xr_HR_diff_fun = lambda i : xr_HR_true_fun(i) - xr_HR_pred_fun(i)

        Kt_HR_diff_fun = lambda i : Kt_HR_true_fun(i) - Kt_HR_pred_fun(i)

        Rs_true_fun = lambda i : test_data[i,:,:,0] - test_data_ft[i,:,:,0]
        Rs_pred_fun = lambda i : predictions[i,:,:,0] - test_data_ft[i,:,:,0]
        Rs_diff_fun = lambda i : Rs_true_fun(i) - Rs_pred_fun(i)

        vmax = Kt_HR_true_fun(0).max()
        vmin_diff = -0.1
        vmax_diff =  0.1

        plot_instructions = {
            'Kt_HR true' : {'values' : Kt_HR_true_fun,
                            'type' : '2d',
                            'vmin' : 0,
                            'vmax' : vmax,
                            'cmap' : 'viridis'},

            'Kt_HR pred' : {'values' : Kt_HR_pred_fun,
                            'type' : '2d',
                            'vmin' : 0,
                            'vmax' : vmax,
                            'cmap' : 'viridis'},

            'Kt_LR true' : {'values' : Kt_LR_true_fun,
                            'type' : '2d',
                            'vmin' : 0,
                            'vmax' : vmax,
                            'cmap' : 'viridis'},

            'res true' : {'values' : Rs_true_fun,
                          'type' : '2d',
                          'vmin' : vmin_diff,
                          'vmax' : vmax_diff,
                          'cmap' : 'RdBu'},

            'res pred' : {'values' : Rs_pred_fun,
                          'type' : '2d',
                          'vmin' : vmin_diff,
                          'vmax' : vmax_diff,
                          'cmap' : 'RdBu'},

            'error' : {'values' : Kt_HR_diff_fun,
                       'type' : '2d',
                       'vmin' : vmin_diff,
                       'vmax' : vmax_diff,
                       'cmap' : 'RdBu'},

            'spectrum along flow' :
            {
                'type' : '1d',
                'values' :
                {'HR truth' : lambda i : self.spec_along['truth'][i,:],
                 'Model prediction' : lambda i : self.spec_along['pred'][i,:],
                 'LR forcing' : lambda i : self.spec_along['lowres'][i,:]
                 },
                'vmin' : 5e-6,
                'vmax' : 2,
            },

            'spectrum across flow' :
            {
                'type' : '1d',
                'values' :
                {'HR truth' : lambda i : self.spec_across['truth'][i,:],
                 'Model prediction' : lambda i : self.spec_across['pred'][i,:],
                 'LR forcing' : lambda i : self.spec_across['lowres'][i,:]
                 },
                'vmin' : 5e-6,
                'vmax' : 2,
            } }

        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  movie_dir=self.dirs['movies'],
                                  time_array=test_time,
                                  trial_id=self.trial_id)

        # plotmachine.create_transect(plot_instructions)
        plotmachine.plot_single_frame(50, plot_instructions)
        plotmachine.create_movie(plot_instructions)

    def create_postfix(self):

        postfix = ''
        if self.trial_id != None:
            postfix += f'_trial_{self.trial_id}'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        postfix += f'_{timestamp}'

        return postfix, timestamp




if __name__=="__main__":
    exp = AE_Experiment(exp_name='test',
                        tuning_config='default',
                        detide=False,
                        compute_data=False)
    exp.run_optuna_study()
