from importlib import reload

import os
import sys
import dill

from datetime import datetime
from tabulate import tabulate
import time
import xarray as xr

import numpy as np
import keras
from keras import ops
from keras.src.losses.loss import squeeze_or_expand_to_same_rank

import optuna
import data_manager
reload(data_manager)

from data_manager import DataManager
from data_manager import DataGenerator

import plot_utils
reload(plot_utils)

import ae_model
reload(ae_model)

import esn_interface
reload(esn_interface)

from ae_model import AutoEncoder
from ae_model import TriggerESN
from ae_model import CustomValidation
from plot_utils import PlotMachine
from esn_interface import ESN_embedded

# from sklearn.preprocessing import MinMaxScaler

import compute_tool
reload(compute_tool)
from compute_tool import ComputeTool

#-------------------------------------------------------
#-------------------------------------------------------

class AE_Experiment():

    def __init__(self, existing_model=None, exp_name=None,
                 tuning_config=None,
                 detide=False,
                 compute_data=False,
                 coarsening_method='gaussian_filter',
                 truncation=1000,
                 sigma=[1,1,1],
                 feedthrough_type='hybrid'):

        self.init_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = exp_name
        self.tuning_config = tuning_config
        self.do_gridsearch = True
        self.feedthrough_type = feedthrough_type

        self.folder_id = self.init_timestamp \
            if self.exp_name == None else self.exp_name
        self.folder_id = self.exp_name
        self.folder_postfix = ''\
            if self.tuning_config == None else f'-{self.tuning_config}'

        # setup new or existing directories
        self.dm = DataManager()
        self.dirs, self.files = \
            self.dm.setup_directories(self.folder_id, self.folder_postfix)

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
            self.dm.create_training_data(compute_data=compute_data,
                                         detide=detide,
                                         coarsening_method=coarsening_method,
                                         sigma=sigma,
                                         truncation=truncation)
        # -------------------------------------------------------
        self.trial_id = None

        # default hyperparams
        self.hyper_params = {
            'history' : 'all',
            'lookback' : 7,
            'conv_layers_per_block' : 2,
            'future' : 400,
            'noise_stddev' : 0.04,
            'dropout_rate' : 0.0,
            'optimizer' : 'adam',
            'L2_lambda' : 0.0,
            'epochs' : 4,
            'batch_size' : 4,
            'learning_rate' : 0.002,
            'num_filters' : 32,
            'num_filters_exp' : 32,
            'num_filters_red' : 8,
            'inner_stride' : 2,
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
                    'search_space' : [20] },
                'layers_per_block' : {
                    'type' : 'int',
                    'args' : {'name' : 'conv_layers_per_block',
                              'low'  : 1,
                              'high' : 6},
                    'search_space' : [4] },
                'num_filters_red' : {
                    'type' : 'int',
                    'args' : {'name' : 'num_filters_red',
                              'low'  : 1,
                              'high' : 100},
                    'search_space' : [8] },
                'lookback' : {
                    'type' : 'int',
                    'args' : {'name' : 'lookback',
                              'low'  : 0,
                              'high' : 9},
                    'search_space' : [0,1,2,3,4,5,6,7,8,9] },
                'batch_size' : {
                    'type' : 'int',
                    'args' : {'name':'batch_size',
                              'low':1,
                              'high':100},
                    'search_space' : [4] } },

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


        self.ct=ComputeTool()


    def run_optuna_study(self):
        self.init_log()

        tuning_dir = self.dirs['tuning']
        storage = f'sqlite:///{tuning_dir}/storage.db'
        reload_tuning=True
        timeout=60*60*4 # 6h
        n_trials=1

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

        self.study.optimize(self.objective,
                            n_trials=n_trials,
                            timeout=timeout)

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

    def build_and_run_model(self,
                            predict_only=False,
                            evaluate=True,
                            alternative_control=None):

        # AE-MODEL CONFIG
        use_feedthrough = True if (self.feedthrough_type == 'hybrid' or
                                   self.feedthrough_type == 'only') else False
        feedthrough_only = (self.feedthrough_type == 'only')

        print(f'feedthrough_type: {self.feedthrough_type}')
        print(f'  - use_feedthrough: {use_feedthrough}')
        print(f'  - feedthrough_only: {feedthrough_only}')

        # disabled for now
        use_embedded_ESN = False
        if feedthrough_only: use_embedded_ESN = False

        # DATA CONFIG
        self.history = self.hyper_params['history']
        self.future = self.hyper_params['future']

        # TRAINING CONFIG
        epochs = self.hyper_params['epochs']
        batch_size = self.hyper_params['batch_size']
        esn_train_in_epochs = [0,2,4,8]
        shuffle = True

        self.setup_ranges(self.params)

        # input data
        train_data_inp = self.data['HR'][self.train_range_km1,]
        # output data
        train_data_otp = self.data['HR'][self.train_range_k,]
        # control/feedthrough data
        train_data_ft  = self.data['LR'][self.train_range_k,]
        train_time_ft  = self.data['time'][self.train_range_k,]

        # HR test data
        test_data      = self.data['HR'][self.test_range,]
        # LR/control/feedthrough test data
        test_data_ft   = self.data['LR'][self.test_range,]
        test_time      = self.data['time'][self.test_range,]

        if alternative_control == 'coarse_model':
            x_train = self.dm.get_coarse_data(train_time_ft, interpolate=True)
            x_test = self.dm.get_coarse_data(test_time, interpolate=True)
            train_data_ft = \
                self.scalers['R']\
                    .fit_transform(x_train.reshape(len(train_time_ft),-1))\
                    .reshape(x_train.shape)
            test_data_ft = \
                self.scalers['R']\
                    .transform(x_test.reshape(len(test_time),-1))\
                    .reshape(x_test.shape)


        esn_params = esn_interface.hyperparams

        mdir = self.dirs['models']
        postfix, timestamp = self.create_postfix()

        if self.load_existing_model:
            autoencoder = \
                keras.models.load_model(self.load_path_autoencoder)

            encoder = keras.models.load_model(self.load_path_encoder)
            decoder = keras.models.load_model(self.load_path_decoder)

            esn_params['external']['bypass_mode'] = not use_embedded_ESN
            esn = ESN_embedded(esn_params=esn_params)
        else:

            esn_params['external']['bypass_mode'] = not use_embedded_ESN
            esn = ESN_embedded(esn_params=esn_params)
            ae = AutoEncoder(test_vec = self.data['HR'][0,:,:,:],
                             mask = self.params['mask'],
                             log_file = self.files['log'] + f'{postfix}',
                             esn = esn,
                             lookback = self.hyper_params['lookback'])

            model_pars = {
                'use_feedthrough':use_feedthrough,
                'feedthrough_only':feedthrough_only,
                'feedthrough_type':'multiply',
                'learning_rate':self.hyper_params['learning_rate'],
                'conv_layers_per_block':\
                self.hyper_params['conv_layers_per_block'],
                'optimizer':self.hyper_params['optimizer'],
                'L2_lambda':self.hyper_params['L2_lambda'],
                'dropout_rate':self.hyper_params['dropout_rate'],
                'noise_stddev':self.hyper_params['noise_stddev'],
                'num_filters':self.hyper_params['num_filters'],
                'num_filters_red':self.hyper_params['num_filters_red'],
                'num_filters_exp':self.hyper_params['num_filters_exp'],
                'inner_stride':self.hyper_params['inner_stride']
            }

            autoencoder, encoder, decoder =  ae.build_model(**model_pars)


        # print a summary
        autoencoder.summary()
        # save dot and png
        model_png_file = f'{mdir}/autoencoder{postfix}.png'
        keras.utils.plot_model(autoencoder, to_file=model_png_file,
                               show_shapes=True, rankdir='TB',
                               dpi=200, show_layer_activations=False,
                               show_layer_names=True)

        print('----------------------------------------------------------')
        print(f'experiment: {self.folder_id}{self.folder_postfix},       ')
        print(f'model: {postfix}                                         ')
        print('--------------------------------------------------------- ')

        tic = time.time()
        lookback = self.hyper_params['lookback']
        datagen_train = DataGenerator(
            x = [train_data_inp, train_data_ft],
            y = [train_data_otp],
            ft_type = self.feedthrough_type,
            batch_size=batch_size,
            shuffle=shuffle,
            lookback=lookback
        )

        esn_callback = TriggerESN(esn,
                                  train_in_epochs=esn_train_in_epochs,
                                  num_samples=self.params['train_range'].stop)


        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  trial_id=self.trial_id)

        self.validation_callback = \
            CustomValidation(data = self.data,
                             test_inds = self.test_range,
                             plotmachine=plotmachine,
                             pars = {'feedthrough_only': feedthrough_only,
                                     'use_feedthrough': use_feedthrough,
                                     'predict_only' : predict_only,
                                     'evaluate' : evaluate,
                                     'lookback' : lookback})

        callbacks = [esn_callback, self.validation_callback]

        # TRAINING --------------------------------------------
        self.hist = autoencoder.fit(x=datagen_train,
                                    epochs=epochs,
                                    callbacks=callbacks)

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


    def setup_ranges(self, params):

        if self.history == 'all':  # use all data we have
            self.history = params['train_range'].stop
        if self.future == 'all':  # use all data we have
            self.future = params['test_range'].stop

        # setup ranges
        full_train_range = np.arange(params['train_range'].start,
                                     params['train_range'].stop)

        self.train_range_km1 = full_train_range[:-1,][-self.history:,]
        self.train_range_k = full_train_range[1:,][-self.history:,]

        full_test_range = np.arange(params['test_range'].start,
                                    params['test_range'].stop)
        self.test_range = full_test_range[:self.future,]


    def my_loss(self, y_true, y_pred):

        # usage:
        # autoencoder = \
        #     keras.models.load_model(self.load_path_autoencoder,
        #                             compile=False)
        # autoencoder.compile(loss=self.my_loss)

        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)

        def compute_2d_energy_spectrum(tensor):
            im = ops.zeros_like(tensor[...,0]) # imaginary part
            s_u = ops.fft2((tensor[...,0], im))
            s_v = ops.fft2((tensor[...,1], im))
            u = ops.square(ops.sqrt(ops.square(s_u[0]) +
                                    ops.square(s_u[1])))
            v = ops.square(ops.sqrt(ops.square(s_v[0]) +
                                    ops.square(s_v[1])))
            E = (u + v)/2
            E = E / ops.max(E)

            return E

        s_true = compute_2d_energy_spectrum(y_true)
        s_pred = compute_2d_energy_spectrum(y_pred)

        epsilon = 1e-10
        bias=0.0
        first_log = ops.log(ops.maximum(s_true, epsilon) + bias)
        second_log = ops.log(ops.maximum(s_pred, epsilon) + bias)
        out = ops.mean(ops.square(first_log - second_log), axis=(1,2))
        return out


    def plot_spectra(self):
        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  trial_id=self.trial_id)

        data_dict = {
            'truth'  : self.data['HR'][self.test_range,],
            'lowres' : self.data['LR'][self.test_range,],
            'pred'   : self.validation_callback.predictions,
            'scaler_truth' : self.scalers['HR'],
            'scaler_lowres' : self.scalers['LR'],
            'time'   : self.data['time'][self.test_range,]
        }

        self.spec_along = \
            plotmachine.plot_energy_spectrum(transect_name='along_flow',
                                             data=data_dict)

        plotmachine.plot_enstrophy_spectrum(transect_name='along_flow',
                                            data=data_dict)


        self.spec_across = \
            plotmachine.plot_energy_spectrum(transect_name='across_flow',
                                             data=data_dict)
        plotmachine.plot_enstrophy_spectrum(transect_name='across_flow',
                                            data=data_dict)


    def plot_history(self):
        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  trial_id=self.trial_id)
        plotmachine.plot_history(self.hist)


    def create_movie(self, alternative_control=False):

        Nlon = self.params['Nlon']
        Nlat = self.params['Nlat']
        num_channels = self.params['num_channels']

        truth = self.data['HR'][self.test_range,]
        lowres = self.data['LR'][self.test_range,]
        test_time = self.data['time'][self.test_range,]

        if alternative_control == 'coarse_model':
            x = self.dm.get_coarse_data(test_time, interpolate=True)
            lowres = self.scalers['LR']\
                         .transform(x.reshape(len(test_time),-1))\
                         .reshape(x.shape)


        predictions = self.validation_callback.predictions

        vort_truth = self.ct.vorticity(truth, self.scalers['HR'])
        vort_truth_fn = lambda i : vort_truth[i,]

        vort_pred = self.ct.vorticity(predictions,
                                 self.scalers['HR'])
        vort_pred_fn = lambda i : vort_pred[i,]

        vort_lowres = self.ct.vorticity(lowres,
                                   self.scalers['LR'])
        vort_lowres_fn = lambda i : vort_lowres[i,]

        error = np.sqrt(np.square(vort_truth - vort_pred))
        error_fn = lambda i : error[i,]

        def get_rolling_spec(field, window_size=4*12):
            return xr.DataArray(field,
                                dims=['time','wavenumber'],
                                coords={'time' : test_time})\
                     .rolling(time=window_size).mean()

        vort_max = 20
        vort_min = -vort_max

        plot_instructions = {
            'Truth' : {'values' : vort_truth_fn,
                       'type' : '2d',
                       'vmin' : vort_min,
                       'vmax' : vort_max,
                       'cmap' : 'RdBu',
                       'cbar_label' : 'vorticity (day$^{-1}$)'},

            'Prediction' : {'values' : vort_pred_fn,
                            'type' : '2d',
                            'vmin' : vort_min,
                            'vmax' : vort_max,
                            'cmap' : 'RdBu',
                            'cbar_label' : 'vorticity (day$^{-1}$)'},

            'Low resolution' : {'values' : vort_lowres_fn,
                                'type' : '2d',
                                'vmin' : vort_min,
                                'vmax' : vort_max,
                                'cmap' : 'RdBu',
                                'cbar_label' : 'vorticity (day$^{-1}$)'},

            'Error' : {'values' : error_fn,
                       'type' : '2d',
                       'vmin' : 0,
                       'vmax' : vort_max/3,
                       'cmap' : 'viridis',
                       'cbar_label' : 'vorticity (day$^{-1}$)'},


            'spectrum along flow' :
            {
                'type' : '1d',
                'values' :
                {'HR truth' : lambda i :
                 get_rolling_spec(self.spec_along['truth'])[i,:],
                 'Model prediction' : lambda i :
                 get_rolling_spec(self.spec_along['pred'])[i,:],
                 'LR forcing' : lambda i :
                 get_rolling_spec(self.spec_along['lowres'])[i,:]
                 },
                'ymin' : 1e-6,
                'ymax' : 2,
                'xmin' : 1,
                'xmax' : 55,
            },

            'spectrum across flow' :
            {
                'type' : '1d',
                'values' :
                {'HR truth' : lambda i :
                 get_rolling_spec(self.spec_across['truth'])[i,:],
                 'Model prediction' : lambda i :
                 get_rolling_spec(self.spec_across['pred'])[i,:],
                 'LR forcing' : lambda i :
                 get_rolling_spec(self.spec_across['lowres'])[i,:]
                 },
                'ymin' : 1e-6,
                'ymax' : 2,
                'xmin' : 1,
                'xmax' : 55,
            } }

        import plot_utils
        reload(plot_utils)
        from plot_utils import PlotMachine

        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  movie_dir=self.dirs['movies'],
                                  time_array=test_time,
                                  trial_id=self.trial_id)

        plotmachine.create_movie(plot_instructions)

    def create_postfix(self):

        postfix = ''
        if self.trial_id != None:
            postfix += f'_trial_{self.trial_id}'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        postfix += f'_{timestamp}'

        return postfix, timestamp

if __name__=="__main__":
    exp = AE_Experiment(exp_name='rnn_test',
                        tuning_config='default',
                        detide=False,
                        compute_data=False,
                        coarsening_method='gaussian_filter',
                        truncation=100,
                        sigma=[1,1.5,1.5],
                        feedthrough_type='hybrid')
    
    exp.run_optuna_study()
    exp.create_movie()
