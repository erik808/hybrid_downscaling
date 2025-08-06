from importlib import reload

import sys
import dill

from datetime import datetime
from tabulate import tabulate
import time
import optuna

import xarray as xr
import numpy as np
import keras
from keras import ops

import data_utils
reload(data_utils)

from data_utils import DataFactory
from data_utils import DataGenerator

import plot_utils
reload(plot_utils)

import tools
reload(tools)
from tools import Tee

import ae_model
reload(ae_model)

from ae_model import CustomValidation
from ae_model import LSModelWrapper
from plot_utils import PlotMachine

import compute_tool
reload(compute_tool)
from ae_model import AutoEncoder
from compute_tool import ComputeTool

import plot_utils
reload(plot_utils)

# -------------------------------------------------------


class AE_Experiment():

    def __init__(
            self,
            existing_model=None,
            exp_name=None,
            tuning_config=None,
            case_study='cmems',
            feedthrough_type='hybrid',
            testing_mode=False
    ):
        tools.load_config(self, config_name='default')

        self.init_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.case_study = case_study
        self.exp_name = exp_name
        self.tuning_config = tuning_config
        self.do_gridsearch = True
        self.feedthrough_type = feedthrough_type

        self.folder_id = self.init_timestamp \
            if self.exp_name is None else self.exp_name
        self.folder_id = self.exp_name
        self.folder_postfix = ''\
            if self.tuning_config is None else f'-{self.tuning_config}'

        # setup new or existing directories
        self.testing_mode = testing_mode
        self.dm = DataFactory(
            case_study=self.case_study,
            testing_mode=self.testing_mode
        )
        self.dirs, self.files = \
            self.dm.setup_directories(self.folder_id, self.folder_postfix)

        if existing_model is None:
            self.load_existing_model = False
        else:
            self.load_existing_model = True
            self.load_model_folder = existing_model['folder']
            self.load_model_postfix = existing_model['postfix']

        if self.load_existing_model:
            mdir = self.load_model_folder
            self.load_path_autoencoder = \
                f'{mdir}/autoencoder_{self.load_model_postfix}.keras'
            self.load_path_encoder = \
                f'{mdir}/encoder_{self.load_model_postfix}.keras'
            self.load_path_decoder = \
                f'{mdir}/decoder_{self.load_model_postfix}.keras'

        # -------------------------------------------------------
        # Load and/or compute data
        self.data, self.params, self.scalers, _ = \
            self.dm.create_training_data()

        # -------------------------------------------------------
        self.ct=ComputeTool(case_study=self.case_study)
        self.trial_id = None

    def run_optuna_study(self):
        self.init_log()

        tuning_dir = self.dirs['tuning']
        storage = f'sqlite:///{tuning_dir}/storage.db'
        reload_tuning=True
        timeout=60 * 60 * 4  # 6h

        self.setup_search_space()

        if self.do_gridsearch:
            sampler = self.gridSampler
        else:
            sampler = optuna.samplers.TPESampler()

        self.study = \
            optuna.create_study(
                sampler=sampler,
                direction="minimize",
                storage=storage,
                study_name=f'{self.exp_name}_{self.tuning_config}',
                load_if_exists=reload_tuning
            )

        self.study.optimize(self.objective,
                            timeout=timeout)

    def objective(self, trial):
        self.trial_id = trial._trial_id - 1
        self.setup_search_space(trial)
        self.log(trial)
        err = self.build_and_run_model()
        return err

    def hyper_param_helper(self, vartype: str, suggest_args={},
                           search_space=[], trial=None):

        var = suggest_args['name']
        trial_mode = True if trial is not None else False
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
            print('study log ______________________________',
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
                            evaluate=True):
        # AE-MODEL CONFIG
        use_feedthrough = True if (self.feedthrough_type == 'hybrid' or
                                   self.feedthrough_type == 'only') else False
        feedthrough_only = (self.feedthrough_type == 'only')

        print(f'feedthrough_type: {self.feedthrough_type}')
        print(f'  - use_feedthrough: {use_feedthrough}')
        print(f'  - feedthrough_only: {feedthrough_only}')

        self.unroll_dim = self.hyper_params['unroll_dim']
        self.latent_space_model = self.hyper_params['latent_space_model']

        # DATA CONFIG
        self.history = self.hyper_params['history']
        self.future = self.hyper_params['future']

        # TRAINING CONFIG
        epochs = self.hyper_params['epochs']
        batch_size = self.hyper_params['batch_size']
        shuffle = True

        self.setup_ranges(self.params)

        # input data
        if self.case_study == 'cmems':
            train_data_inp = self.data['HR'][self.train_range_km1,]
        elif self.case_study == 'swot':
            train_data_inp = self.data['LR'][self.train_range_k,]

        # output data (truth)
        train_data_otp = self.data['HR'][self.train_range_k,]

        # control/feedthrough data
        if self.case_study == 'cmems':
            train_data_ft  = self.data['LR'][self.train_range_k,]
        elif self.case_study == 'swot':  # not used
            train_data_ft  = self.data['LR'][self.train_range_k,]

        self.postfix, self.timestamp = self.create_postfix()
        sys.stdout = Tee(self.files['log'] + f'{self.postfix}')

        ae_model_pars = {
            'use_feedthrough'  : use_feedthrough,
            'feedthrough_only' : feedthrough_only,
            'feedthrough_type' : 'multiply',
        }
        ae_model_pars.update(self.hyper_params)

        ae = AutoEncoder(
            test_vec=self.data['HR'][0, :, :, :],
            mask=self.params['mask'],
            **ae_model_pars
        )

        if self.load_existing_model:
            autoencoder = \
                keras.models.load_model(self.load_path_autoencoder)
            encoder = keras.models.load_model(self.load_path_encoder)
            decoder = keras.models.load_model(self.load_path_decoder)

        else:
            autoencoder, encoder, decoder = ae.build_model()

        print('----------------------------------------------------------')
        print(f'experiment: {self.folder_id}{self.folder_postfix},       ')
        print(f'model: {self.postfix}                                    ')
        print('--------------------------------------------------------- ')

        tic = time.time()
        dgen_args = {'ft_type' : self.feedthrough_type,
                     'batch_size' : batch_size,
                     'shuffle' : shuffle,
                     'lookback' : self.hyper_params['lookback'],
                     'unroll_dim' : self.unroll_dim,
                     'encoder' : encoder,
                     }

        datagen_train = DataGenerator(
            x=[train_data_inp, train_data_ft],
            y=[train_data_otp],
            **dgen_args
        )

        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  trial_id=self.trial_id)

        self.validation_callback = \
            CustomValidation(data=self.data,
                             test_inds=self.test_range,
                             plotmachine=plotmachine,
                             pars={'feedthrough_only' : feedthrough_only,
                                   'use_feedthrough' : use_feedthrough,
                                   'multihead_output' : False,
                                   'unroll_dim' : self.unroll_dim,
                                   'predict_only' : predict_only,
                                   'evaluate' : evaluate,
                                   'lookback' : self.hyper_params['lookback']
                                   },
                             scalers=self.scalers,
                             case_study=self.case_study
                             )

        cdir = self.dirs['checkpoints']

        model_checkpoint_file = \
            f'{cdir}/autoencoder{self.postfix}.checkpoint.keras'

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_file,
            monitor='error',
            mode='min',
            save_best_only=True)

        callbacks = [self.validation_callback,
                     model_checkpoint_callback]

        # Final assembly -------------------------------------
        if (
                not feedthrough_only and
                self.latent_space_model in [
                    'VAE',
                    'VAE+RNN',
                    'RNN',
                    'LSTM',
                    'GRU',
                ]
        ):
            model = LSModelWrapper(
                encoder,
                decoder,
                model=self.latent_space_model
            )
        else:
            # default model
            model = autoencoder

        if self.unroll_dim > 0:
            model.summary()
            model = ae.create_unrolled_model(autoencoder,
                                             self.unroll_dim)

        ae.compiler(model)
        # model.summary()
        # self.plot_model(model)

        # TRAINING --------------------------------------------
        self.hist = model.fit(x=datagen_train,
                              epochs=epochs,
                              callbacks=callbacks)

        toc = time.time()
        print(f'total training time: {(toc-tic)/60}m')

        # SAVING -----------------------------------------------
        # save model and metadata
        mdir = self.dirs['models']
        mdata_file = f'{mdir}/mdata{self.postfix}.dill'
        container = {
            'hist' : self.hist,
            'epochs' : epochs,
            'batch_size' : batch_size,
            'shuffle' : shuffle
        }

        with open(mdata_file, 'wb') as file:
            dill.dump(container, file)

        # save models
        self.save_path_autoencoder = f'{mdir}/autoencoder{self.postfix}.keras'
        self.save_path_encoder     = f'{mdir}/encoder{self.postfix}.keras'
        self.save_path_decoder     = f'{mdir}/decoder{self.postfix}.keras'

        print(f'saving autoencoder to {self.save_path_autoencoder}')
        print(f'saving encoder to {self.save_path_encoder}')
        print(f'saving decoder to {self.save_path_decoder}')
        autoencoder.save(self.save_path_autoencoder)
        encoder.save(self.save_path_encoder)
        decoder.save(self.save_path_decoder)

        self.plot_history()
        self.plot_spectra()

        print(f'final error: {self.validation_callback.final_error}')
        return self.validation_callback.final_error

    def plot_model(self, model):
        mdir = self.dirs['models']
        model_png_file = f'{mdir}/model{self.postfix}.png'
        print(f'see {model_png_file}')
        keras.utils.plot_model(model, to_file=model_png_file,
                               show_shapes=True, rankdir='TB',
                               dpi=200, show_layer_activations=False,
                               show_layer_names=True)

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
            im = ops.zeros_like(tensor[..., 0])  # imaginary part
            s_u = ops.fft2((tensor[..., 0], im))
            s_v = ops.fft2((tensor[..., 1], im))
            u = ops.square(ops.sqrt(ops.square(s_u[0]) +
                                    ops.square(s_u[1])))
            v = ops.square(ops.sqrt(ops.square(s_v[0]) +
                                    ops.square(s_v[1])))
            E = (u + v) / 2
            E = E / ops.max(E)

            return E

        s_true = compute_2d_energy_spectrum(y_true)
        s_pred = compute_2d_energy_spectrum(y_pred)

        epsilon = 1e-10
        bias=0.0
        first_log = ops.log(ops.maximum(s_true, epsilon) + bias)
        second_log = ops.log(ops.maximum(s_pred, epsilon) + bias)
        out = ops.mean(ops.square(first_log - second_log), axis=(1, 2))
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

    def create_movie(self):

        truth = self.data['HR'][self.test_range,]
        lowres = self.data['LR'][self.test_range,]
        test_time = self.data['time'][self.test_range,]

        predictions = self.validation_callback.predictions

        vort_truth = self.ct.vorticity(truth, self.scalers['HR'])

        def vort_truth_fn(i):
            return vort_truth[i,]

        vort_pred = self.ct.vorticity(predictions,
                                      self.scalers['HR'])

        def vort_pred_fn(i):
            return vort_pred[i,]

        vort_lowres = self.ct.vorticity(lowres,
                                        self.scalers['LR'])

        def vort_lowres_fn(i):
            return vort_lowres[i,]

        error = np.sqrt(np.square(vort_truth - vort_pred))

        def error_fn(i):
            return error[i,]

        def get_rolling_spec(field, window_size=4 * 12):
            return xr.DataArray(field,
                                dims=['time', 'wavenumber'],
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
                       'vmax' : vort_max / 3,
                       'cmap' : 'viridis',
                       'cbar_label' : 'vorticity (day$^{-1}$)'},


            'spectrum along flow' :
            {
                'type' : '1d',
                'values' :
                {'HR truth' : lambda i :
                 get_rolling_spec(self.spec_along['truth'])[i, :],
                 'Model prediction' : lambda i :
                 get_rolling_spec(self.spec_along['pred'])[i, :],
                 'LR forcing' : lambda i :
                 get_rolling_spec(self.spec_along['lowres'])[i, :]
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
                 get_rolling_spec(self.spec_across['truth'])[i, :],
                 'Model prediction' : lambda i :
                 get_rolling_spec(self.spec_across['pred'])[i, :],
                 'LR forcing' : lambda i :
                 get_rolling_spec(self.spec_across['lowres'])[i, :]
                 },
                'ymin' : 1e-6,
                'ymax' : 2,
                'xmin' : 1,
                'xmax' : 55,
            }
        }

        plotmachine = PlotMachine(results_dir=self.dirs['results'],
                                  movie_dir=self.dirs['movies'],
                                  time_array=test_time,
                                  trial_id=self.trial_id)

        plotmachine.create_movie(plot_instructions)

    def create_postfix(self):

        postfix = ''
        if self.trial_id is not None:
            postfix += f'_trial_{self.trial_id}'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        postfix += f'_{timestamp}'

        return postfix, timestamp


if __name__=="__main__":

    exp = AE_Experiment(
        exp_name='vae_tests',
        tuning_config='latent_space_dim',
        feedthrough_type='hybrid',
        case_study='cmems',
    )

    exp.run_optuna_study()
