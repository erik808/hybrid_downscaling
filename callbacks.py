import dill
import numpy as np
import keras
import pandas as pd
from keras import ops
import plot_utils
import importlib
import tools
import ESN.ESN as ESN_mod
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


importlib.reload(plot_utils)

plt.switch_backend('Agg')
# if os.environ.get('DISPLAY') is not None:
#     plt.switch_backend('qtagg')


class DMD(keras.callbacks.Callback):

    def __init__(
            self,
            data_gen,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dgen = data_gen

    def on_epoch_begin(self, epoch, logs=None):

        predictor_layer = self.model\
                              .model\
                              .get_layer('latent_predictor')

        # do some checks
        DMDcheck = (len(predictor_layer.weights) == 1 and
                    'W_out' in predictor_layer.weights[0].path)

        if DMDcheck:
            print('ESN/DMD layer detected')
        else:
            print('ESN/DMD inactive')
            return None

        # temp increase batch size
        batch_size = self.dgen.batch_size
        self.dgen.batch_size = batch_size * 100
        num_batches = self.dgen.__len__()
        # unshuffle
        self.dgen.indices = np.sort(self.dgen.indices)

        # create data matrix
        pb_i = keras.utils.Progbar(num_batches, interval=0.5)
        x_enc_mat = []
        x_enc_LR_mat = []
        print('create training data for ESN/DMD using available encoder')
        for b in range(num_batches):
            pb_i.add(1)
            batch_x, batch_y = self.dgen.__getitem__(b)
            x_enc = self.model.encoder(
                ops.nan_to_num(
                    ops.squeeze(
                        batch_x['HR_data'][:,
                                           1,  # target lookback index
                                           ...],
                        axis=1)),
                training=False
            )[0].cpu().detach().numpy()  # take only the mean

            x_enc_mat += [x_enc]
            x_enc_LR_mat += [ops.squeeze(batch_x['LR_data'][:, 0, ...])]

        # decrease batch size again
        self.dgen.batch_size = batch_size

        # shuffle
        if self.dgen.shuffle:
            np.random.shuffle(self.dgen.indices)

        X = np.concatenate(x_enc_mat, 0)
        X = (X.reshape(X.shape[0], -1)).T

        X_LR = np.concatenate(x_enc_LR_mat, 0)
        X_LR = (X_LR.reshape(X_LR.shape[0], -1)).T

        esn_pars = {}
        esn_pars['scalingType'] = 'none'
        esn_pars['dmdMode'] = True
        esn_pars['tikhonov_lambda'] = self.model.lambdaDMD
        esn_pars['feedThrough'] = True
        # esn_pars['ftRange'] = range(0, N)
        esn_pars['fCutoff'] = self.model.cutoffDMD

        if self.model.predictor == 'DMD':
            U = X[:, :-1].T
        elif self.model.predictor == 'DMDc':
            U = np.vstack([X[:, :-1], X_LR[:, :-1]]).T

        Y = X[:, 1:].T

        np.random.seed(1)
        esn = ESN_mod.ESN(100, U.shape[1], Y.shape[1])
        esn.setPars(esn_pars)
        esn.initialize()
        esn.train(U, Y)

        # assign ESN weights to layer
        predictor_layer.set_weights([esn.W_out])

    def on_epoch_end(self, epoch, logs=None):
        return None


class AnalysisBase(keras.callbacks.Callback, ABC):
    def __init__(
            self,
            data_gen,
            plot=[
                'reconstruction',
                'spectra',
            ],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dgen = data_gen

        self.plot_instructions = plot
        self.reconstruction = 'reconstruction' in self.plot_instructions
        self.spectra = ('spectra' in self.plot_instructions or
                        'spectrum' in self.plot_instructions)

        self.plot_machine = \
            plot_utils.PlotMachine(dm=self.dgen.dm)

        # create a nan-mask
        self.mask_constructed = False
        self.cfg_printed = False
        self.history = {}

    @abstractmethod
    def call_model(self, x):
        pass

    @abstractmethod
    def restrict_x(self, x):
        pass

    @abstractmethod
    def restrict_y(self, y):
        pass

    @property
    @abstractmethod
    def scaler_list(self):
        pass

    def construct_mask(self):
        if not self.mask_constructed:
            self.mask = ops.not_equal(self.model.masking.mask, 0.0)
            self.mask = \
                (self.model.masking.mask /
                 ops.cast(self.mask, self.model.masking.mask.dtype)
                 )
            self.mask_constructed = True
        else:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        if not self.cfg_printed:
            self.cfg_printed = \
                tools.print_configuration(self.dgen, self.model)
        return None

    def on_epoch_end(self, epoch, logs=None):
        self.update_history(logs)
        plt.close('all')
        self.plot_machine.create_postfix()
        self.plot_machine.create_results_dir(epoch)
        self.construct_mask()
        if epoch % 1 == 0 or epoch == self.params['epochs'] - 1:
            self.timestepping(
                epoch,
                logs,
                spectra=self.spectra,
                reconstruction=self.reconstruction,
            )

    def update_history(self, logs):
        for key, value in logs.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]

        tools.print_history(self.dgen, self.history)
        self.plot_machine.plot_history(self.history)

    def plot_reconstruction(
            self,
            x,
            y,
            z,
            epoch,
            time=0
    ):
        assert y.shape == z.shape
        x = x[time,]
        y = y[time,]
        z = z[time,]

        def wrapper(update):
            data_template = {
                'data': [],
                'vmin': 0,
                'vmax': 1,
            }
            data_template.update(update)
            return data_template

        err_uo = np.abs(y[..., 0] - z[..., 0])
        err_zos = np.abs(y[..., 2] - z[..., 2])

        plot_dict = {
            'meta': {'epoch': epoch,
                     'time': time,
                     'prefix': 'model_',
                     'subplot_shape': [2, 4]},
            'input uo': wrapper(
                {'data': x[..., 0]}),
            'pred uo': wrapper(
                {'data': z[..., 0]}),
            'truth uo': wrapper(
                {'data': y[..., 0]}),
            'err uo': wrapper(
                {'data': err_uo,
                 'vmin': 0, 'vmax': 0.1}),
            'input zos': wrapper(
                {'data': x[..., 2]}),
            'pred zos': wrapper(
                {'data': z[..., 2]}),
            'truth zos': wrapper(
                {'data': y[..., 2]}),
            'err zos': wrapper(
                {'data': err_zos,
                 'vmin': 0, 'vmax': 0.1}),
        }
        self.plot_machine.plot_reconstructions(plot_dict)

        # if x.shape != z.shape:
        #     x_bilin = np.ascontiguousarray(x.transpose((2, 0, 1)))
        #     x_bilin = self.dgen.dm.bilin_upsampler(x_bilin)\
        #                           .transpose((1, 2, 0))

        #     err_uo = np.abs(y[..., 0] - x_bilin[..., 0])
        #     err_zos = np.abs(y[..., 2] - x_bilin[..., 2])

        #     plot_dict = {
        #         'meta': {'epoch': epoch,
        #                  'time': time,
        #                  'prefix': 'bilin_',
        #                  'subplot_shape': [2, 4]},
        #         'input uo': wrapper(
        #             {'data': x[..., 0]}),
        #         'bilin uo': wrapper(
        #             {'data': x_bilin[..., 0]}),
        #         'truth uo': wrapper(
        #             {'data': y[..., 0]}),
        #         'err uo': wrapper(
        #             {'data': err_uo,
        #              'vmin': 0, 'vmax': 0.1}),
        #         'input zos': wrapper(
        #             {'data': x[..., 2]}),
        #         'bilin zos': wrapper(
        #             {'data': x_bilin[..., 2]}),
        #         'truth zos': wrapper(
        #             {'data': y[..., 2]}),
        #         'err zos': wrapper(
        #             {'data': err_zos,
        #              'vmin': 0, 'vmax': 0.1}),
        #     }
        #     self.plot_machine.plot_reconstructions(plot_dict)

    def plot_history(self, hist):
        self.plot_machine.plot_history(hist)

    def timestepping(
            self,
            epoch,
            logs=None,
            spectra=False,
            reconstruction=False
    ):

        def x_(batch_x, b_i, x_old=None):
            """ create timestepping model input from batch """
            x_HR = np.expand_dims(batch_x['HR_data'][b_i,].copy(), 0)
            x_LR = np.expand_dims(batch_x['LR_data'][b_i,].copy(), 0)
            time = pd.to_datetime(batch_x['meta']['time'][b_i, 0], unit="s")

            # remove truth (just to be sure)
            # x_HR[0, 0, ] = np.zeros_like(x_HR[0, 0, ])

            # update x lookback dimension with previous time step if
            # available
            if x_old is not None:
                x_HR[0, 1:, ] = x_old['HR_data'][0, :-1, ]

            return {
                'HR_data': x_HR,
                'LR_data': x_LR,
                'time': time,
            }

        # temp increase batch size
        batch_size_org = self.dgen.batch_size
        self.dgen.batch_size = batch_size_org * 100
        num_batches = self.dgen.__len__()

        # initialization
        x_km1 = None
        results = []
        truths = []
        losses = []
        print('\ntimestepping')
        pb_i = keras.utils.Progbar(num_batches, interval=0.5)
        for b in range(num_batches):
            pb_i.add(1)
            batch_x, batch_y = self.dgen.__getitem__(b)
            batch_size = batch_x['HR_data'].shape[0]

            batch_results = []
            for k in range(batch_size):
                # create kth model input
                x_k = x_(batch_x, k, x_km1)
                # perform time step and update x_k
                x_k['HR_data'][0, 0, ], x_k['latent'] = self.call_model(x_k)
                batch_results.append(x_k)
                x_km1 = x_k

            batch_results_HR = \
                np.concatenate([br['HR_data'] for br in batch_results], 0)

            results += batch_results
            truths.append(batch_y)

            batch_loss = \
                self.model.loss_MSE(
                    batch_results_HR[:,
                                     0,
                                     self.model.masking.rows.cpu(),
                                     self.model.masking.cols.cpu(),
                                     ],
                    batch_y['HR_data'][:,
                                       0,
                                       self.model.masking.rows.cpu(),
                                       self.model.masking.cols.cpu(),
                                       ])
            losses.append(batch_loss)

        logs['timestepper'] = np.mean([ll.cpu() for ll in losses])

        # decrease batch size again
        self.dgen.batch_size = batch_size_org

        self.plot_machine.plot_timestepping(
            results,
            truths,
            epoch,
            {'rows': self.model.masking.rows.cpu(),
             'cols': self.model.masking.cols.cpu(),
             },
        )

        x = np.concatenate([self.restrict_x(r) for r in results], 0)
        y = np.concatenate([self.restrict_y(t) for t in truths], 0)
        z = np.concatenate([self.restrict_y(r) for r in results], 0)
        results_dir_base = self.plot_machine.dirs['results']
        timestepping_file = \
            f'{results_dir_base}/timeseries.dill'
        with open(timestepping_file, 'wb') as file:
            dill.dump({
                'x': x,
                'y': y,
                'z': z,
            }, file)

        if spectra:
            self.plot_machine.spectra_wrapper(x, y, z)

        if reconstruction:
            t_range = np.linspace(0, x.shape[0] - 1, 4).astype(int)
            for t in t_range:
                self.plot_reconstruction(x, y, z, epoch, t)


class AnalysisResNet(AnalysisBase):
    def __init__(
            self,
            data_gen,
            **kwargs,
    ):
        super().__init__(data_gen, **kwargs)

    def call_model(self, x):
        z = self.model(
            {'LR_data': x['LR_data']},
            training=False
        )
        # apply nan mask and detach
        z = (z * self.mask).cpu().detach().numpy()
        return z, []

    def restrict_x(self, x):
        # keep relevant keys, ignore lookback
        x = x['LR_data'][:, 0,]
        return x

    def restrict_y(self, y):
        # keep relevant keys, ignore lookback
        y = y['HR_data'][:, 0,]
        return y

    @property
    def scaler_list(self):
        """Provide a list of scalers to use. Ordering (x,y,z) with x: model
        input, y: truth, z: model prediction

        """
        return ['LR', 'HR', 'HR']


class AnalysisVAE(AnalysisBase):
    def __init__(
            self,
            data_gen,
            **kwargs,
    ):
        super().__init__(data_gen, **kwargs)

    def call_model(self, x):
        z = self.model({'HR_data': ops.nan_to_num(x['HR_data'])},
                       training=False)
        z_decoded = z['decoded']
        z_mean = z['mean'].cpu().detach().numpy()
        # apply nan mask and detach
        z_decoded = (z_decoded * self.mask).cpu().detach().numpy()
        return z_decoded, z_mean

    def restrict_x(self, x):
        # keep relevant keys, ignore lookback
        x = x['HR_data'][:, 0,]
        return x

    def restrict_y(self, y):
        # keep relevant keys, ignore lookback
        y = y['HR_data'][:, 0,]
        return y

    @property
    def scaler_list(self):
        """Provide a list of scalers to use. Ordering (x,y,z) with x: model
        input, y: truth, z: model prediction

        """
        return ['HR', 'HR', 'HR']


class AnalysisPredictor(AnalysisBase):
    """ for now this is the same as the VAE version """
    def __init__(
            self,
            data_gen,
            **kwargs,
    ):
        super().__init__(data_gen, **kwargs)

    def call_model(self, x):
        z = self.model(
            {
                'HR_data': ops.nan_to_num(x['HR_data']),
                'LR_data': ops.nan_to_num(x['LR_data']),
            },
            training=False)

        z_decoded = z['decoded']
        z_mean = z['mean'].cpu().detach().numpy()
        # apply nan mask and detach
        z_decoded = (z_decoded * self.mask).cpu().detach().numpy()
        return z_decoded, z_mean

    def restrict_x(self, x):
        # keep relevant keys, ignore lookback
        x = x['LR_data'][:, 0,]
        return x

    def restrict_y(self, y):
        # keep relevant keys, ignore lookback
        y = y['HR_data'][:, 0,]
        return y

    @property
    def scaler_list(self):
        """Provide a list of scalers to use. Ordering (x,y,z) with x: model
        input, y: truth, z: model prediction

        """
        return ['LR', 'HR', 'HR']


class AnalysisHybrid(AnalysisBase):
    """ for now this is the same as the VAE version """
    def __init__(
            self,
            data_gen,
            **kwargs,
    ):
        super().__init__(data_gen, **kwargs)

    def call_model(self, x):
        z = self.model({'HR_data': ops.nan_to_num(x['HR_data']),
                        'LR_data': ops.nan_to_num(x['LR_data'])
                        },
                       training=False)
        z_hybrid = z['hybrid']
        z_mean = z['mean'].cpu().detach().numpy()
        # apply nan mask and detach
        z_hybrid = (z_hybrid * self.mask).cpu().detach().numpy()
        return z_hybrid, z_mean

    def restrict_x(self, x):
        # keep relevant keys, ignore lookback
        x = x['LR_data'][:, 0,]
        return x

    def restrict_y(self, y):
        # keep relevant keys, ignore lookback
        y = y['HR_data'][:, 0,]
        return y

    @property
    def scaler_list(self):
        """Provide a list of scalers to use. Ordering (x,y,z) with x: model
        input, y: truth, z: model prediction

        """
        return ['LR', 'HR', 'HR']
