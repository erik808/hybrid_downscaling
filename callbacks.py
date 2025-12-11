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

    def get_predictor_layer(self):
        predictor_layer = self.model\
                              .model\
                              .get_layer('latent_predictor')

        # do some checks
        layer_check = (
            len(predictor_layer.weights) == 4 and
            'bias' in predictor_layer.weights[0].path and
            'W_out' in predictor_layer.weights[3].path  # and
            # epoch > 1
        )

        if layer_check:
            print('ESN/DMD layer active')
        else:
            print('ESN/DMD layer inactive')
            return None

        return predictor_layer

    def create_esn_dmd_data(self):
        # temp increase batch size
        batch_size = self.dgen.batch_size
        self.dgen.batch_size = batch_size * 100
        num_batches = self.dgen.__len__()
        # unshuffle
        self.dgen.indices = np.sort(self.dgen.indices)

        # create data matrix
        pb_i = keras.utils.Progbar(num_batches, interval=0.5)
        x_enc_mat = []
        S_mat = []
        x_enc_LR_mat = []
        print('create data for ESN/DMD using available encoder')
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
            x_enc_LR_mat += [ops.squeeze(batch_x['LR_data'][:, 1, ...])
                             .cpu().detach().numpy()]
            S_mat += [batch_x['hidden']]

        # decrease batch size again
        self.dgen.batch_size = batch_size

        # shuffle
        if self.dgen.shuffle:
            np.random.shuffle(self.dgen.indices)

        X = np.concatenate(x_enc_mat, 0)
        X_LR = np.concatenate(x_enc_LR_mat, 0)
        S = np.concatenate(S_mat, 0)

        return X, X_LR, S

    def test_esn_dmd(self, epoch, logs=None):
        X, X_LR, S = self.create_esn_dmd_data()
        sk = np.expand_dims(S[0,], 0)
        xk = np.expand_dims(X[0,], 0)
        Z = np.zeros_like(X)
        ZS = np.zeros_like(S)
        predictor_layer = self.get_predictor_layer()
        for i in range(self.dgen.n):
            xk_LR = np.expand_dims(X_LR[i,], 0)
            xk = np.expand_dims(xk, 0)
            xk, sk = predictor_layer(xk, sk, xk_LR)
            xk = xk.cpu().detach().numpy()
            sk = sk.cpu().detach().numpy()

            Z[i, ] = xk
            ZS[i, ] = sk

        Z = (Z.reshape(Z.shape[0], -1)).T
        X = (X.reshape(X.shape[0], -1)).T
        ZS = (ZS.reshape(ZS.shape[0], -1)).T
        S = (S.reshape(S.shape[0], -1)).T

        plt.switch_backend('qtagg')

        plt.figure(figsize=(10, 10))

        plt.subplot(3, 2, 1)
        a = plt.pcolormesh(Z)
        plt.colorbar(a)
        plt.title('Z')

        plt.subplot(3, 2, 2)
        a = plt.pcolormesh(np.abs(Z - X))
        plt.colorbar(a)
        plt.title('|Z-X|')

        plt.subplot(3, 2, 3)
        a = plt.pcolormesh(ZS)
        plt.colorbar(a)
        plt.title('ZS')

        plt.subplot(3, 2, 4)

        X_norms = np.linalg.norm(X, ord=2, axis=0)
        plt.plot(
            X_norms,
            '.-',
            label='||X||',
        )
        plt.plot(
            np.linalg.norm(Z, ord=2, axis=0),
            '.-',
            label='||Z||',
        )
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(np.linalg.norm(ZS, ord=2, axis=0),
                 'k.-',
                 label='||ZS||',
                 )
        plt.legend()

        plt.subplot(3, 2, 6)
        err = np.linalg.norm((Z - X), ord=2, axis=0) / X_norms
        err_mn = np.mean(err)
        plt.plot(
            err,
            'k.-',
            label='||Z-X||',
        )
        plt.title(f'mean normalized err: {err_mn}')
        print(f'mean normalized err: {err_mn}')

        # plt.plot(
        #     np.linalg.norm((ZS - S), ord=2, axis=0),
        #     '.-',
        #     label='||ZS-S||',
        # )
        plt.legend()

        plt.tight_layout()
        figname = (
            f"{self.dgen.dm.dirs['results']}/"
            f"ESN_DMD_testing_epoch_{epoch}.png"
        )
        plt.savefig(figname)
        print(figname)

        plt.pause(1)

        # self.model.stop_training = True

    def train_esn_dmd(self, epoch, logs=None):
        np.random.seed(1)

        predictor_layer = self.get_predictor_layer()

        X, X_LR, _ = self.create_esn_dmd_data()
        X = (X.reshape(X.shape[0], -1)).T
        X_LR = (X_LR.reshape(X_LR.shape[0], -1)).T

        N = X.shape[0]
        N_LR = X_LR.shape[0]

        esn_dmd_pars = {}
        esn_dmd_pars.update(self.model.esn_dmd_pars)

        if 'DMD' in self.model.predictor:
            esn_dmd_pars['dmdMode'] = True
            esn_dmd_pars['Nr'] = self.dgen.dm.hidden_states.shape[1]
            esn_dmd_pars['feedThrough'] = True

        use_control = True if (
            self.model.predictor == 'DMDc' or
            self.model.predictor == 'ESNc'
        ) else False

        if self.model.predictor == 'ESNc':
            esn_dmd_pars['feedThrough'] = True
            esn_dmd_pars['ftRange'] = range(N, N + N_LR)

        if use_control:
            U = np.vstack([X[:, :-1], X_LR[:, 1:]]).T
        else:
            U = X[:, :-1].T

        Y = X[:, 1:].T

        esn = ESN_mod.ESN(esn_dmd_pars['Nr'], U.shape[1], Y.shape[1])
        esn.setPars(esn_dmd_pars)
        esn.initialize()
        esn.train(U, Y)

        # assign ESN weights to W_out
        bias, W, W_in, W_out = predictor_layer.get_weights()
        predictor_layer.set_weights(
            [
                bias,
                esn.W.todense(),
                esn.W_in.todense(),
                esn.W_out,
            ])

        # fill hidden state in datagenerator
        inds = np.sort(self.dgen.indices)[1:]
        if 'ESN' in self.model.predictor:  # use hidden states
            self.dgen.dm.hidden_states[inds, :] = esn.X

        print('plotting ESN/DMD training data', end='')
        plt.figure(figsize=(14, 10))
        plt.subplot(3, 2, 1)
        a = plt.pcolormesh(U[::10, ::10].T)
        plt.colorbar(a)
        plt.gca().set_title('ESN/DMD training input (U)')
        plt.subplot(3, 2, 2)
        a = plt.pcolormesh(U[-200:, -200:].T)
        plt.colorbar(a)

        plt.subplot(3, 2, 3)
        a = plt.pcolormesh(esn.X[::10, ::10].T)
        plt.colorbar(a)
        plt.gca().set_title(
            'esn X '
        )
        plt.subplot(3, 2, 4)
        a = plt.pcolormesh(esn.X[-200:, -200:].T)
        plt.colorbar(a)

        plt.subplot(3, 2, 5)
        a = plt.imshow(np.log(np.abs(esn.W_out[::10, ::10])))
        plt.gca().set_title(
            'log(abs(W_out)) (coarsened)'
        )
        plt.colorbar(a)

        plt.subplot(3, 2, 6)
        plt.plot(bias)
        plt.gca().set_title(
            'bias'
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.dgen.dm.dirs['results']}/"
            f"DMD_analysis_epoch_{epoch}.png")
        print(' done')

    def on_epoch_begin(self, epoch, logs=None):

        # self.dgen.mode = 'train'
        # self.dgen.create_indices()
        # self.train_esn_dmd(epoch, logs)

        # self.dgen.mode = 'test'
        # self.dgen.create_indices()
        # self.test_esn_dmd(epoch, logs)

        if self.dgen.mode == 'train':
            self.train_esn_dmd(epoch, logs)
        elif self.dgen.mode == 'test':
            self.test_esn_dmd(epoch, logs)

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
        self.plot_machine.create_postfix()
        self.plot_machine.create_results_dir(epoch)
        self.construct_mask()
        # self.model.stop_training = True
        self.timestepping(
            epoch,
            logs,
            spectra=self.spectra,
            reconstruction=self.reconstruction,
        )

        raise Exception('doei')

        return None

    def on_epoch_end(self, epoch, logs=None):
        self.plot_machine.create_postfix()
        self.plot_machine.create_results_dir(epoch)
        self.update_history(logs)
        plt.close('all')
        self.timestepping(
            epoch,
            logs,
            spectra=self.spectra,
            reconstruction=self.reconstruction,
        )
        return None

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
            hidden = np.expand_dims(batch_x['hidden'][b_i,].copy(), 0)
            time = pd.to_datetime(batch_x['meta']['time'][b_i, 0], unit="s")

            # remove truth (just to be sure)
            # x_HR[0, 0, ] = np.zeros_like(x_HR[0, 0, ])

            # update x lookback dimension with previous time step if
            # available

            if x_old is not None:
                x_HR[0, 1:, ] = x_old['HR_data'][0, :-1, ]
                hidden = x_old['hidden']

            return {
                'HR_data': x_HR,
                'LR_data': x_LR,
                'hidden': hidden,
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
                x_k['HR_data'][0, 0, ], add_out = self.call_model(x_k)

                for k, v in add_out.items():
                    x_k[k] = v

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

        mask = {
            'rows': self.model.masking.rows.cpu(),
            'cols': self.model.masking.cols.cpu(),
        }

        self.plot_machine.plot_timestepping(
            results,
            truths,
            epoch,
            mask,
        )

        results_dir_base = self.plot_machine.dirs['results']
        timestepping_file = \
            f'{results_dir_base}/results.dill'

        with open(timestepping_file, 'wb') as file:
            dill.dump(
                {
                    'results': results,
                    'truths': truths,
                    'logs': logs,
                    'mask': mask,
                }, file)

        x_mat = np.concatenate([self.restrict_x(r) for r in results], 0)
        y_mat = np.concatenate([self.restrict_y(t) for t in truths], 0)
        z_mat = np.concatenate([self.restrict_y(r) for r in results], 0)
        t_arr = np.array([np.datetime64(r['time']) for r in results])

        if spectra:
            self.plot_machine.spectra_wrapper(
                x_mat,
                y_mat,
                z_mat,
                t_arr
            )

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
        return z, {}

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
        return z_decoded, {'ls_mean': z_mean}

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
                'hidden': ops.nan_to_num(x['hidden']),
            },
            training=False)

        z_decoded = z['decoded']
        z_hidden = z['hidden']
        z_mean = z['mean'].cpu().detach().numpy()
        z_ls_pred = z['ls_pred'].cpu().detach().numpy()
        # apply nan mask and detach2
        z_decoded = (z_decoded * self.mask).cpu().detach().numpy()
        # print(z_hidden[0, :4])
        return z_decoded, {
            'ls_mean': z_mean,
            'ls_pred': z_ls_pred,
            'hidden': z_hidden
        }

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
                        'LR_data': ops.nan_to_num(x['LR_data']),
                        'hidden': ops.nan_to_num(x['hidden']),
                        },
                       training=False)
        z_hybrid = z['hybrid']
        z_hidden = z['hidden']
        z_mean = z['mean'].cpu().detach().numpy()
        z_ls_pred = z['ls_pred'].cpu().detach().numpy()
        # apply nan mask and detach
        z_hybrid = (z_hybrid * self.mask).cpu().detach().numpy()
        return z_hybrid, {
            'ls_mean': z_mean,
            'ls_pred': z_ls_pred,
            'hidden': z_hidden,
        }

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
