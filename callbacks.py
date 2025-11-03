import numpy as np
import keras
from keras import ops
import os
import plot_utils
import importlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

importlib.reload(plot_utils)

if os.environ.get('DISPLAY') is not None:
    plt.switch_backend('qtagg')
else:
    plt.switch_backend('Agg')


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
        self.output_path = self.dgen.dm.dirs['results']
        self.plot_machine = \
            plot_utils.PlotMachine(dm=self.dgen.dm)

        # create a nan-mask
        self.mask_constructed = False
        plt.close('all')

    @abstractmethod
    def call_model(self, x):
        pass

    @abstractmethod
    def restrict_xy(self, x, y):
        pass

    @property
    @abstractmethod
    def scaler_list(self):
        pass

    def construct_mask(self):
        if not self.mask_constructed:
            self.mask = ops.not_equal(self.model.mask, 0.0)
            self.mask = \
                self.model.mask / ops.cast(self.mask, self.model.mask.dtype)
            self.mask_constructed = True
        else:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        return None

    def on_epoch_end(self, epoch, logs=None):
        self.construct_mask()
        if epoch % 1 == 0 or epoch == self.params['epochs'] - 1:
            if 'reconstruction' in self.plot_instructions:
                self.random_prediction(epoch)
            if 'spectra' in self.plot_instructions:
                self.plot_spectra(epoch)

    def random_prediction(self, epoch):
        n = self.dgen.__len__()
        idx = np.random.randint(n)
        self.predict_and_plot(idx, epoch)

    def predict_and_plot(self, idx, epoch):
        x, y = self.dgen.__getitem__(idx)
        z = self.call_model(x)
        x, y = self.restrict_xy(x, y)
        self.plot_reconstruction(x[0,], y[0,], z[0,], epoch)

    def plot_spectra(self, epoch):
        n = self.dgen.__len__()
        x_list, y_list, z_list = [], [], []
        for i in range(n):
            x, y = self.dgen.__getitem__(i)
            z = self.call_model(x)
            x, y = self.restrict_xy(x, y)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        z = np.concatenate(z_list, axis=0)

        # unscale variables
        def unscale_var(var, scaler):
            var_shape = var.shape
            Tdim = var_shape[0] if len(var_shape) > 3 else 1
            return scaler\
                .inverse_transform(var.reshape(Tdim, -1))\
                .reshape(var_shape)

        x_unscaled, y_unscaled, z_unscaled = \
            [unscale_var(d, self.dgen.dm.scalers[res])
             for d, res in zip([x, y, z], self.scaler_list)]

        if x.shape != y.shape:
            # upsample unscaled x (bilinear interpolation)
            x_unscaled = \
                np.ascontiguousarray(x_unscaled.transpose((0, 3, 1, 2)))
            x_unscaled = \
                self.dgen.dm.bilin_upsampler(x_unscaled)\
                            .transpose((0, 2, 3, 1))

        data = {
            'lowres': np.nan_to_num(x_unscaled),
            'scaler_lowres': None,
            'truth': np.nan_to_num(y_unscaled),
            'scaler_truth': None,
            'pred': np.nan_to_num(z_unscaled),
        }

        self.plot_machine\
            .plot_energy_spectrum(data,
                                  epoch,
                                  transect_name='along_flow')
        self.plot_machine\
            .plot_energy_spectrum(data,
                                  epoch,
                                  transect_name='across_flow')
        self.plot_machine\
            .plot_enstrophy_spectrum(data,
                                     epoch,
                                     transect_name='along_flow')
        self.plot_machine\
            .plot_enstrophy_spectrum(data,
                                     epoch,
                                     transect_name='across_flow')

    def plot_reconstruction(self, x, y, z, epoch):
        assert y.shape == z.shape

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

        if x.shape != z.shape:
            x_bilin = np.ascontiguousarray(x.transpose((2, 0, 1)))
            x_bilin = self.dgen.dm.bilin_upsampler(x_bilin)\
                                  .transpose((1, 2, 0))

            err_uo = np.abs(y[..., 0] - x_bilin[..., 0])
            err_zos = np.abs(y[..., 2] - x_bilin[..., 2])

            plot_dict = {
                'meta': {'epoch': epoch,
                         'prefix': 'bilin_',
                         'subplot_shape': [2, 4]},
                'input uo': wrapper(
                    {'data': x[..., 0]}),
                'bilin uo': wrapper(
                    {'data': x_bilin[..., 0]}),
                'truth uo': wrapper(
                    {'data': y[..., 0]}),
                'err uo': wrapper(
                    {'data': err_uo,
                     'vmin': 0, 'vmax': 0.1}),
                'input zos': wrapper(
                    {'data': x[..., 2]}),
                'bilin zos': wrapper(
                    {'data': x_bilin[..., 2]}),
                'truth zos': wrapper(
                    {'data': y[..., 2]}),
                'err zos': wrapper(
                    {'data': err_zos,
                     'vmin': 0, 'vmax': 0.1}),
            }
            self.plot_machine.plot_reconstructions(plot_dict)

    def plot_history(self, hist):
        self.plot_machine(hist)


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
        return z

    def restrict_xy(self, x, y):
        # keep relevant keys, ignore lookback
        x, y = x['LR_data'][:, 0,], y['HR_data'][:, 0,]
        return x, y

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
        z = z['decoded']
        # apply nan mask and detach
        z = (z * self.mask).cpu().detach().numpy()
        return z

    def restrict_xy(self, x, y):
        # keep relevant keys, ignore lookback
        x, y = x['HR_data'][:, 0,], y['HR_data'][:, 0,]
        return x, y

    @property
    def scaler_list(self):
        """Provide a list of scalers to use. Ordering (x,y,z) with x: model
        input, y: truth, z: model prediction

        """
        return ['HR', 'HR', 'HR']
