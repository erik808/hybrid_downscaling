import numpy as np
import os
import tools
import scipy
import matplotlib.pyplot as plt
import dill
from sklearn.metrics.pairwise import haversine_distances
from math import radians


from transectpicker.transectpicker import TransectPicker


class ComputeTool():
    """This class takes care of computations for the CMEMS data: spectra,
    vorticity, energy, enstrophy.

    """

    def __init__(self, dm):
        self.dm = dm
        self.grid = self.dm.load_coords()
        self.binary_mask = self.dm.mask[0,]
        self.mask = np.where(self.binary_mask == 0, np.nan, 1)
        self.e1 = self.grid.e1t.data  # in m
        self.e2 = self.grid.e2t.data  # in m
        self.tdim = 60 * 60 * 24  # seconds to days
        self.transect_regridders = {}
        self.transect_distance = {}

    def get_regridder(self, transect_name):

        if transect_name not in self.transect_regridders:
            tpicker, transect = self.get_transect(transect_name)
            transect_res = len(tpicker.x_trans)
            distance = self.get_transect_distance(transect,
                                                  resolution=transect_res)
            self.transect_distance.update({transect_name: distance})
            regridder = \
                self.regrid_to_transect(tpicker,
                                        transect,
                                        resolution=transect_res)
            self.transect_regridders.update({transect_name: regridder})

        self.regridder = self.transect_regridders[transect_name]

    def get_transect_distance(self, transect, resolution):
        # add distance over transect
        start = [radians(transect['lat_start']),
                 radians(transect['lon_start'])]
        end = [radians(transect['lat_end']),
               radians(transect['lon_end'])]

        # distance between points on globe in kms
        dist = haversine_distances([start, end]) * 6371000/1000
        dist = dist[0, 1]
        return dist, dist / resolution

    def get_transect(self, transect_name):
        dill_file = f'{self.dm.transect_dir}/{transect_name}.dill'
        print(f'Loading transect from {dill_file}')
        with open(dill_file, 'rb') as file:
            tpicker = dill.load(file)['tpicker']

        grid_HR = self.dm.grid_HR

        lons = grid_HR['lon'][0, :]
        lats = grid_HR['lat'][:, 0]

        transect = {
            'lon_start': lons[tpicker.x_trans[0]],
            'lon_end': lons[tpicker.x_trans[-1]],
            'lat_start': lats[tpicker.y_trans[0]],
            'lat_end': lats[tpicker.y_trans[-1]]
        }

        return tpicker, transect

    def regrid_to_transect(self,
                           tpicker,
                           transect,
                           resolution=1e2):
        grid_HR = self.dm.grid_HR
        return tools.regrid_to_transect(grid_HR,
                                        resolution=resolution,
                                        **transect)

    def doodson_filter(self, data):
        kernel = np.array([
            1, 0, 1, 0,
            0, 1, 0, 1,
            1, 0, 2, 0,
            1, 1, 0, 2,
            1, 1, 2, 0,
            2, 1, 1, 2,
            0, 1, 1, 0,
            2, 0, 1, 1,
            0, 1, 0, 0,
            1, 0, 1,
        ])
        kernel = kernel / 30

        for i in range(data.ndim-1):
            kernel = np.expand_dims(kernel, -1)

        return scipy.signal.fftconvolve(data,
                                        kernel,
                                        mode='valid',
                                        axes=0)

    def lanczos_filter(self, data):

        def lanczos_weights(window_len=121,
                            cutoff_period=40.0,
                            dt=1.0):
            half_len = (window_len - 1) // 2
            n = np.arange(-half_len, half_len + 1)

            # Fundamental frequency calculation
            f_c = (1.0 / cutoff_period) * dt

            with np.errstate(divide='ignore', invalid='ignore'):
                weights = np.sinc(2 * f_c * n) * np.sinc(n / half_len)
                weights[half_len] = 2 * f_c  # Correct central weight

            return weights / np.sum(weights)

        kernel = lanczos_weights()

        for i in range(data.ndim-1):
            kernel = np.expand_dims(kernel, -1)

        return scipy.signal.fftconvolve(data,
                                        kernel,
                                        mode='same',
                                        axes=0)

    def detide(self, data):
        # Doodson filter might be cheaper and more general
        data_detrend = scipy.signal.detrend(data, axis=0)
        data_detide = self.doodson_filter(data_detrend)

        return data_detide

    def hovmöller_along_transect(
            self,
            data,
            scaler=None,
            transect_name='along_flow',
            spectrum_type='energy',
            detide=False,
    ):
        self.get_regridder(transect_name)

        if (
                spectrum_type == 'energy' or
                spectrum_type == 'uo' or
                spectrum_type == 'vo'
        ):
            transect_data = self.invert_and_regrid(data, scaler)[..., :2]

        elif (spectrum_type == 'enstrophy' or
              spectrum_type == 'vorticity'):
            zeta = self.vorticity(data, scaler, crop=False)
            transect_data = self.do_regridding(zeta)

        elif spectrum_type == 'ssh':
            ssh = self.get_ssh(data, scaler)
            transect_data = self.do_regridding(ssh)

        elif spectrum_type == 'TKE':
            transect_data = self.invert_and_regrid(data, scaler)

            window = \
                self.get_window_view(transect_data[..., :2],
                                     self.dm.window_size)

            # MKE = <u>**2 + <v>**2
            MKE = np.sum(window[:, :, :2,].mean(axis=-1)**2, axis=-1)

            # means of squared velocities: UV2 = <u**2> + <v**2>
            UV2 = np.sum((window[:, :, :2,]**2).mean(axis=-1), axis=-1)

            # TKE = <u**2> - <u>**2 + <v**2> - <v>**2
            TKE = UV2 - MKE

            transect_data = TKE

        elif spectrum_type == 'MKE':
            transect_data = self.invert_and_regrid(data, scaler)

            window = \
                self.get_window_view(transect_data[..., :2],
                                     self.dm.window_size)

            # MKE = <u>**2 + <v>**2
            MKE = np.sum(window[:, :, :2,].mean(axis=-1)**2, axis=-1)
            transect_data = MKE

        if detide:
            transect_data = self.detide(transect_data)

        return transect_data

    def get_window_view(self, data, wsize):
        window_view = \
            np.lib.stride_tricks.sliding_window_view(
                data,
                wsize,
                axis=0,
            )

        return window_view

    def compute_spectrum_along_transect(
            self,
            data,
            scaler=None,
            transect_name='along_flow',
            spectrum_type='energy',
            direction='spatial',
            detide=False
    ):
        transect_data = self.hovmöller_along_transect(
            data,
            scaler=scaler,
            transect_name=transect_name,
            spectrum_type=spectrum_type,
            detide=detide,
        )

        k, S = self.compute_spectrum(
            transect_data,
            direction=direction,
        )

        return k, S, transect_data

    def taper_data(self, data):
        # taper the boundaries
        n = data.shape[1]
        x = np.linspace(0, 1, n)
        tpr = tpr_fun(x, offset=0.1, steepness=3e1)

        if data.ndim == 3:
            data = (data.transpose(2, 0, 1) * tpr)\
                .transpose(1, 2, 0)
        elif data.ndim == 2:
            data = data * tpr
        else:
            raise Exception('data has wrong shape')

        return data

    def compute_spectrum(
            self,
            data,
            direction='spatial',
            method='welch',
    ):

        # reorder such that the dimension along which we compute a
        # spectrum is first always
        if direction == 'spatial':
            specdim = 1
        elif direction == 'temporal':
            specdim = 0
        else:
            raise Exception('invalid fft direction')

        remdim = (specdim + 1) % 2
        reorder = (specdim, remdim) + tuple(range(2, len(data.shape)))
        data = data.transpose(reorder)

        # detrend along specdim
        data_detrend = scipy.signal.detrend(data, axis=0)
        # detrend along the other dim as well
        data_detrend = scipy.signal.detrend(data_detrend, axis=1)

        if method == 'fft':  # pad data for fft
            dshape = data_detrend.shape
            N = dshape[0]
            pfac = 10
            padding = ((N // pfac, N // pfac), (0, 0), (0, 0))
            padding = padding[:len(dshape)]

            data_padded = np.pad(
                data_detrend,
                padding,
            )

            Npad = data_padded.shape[0]
            newshape = (Npad // 2 + 1, *dshape[1:])
            f = np.linspace(0.0, 0.5, newshape[0])
            H = np.fft.fft(data_padded, axis=0)

            S = np.zeros(newshape)
            for i in range(1, len(f)):
                mult = 1 if i == 0 or i == (Npad // 2) else 2
                S[i,] = mult * np.abs(H[i,])**2 / Npad

        elif method == 'welch':
            nperseg = data_detrend.shape[0] \
                if data_detrend.shape[0] < 512 else 512

            f, S = scipy.signal.welch(
                data_detrend,
                axis=0,
                scaling='density',
                nperseg=nperseg,
            )

        if S.ndim == 3:
            S = np.sum(S, axis=-1)

        return f, S

    def do_regridding(self, field):
        # regrid
        if field.shape[-1] != self.regridder.shape_in[-1]:
            field_tr = \
                self.regridder(
                    np.ascontiguousarray(
                        field.transpose(0, 3, 1, 2))
                ).transpose(0, 2, 3, 1)
        else:
            field_tr = \
                self.regridder(np.ascontiguousarray(field))

        # select transect along diagonal
        field_tr = field_tr[:, np.arange(field_tr.shape[1]),
                            np.arange(field_tr.shape[2]), ]

        return field_tr

    def invert_and_regrid(self, data, scaler):
        data = self.check_data_dims(data)
        field = self.inverse_transform(data, scaler)
        field_tr = self.do_regridding(field)
        return field_tr

    def inverse_transform(self, data, scaler=None):
        data = self.check_data_dims(data)
        if scaler is None:
            return data
        else:
            Nt, Nlat, Nlon, num_channels = data.shape
            return scaler.inverse_transform(data.reshape(Nt, -1))\
                         .reshape(Nt, Nlat, Nlon, num_channels)

    def check_data_dims(self, data):

        assert (data.ndim >= 3 and
                data.ndim <= 4), " wrong data input format "

        if data.ndim == 3:  # assume time dimension is not present, prepend it.
            data = np.expand_dims(data, axis=0)
        return data

    def vorticity(self, data, scaler, crop=True):
        """
        returns
        zeta: vorticity in /day
        """

        data = self.inverse_transform(data, scaler)

        # assume last dimension has variables, ordered as (u,v,...)
        u = data[..., 0]  # m/s
        v = data[..., 1]  # m/s

        # compute vorticity
        zeta = self.tdim / (self.e1 * self.e2) *\
            (np.diff(v * self.e2, axis=2, prepend=np.nan) -
             np.diff(u * self.e1, axis=1, prepend=np.nan))

        # crop nans away
        if crop:
            zeta = zeta[..., 1:, 1:]

        return zeta.squeeze()

    def get_ssh(self, data, scaler):
        """
        returns ssh
        """
        data = self.inverse_transform(data, scaler)
        # assume last dimension has variables, ordered as (u,v,ssh)
        ssh = data[..., 2]

        return ssh

    def divergence(self, data, scaler, crop=True):
        """
        returns
        xi: divergence in /day
        """

        data = self.inverse_transform(data, scaler)

        # assume last dimension has variables, ordered as (u,v,...)
        u = data[..., 0]  # m/s
        v = data[..., 1]  # m/s

        # compute divergence
        xi = self.tdim / (self.e1 * self.e2) *\
            (np.diff(u * self.e2, axis=2, prepend=np.nan) +
             np.diff(v * self.e1, axis=1, prepend=np.nan))

        # crop nans away
        if crop:
            xi = xi[..., 1:, 1:]

        return xi.squeeze()

    def create_transect(self, field):
        """Support function that wraps the transectpicker.
        Supply a field, draw a transect and save to file.

        input: field
        """

        # create transect dir if not existing
        os.system(f'mkdir -p {self.dm.transect_dir}')

        plt.subplots(figsize=(5, 4))
        im = plt.pcolormesh(field)
        tpicker = TransectPicker(im, field)
        plt.show()

        transect_name = input('Give a name for the transect\n')
        dill_file = f'{self.dm.transect_dir}/{transect_name}.dill'

        container = {'tpicker': tpicker}

        print(f'writing to {dill_file}')
        with open(dill_file, 'wb') as file:
            dill.dump(container, file)


def tpr_fun(x, offset=0.1, steepness=3e1):
    tpr = (1 + np.tanh((x - offset) * steepness)) / 2
    tpr = tpr + np.flip(tpr) - 1
    return tpr
