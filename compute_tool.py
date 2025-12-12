import numpy as np
import os
import tools
import scipy
import matplotlib.pyplot as plt
import dill
from scipy.stats import binned_statistic

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

    def get_regridder(self, transect_name):

        if transect_name not in self.transect_regridders:
            key = transect_name
            dill_file = f'{self.dm.transect_dir}/{transect_name}.dill'
            print(f'Loading transect from {dill_file}')
            with open(dill_file, 'rb') as file:
                tpicker = dill.load(file)['tpicker']

                transect_res = len(tpicker.x_trans)
                regridder = \
                    self.regrid_to_transect(tpicker,
                                            resolution=transect_res)

            self.transect_regridders.update({key: regridder})

        self.regridder = self.transect_regridders[transect_name]

    def regrid_to_transect(self, tpicker, resolution=1e2):

        print('Create transect regridder')
        grid_HR = self.dm.grid_HR

        lons = grid_HR['lon'][0, :]
        lats = grid_HR['lat'][:, 0]

        transect = {
            'lon_start': lons[tpicker.x_trans[0]],
            'lon_end': lons[tpicker.x_trans[-1]],
            'lat_start': lats[tpicker.y_trans[0]],
            'lat_end': lats[tpicker.y_trans[-1]]
        }
        return tools.regrid_to_transect(grid_HR,
                                        resolution=resolution,
                                        **transect)

    def detide(self, data, time):
        import pytide
        wt = pytide.WaveTable(["M2", "S2", "N2", "K1",
                               "O1", "Q1", "M4",
                               "K2", "P1", "Mf", "Mm"])

        dates = time.values
        f, vu = wt.compute_nodal_modulations(dates)
        breakpoint()

    def hovmöller_along_transect(
            self,
            data,
            time,
            scaler=None,
            detide=False,
            transect_name='along_flow',
            spectrum_type='energy',
    ):
        self.get_regridder(transect_name)

        if spectrum_type == 'energy':
            transect_data = self.invert_and_regrid(data, scaler)

        elif spectrum_type == 'enstrophy':
            zeta = self.vorticity(data, scaler, crop=False)
            transect_data = self.do_regridding(zeta)

        elif spectrum_type == 'ssh':
            ssh = self.get_ssh(data, scaler)
            transect_data = self.do_regridding(ssh)

        if detide:
            transect_data = self.detide(transect_data, time)
        return transect_data

    def compute_spectrum_along_transect(
            self,
            data,
            time,
            scaler=None,
            transect_name='along_flow',
            spectrum_type='energy',
            direction='spatial',
    ):

        transect_data = self.hovmöller_along_transect(
            data,
            time,
            scaler=scaler,
            transect_name=transect_name,
            spectrum_type=spectrum_type,
        )

        k, S = self.compute_spectrum(
            transect_data,
            spectrum_type=spectrum_type,
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
            spectrum_type,
            direction='spatial',
            method='welch',
    ):

        # reorder such that the dimension along which we compute a
        # spectrum is first always
        specdim = 1 if direction == 'spatial' else 0
        remdim = (specdim + 1) % 2
        reorder = (specdim, remdim) + tuple(range(2, len(data.shape)))
        data = data.transpose(reorder)

        plt.close('all')
        plt.figure()

        # data_tp = self.taper_data(data)

        # data_detrend = data
        data_detrend = scipy.signal.detrend(data, axis=0)
        # data_detrend = scipy.signal.detrend(data_detrend, axis=1)

        pfac = 5
        N, M, C = data_detrend.shape
        data_padded = np.pad(
            data_detrend,
            ((N // pfac, N // pfac), (0, 0), (0, 0))
        )
        plt.pcolormesh(data_padded[..., 0])

        H = np.fft.fft(data_padded, axis=0)
        energy = np.zeros((N // 2 + 1, M, C))
        freqs = np.linspace(0.0, 0.5, N // 2 + 1)

        for i in range(len(freqs)):
            mult = 1 if i == 0 or i == N // 2 else 2
            energy[i,] = mult * np.abs(H[i,])**2 / N

        f, S = scipy.signal.welch(
            data_padded,
            axis=0,
            scaling='density',
        )

        df = f[2] - f[1]

        variance_orig = np.var(data_padded, axis=0)
        variance_fft = np.sum(energy, axis=0) / N
        variance_welch = np.sum(S, axis=0) * df

        # plt.plot(variance_orig[:, 0])
        # plt.plot(variance_fft[:, 0])
        # plt.plot(variance_welch[:, 0])
        # print(df)

        # plt.loglog(freqs[1:], energy[1:])
        # kinetic energy
        plt.figure()
        mKEfft = 1/2 * np.mean(np.sum(energy, axis=2), axis=1)
        mKEwelch = 1/2 * np.mean(np.sum(S, axis=2), axis=1)
        mKEfft = (energy[:, 20, 0])
        mKEwelch = (S[:, 20, 0])

        plt.loglog(freqs[1:,], mKEfft[1:,])
        plt.loglog(f, mKEwelch)

        plt.figure()
        plt.plot(freqs)
        plt.plot(f)
        plt.pause(1)

        x_ref = np.array([0.01, 1])
        y_ref53 = 5e0 * (x_ref / x_ref[0])**(-5/3)
        y_ref3 = 5e0 * (x_ref / x_ref[0])**(-3)
        y_ref5 = 5e0 * (x_ref / x_ref[0])**(-5)
        plt.loglog(x_ref, y_ref53, '--')
        plt.loglog(x_ref, y_ref3,  '--')
        plt.loglog(x_ref, y_ref5,  '--')

        plt.pause(1)

        data_tp = self.taper_data(data)

        if spectrum_type == 'energy':
            data_tp = 0.5 * np.sum(np.square(np.abs(data_tp)), axis=2)
        elif (
                spectrum_type == 'enstrophy' or
                spectrum_type == 'ssh'
        ):
            data_tp = 0.5 * np.square(np.abs(data_tp))

        if method == 'fft':
            S = (np.abs(np.fft.fft(data_tp, axis=1)))
            S = (np.abs(np.fft.fft(data_tp, axis=1))**2) / data_tp.shape[1]
            f = np.abs(np.fft.fftfreq(data_tp.shape[1]))
        elif method == 'welch':
            f, S = scipy.signal.welch(
                data_tp,
                axis=1,
                nperseg=400,
                # noverlap=64,
                scaling='density',
                average='median',
            )

        breakpoint()

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
