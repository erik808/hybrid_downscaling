import numpy as np
import os
import tools
import scipy
import matplotlib.pyplot as plt
import dill
import pytide

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
        # works well for ssh
        # Doodson filter might be cheaper and more general
        
        wt = pytide.WaveTable(["M2", "S2", "N2", "K1",
                               "O1", "Q1", "M4",
                               "K2", "P1", "Mf", "Mm"])

       breakpoint()
        wt = pytide.WaveTable(["M2", "S2", "N2", "K1"])
        
        wt = pytide.WaveTable()


        f, vu = wt.compute_nodal_modulations(time)
        point_evol = data[:, 4, 0]

        # test this, needs std input or csv
        # import tstoolbox
        # tdf = tstoolbox.filter(["tide_doodson"], "lowpass")
        
        point_evol = scipy.signal.detrend(point_evol, axis=0)
        waves = wt.harmonic_analysis(point_evol, f, vu)
        point_tide = wt.tide_from_tide_series(time, waves)
        point_detide = point_evol - point_tide


        
        plt.figure()
        plt.plot(point_evol[:100])
        plt.plot(point_tide[:100])
        plt.plot(point_detide[:100])
        plt.pause(1)
        


    def hovmöller_along_transect(
            self,
            data,
            time,
            scaler=None,
            transect_name='along_flow',
            spectrum_type='energy',
            detide=False,
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
            detide=False
    ):

        transect_data = self.hovmöller_along_transect(
            data,
            time,
            scaler=scaler,
            transect_name=transect_name,
            spectrum_type=spectrum_type,
            detide=detide,
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

        # detrend along specdim
        data_detrend = scipy.signal.detrend(data, axis=0)

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
            H = np.fft.fft(data_padded, axis=0)

            newshape = (Npad // 2 + 1, *dshape[1:])
            S = np.zeros(newshape)
            f = np.linspace(0.0, 0.5, newshape[0])

            for i in range(1, len(f)):
                mult = 1 if i == 0 or i == (Npad // 2) else 2
                S[i,] = mult * np.abs(H[i,])**2 / Npad

        elif method == 'welch':

            f, S = scipy.signal.welch(
                data_detrend,
                axis=0,
                # nperseg=,
                scaling='density',
            )

        if spectrum_type == 'energy':
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
