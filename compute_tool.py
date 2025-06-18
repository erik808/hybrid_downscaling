import numpy as np
import os
import matplotlib.pyplot as plt
from importlib import reload
import data_utils
reload(data_utils)
from data_utils import DataFactory
import dill

from transectpicker.transectpicker import TransectPicker


class ComputeTool():
    """This class takes care of computations for the CMEMS data: spectra,
    vorticity, energy, enstrophy.

    """

    def __init__(self,
                 case_study='cmems'):

        self.case_study = case_study

        if self.case_study == 'cmems':
            self.dm = DataFactory(case_study=self.case_study)
            self.grid, self.binary_mask = self.dm.get_grid()
            self.mask = np.where(self.binary_mask==0, np.nan, 1)
            self.e1 = self.grid.e1t.data  # in m
            self.e2 = self.grid.e2t.data  # in m
            self.tdim = 60 * 60 * 24  # seconds to days
            self.transect_regridder = 'none'

    def construct_regridder(self, transect_name):

        if self.transect_regridder == transect_name:
            return
        else:
            self.transect_regridder = transect_name
            dill_file = f'{self.dm.transect_dir}/{transect_name}.dill'
            print(f'Loading transect from {dill_file}')
            with open(dill_file, 'rb') as file:
                tpicker = dill.load(file)['tpicker']

                transect_res=len(tpicker.x_trans)
                self.regridder = \
                    self.dm.regrid_to_transect(tpicker,
                                               resolution=transect_res)

    def compute_spectrum_along_transect(self, data, scaler=None,
                                        transect_name='along_flow',
                                        spectrum_type='energy'):

        self.construct_regridder(transect_name)

        if spectrum_type == 'energy':
            transect_data = self.invert_and_regrid(data, scaler)
            spectrum = self.compute_energy_spectrum(transect_data)
        elif spectrum_type == 'enstrophy':
            zeta = self.vorticity(data, scaler, crop=False)
            zeta_tr = self.do_regridding(zeta)
            spectrum = self.compute_enstrophy_spectrum(zeta_tr)
        else:
            raise Exception('Not implemented yet')
        return spectrum

    def taper_data(self, data):
        # taper the boundaries
        n = data.shape[1]
        x = np.linspace(0, 1, n)
        tpr = (1 + np.tanh((x - 0.1) * 3e1)) / 2
        tpr = tpr + np.flip(tpr) - 1
        if data.ndim == 3:
            data = (data.transpose(2, 0, 1) * tpr)\
                .transpose(1, 2, 0)
        elif data.ndim == 2:
            data = data * tpr
        else:
            raise Exception('data has wrong shape')

        return data

    def compute_energy_spectrum(self, data):
        """ normalized energy spectrum
        """
        data = (data.transpose(1, 0, 2) - np.mean(data, axis=1))\
            .transpose(1, 0, 2)  # remove spatial average
        data = data - np.mean(data, axis=0)  # remove time average

        data = self.taper_data(data)
        H = np.fft.rfft(data, axis=1)
        S = 0.5 * np.sum(np.square(np.abs(H)), axis=2)
        S = S / np.max(S)
        return S

    def compute_enstrophy_spectrum(self, data):
        """ normalized enstrophy spectrum
        """
        data = (data.T - np.mean(data, axis=1)).T  # remove spatial average
        data = data - np.mean(data, axis=0)  # remove time average

        data = self.taper_data(data)
        H = np.fft.rfft(data, axis=1)
        S = 0.5 * np.square(np.abs(H))
        S = S / np.max(S)
        return S

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
        zeta = self.tdim /(self.e1 * self.e2) *\
            (np.diff(v * self.e2, axis=2, prepend=np.nan) -
             np.diff(u * self.e1, axis=1, prepend=np.nan))

        # crop nans away
        if crop:
            zeta = zeta[..., 1:, 1:]

        return zeta.squeeze()

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

        container = {'tpicker' : tpicker}

        print(f'writing to {dill_file}')
        with open(dill_file, 'wb') as file:
            dill.dump(container, file)
