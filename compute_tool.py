
import numpy as np
import data_manager as dm
from transectpicker.transectpicker import TransectPicker

class ComputeTool():
    """This class takes care of computations: spectra, vorticity, energy,
    enstrophy.

    """

    def __init__(self):

        self.grid, self.binary_mask = dm.get_grid()
        self.mask = np.where(binary_mask==0, np.nan, 1)
        self.e1 = grid.e1t.data # in m
        self.e2 = grid.e2t.data # in m
        self.tdim = 60 * 60 * 24 # seconds to days

        breakpoint()

    def inverse_transform(self, data, scaler):

        breakpoint()

        Nt, Nlat, Nlon, num_channels = data.shape
        return scaler.inverse_transform(data.reshape(Nt,-1))\
                     .reshape(Nt, Nlat, Nlon, num_channels)

    def vorticity_and_divergence(self, data, scaler):
        """
        returns
        zeta: vorticity in /day
        xi: divergence in /day
        """

        data = self.inverse_transform(data, scaler)

        # assume last dimension has variables, ordered as (u,v,...)
        u = data[...,0]  # m/s
        v = data[...,1]  # m/s

        # compute vorticity
        zeta = self.tdim/(self.e1*self.e2) *\
            (np.diff(v*self.e2, axis=2, prepend=np.nan) -
             np.diff(u*self.e1, axis=1, prepend=np.nan))

        # compute divergence
        xi = self.tdim/(self.e1*self.e2) *\
            (np.diff(u*self.e2, axis=2, prepend=np.nan) +
             np.diff(v*self.e1, axis=1, prepend=np.nan))

        # crop nans away
        zeta = zeta[...,1:,1:]
        xi = xi[...,1:,1:]

        return zeta, xi
