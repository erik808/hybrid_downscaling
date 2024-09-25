
import numpy as np
import data_manager as dm

class Analysis():
    """
    computations of spectra, vorticity, energy, enstrophy
    should be factorized here
    """

    def __init__(self):

        self.grid, self.binary_mask = dm.get_grid()
        self.mask = np.where(binary_mask==0, np.nan, 1)
        self.e1 = grid.e1t.data # in m
        self.e2 = grid.e2t.data # in m
        self.tdim = 60 * 60 * 24 # seconds to days


        breakpoint()


    def inverse_transform(self, data, scaler):
        Nt, Nlat, Nlon, num_channels = data.shape
        return scaler.inverse_transform(data.reshape(Nt,-1))\
                     .reshape(Nt, Nlat, Nlon, num_channels)

    
    


        
