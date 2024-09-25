import numpy as np
import data_manager as dm
from transectpicker.transectpicker import TransectPicker

class ComputeTool():
    """This class takes care of computations: spectra, vorticity, energy,
    enstrophy.

    """

    def __init__(self):

        self.grid, self.binary_mask = dm.get_grid()
        self.mask = np.where(self.binary_mask==0, np.nan, 1)
        self.e1 = self.grid.e1t.data # in m
        self.e2 = self.grid.e2t.data # in m
        self.tdim = 60 * 60 * 24 # seconds to days

    def inverse_transform(self, data, scaler=None):
        
        assert (data.ndim >= 3 and
                data.ndim <= 4), " wrong data input format "

        if scaler == None: # bypass this routine
            return data        

        if data.ndim == 3: # assume time dimension is not present.
            data = np.expand_dims(data,axis=0)

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


    def create_transect(self, field):
        """Support function that wraps the transectpicker.
        Supply a field, draw a transect and save to file.
        
        input: field
        """

        # create transect dir if not existing
        os.system(f'mkdir -p {dm.transect_dir}')

        plt.subplots(figsize=(5,4))
        im = plt.pcolormesh(field)
        tpicker = TransectPicker(im, field)
        plt.show()

        transect_name = input('Give a name for the transect\n')
        dill_file = f'{dm.transect_dir}/{transect_name}.dill'

        container = {'tpicker' : tpicker}

        print(f'writing to {dill_file}')
        with open(dill_file, 'wb') as file:
            dill.dump(container, file)
