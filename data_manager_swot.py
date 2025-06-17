import tools
import xarray as xr
import numpy as np
from data_manager_base import DataManagerBase
from importlib import reload
import data_utils
reload(data_utils)
from data_utils import CustomScaler


class DataManagerSWOT(DataManagerBase):

    def __init__(
            self,
            testing_mode=False
    ):
        tools.load_config(self, config_name='data_config_swot')
        self.data_dir = 'data'
        self.swot_duacs_fname = (f'{self.data_dir}/'
                                 'subset_merge_swot_duacs_1y.nc')

    def create_training_data(self):

        self.ds = xr.open_dataset(self.swot_duacs_fname)

        data_HR = self.ds.ssha.data
        data_LR = self.ds.sla.data

        Nt, Nlat, Nlon = data_HR.shape
        assert data_LR.shape == data_HR.shape

        #  We scale the data with a single scaling for all features
        scaler = CustomScaler('minmax_over_all_features')

        data_HR = scaler.fit_transform(data_HR.reshape(Nt, -1))\
                        .reshape(Nt, Nlat, Nlon)
        scaler.shift = np.nan_to_num(scaler.shift, nan=0.0)
        scaler.scale = np.nan_to_num(scaler.scale, nan=1.0)
        data_LR = scaler.transform(data_LR.reshape(Nt, -1))\
                        .reshape(Nt, Nlat, Nlon)

        breakpoint()
