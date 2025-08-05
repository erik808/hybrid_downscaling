import dill
import tools
import torch
import xarray as xr
import numpy as np

from data_manager_base import DataManagerBase
from tools import CustomScaler


class DataManagerSWOT(DataManagerBase):

    def __init__(
            self,
            testing_mode=False
    ):
        tools.load_config(self, config_name='data_config_swot')
        self.data_dir = 'data'
        self.swot_duacs_fname = (f'{self.data_dir}/'
                                 'subset_merge_swot_duacs_1y.nc')
        self.dill_file = f'{self.data_dir}/subset_merge_swot_duacs_1y.dill'

    def create_training_data(self):

        if self.compute_data:
            print('Create training data')
            self.ds = xr.open_dataset(self.swot_duacs_fname)

            # fix time gaps
            start = np.datetime64(self.ds.time[0].data)
            end   = np.datetime64(self.ds.time[-1].data)
            time_arr = np.arange(start, end, dtype='datetime64[D]')\
                         .astype('datetime64[ns]')

            print('interpolating...')
            self.ds = self.ds.interp(time=time_arr,
                                     method='linear')
            print('interpolating... done')
            data_HR = self.ds.ssha.data
            data_LR = self.ds.sla.data
            time = self.ds.time

            print(f'writing to {self.dill_file}')
            container = {'data_HR' : data_HR,
                         'data_LR' : data_LR,
                         'time' : time}

            with open(self.dill_file, 'wb') as file:
                dill.dump(container, file)
        else:
            print('Load training data from dill file')
            with open(self.dill_file, 'rb') as file:
                data = dill.load(file)
                data_HR = data['data_HR']
                data_LR = data['data_LR']
                time = data['time']

        # define mask
        mask = np.where(np.isnan(data_LR[0,]), 0, 1)
        mask = torch.tensor(mask)[None, :, :, None]

        Nt, Nlat, Nlon = data_HR.shape
        assert data_LR.shape == data_HR.shape

        # remove nans in LR data only
        data_LR = np.nan_to_num(data_LR, 0)

        #  We scale the data with a single scaling for all features
        scaler = CustomScaler('minmax_over_all_features')

        data_HR = scaler.fit_transform(data_HR.reshape(Nt, -1))\
                        .reshape(Nt, Nlat, Nlon)
        scaler.shift = np.nan_to_num(scaler.shift, nan=0.0)
        scaler.scale = np.nan_to_num(scaler.scale, nan=1.0)
        data_LR = scaler.transform(data_LR.reshape(Nt, -1))\
                        .reshape(Nt, Nlat, Nlon)

        split = int(Nt * self.split_factor)
        if split == Nt:
            raise NotImplementedError("unit split_factor")

        # assemble into dicts
        data = {}

        # add channel dimension
        data['HR'] = data_HR.reshape(*data_HR.shape, 1)
        data['LR'] = data_LR.reshape(*data_LR.shape, 1)
        data['time'] = time

        params = {}
        params['train_range'] = range(0, split)
        params['test_range'] = range(split, Nt)
        params['mask'] = mask

        scalers = {}
        scalers['HR'] = scaler
        scalers['LR'] = scaler

        return data, params, scalers, {}
