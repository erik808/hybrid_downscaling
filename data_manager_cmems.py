import numpy as np
import os
import dill
import dask_image.ndfilters as ndf
import tools
import xesmf as xe
import importlib
import xarray as xr

from data_manager_base import DataManagerBase

importlib.reload(tools)


class DataManagerCMEMS(DataManagerBase):

    def __init__(self):
        super().__init__()
        tools.load_config(self, config_name='data_config_cmems')

    def create_training_data(self):
        pass

    def load_HR_data(self):
        # restrict to chosen time range
        self.data_files = \
            tools.apply_time_range(self.data_files, self.time_range)

        self.ds_HR = xr.open_mfdataset(self.data_files,
                                       parallel=True,
                                       preprocess=self.preprocess,
                                       )

        self.ds_HR = self.ds_HR.chunk({
            'time': 4 * 24 * 14,
            'latitude': -1,
            'longitude': -1,
        })

    def load_grid(self):
        self.mask = self.crop(xr.open_dataset(self.bathy_file).mask)
        self.grid_HR = tools.build_grid(self.mask.latitude,
                                        self.mask.longitude)

        self.grid_LR = tools.create_coarse_grid(self.mask.latitude,
                                                self.mask.longitude,
                                                self.coarsening_factor)

        self.regridder = xe.Regridder(self.grid_HR,
                                      self.grid_LR,
                                      "bilinear",
                                      extrap_method="inverse_dist")

    def load_LR_data(self, force_rebuild=False):

        paths = tools.coarse_data_paths(self.ds_HR,
                                        path=self.coarse_data_files,
                                        prefix=self.coarse_data_prefix,
                                        )

        if not force_rebuild and np.all([os.path.exists(p) for p in paths]):
            print('Loading coarse data')

            self.ds_LR = xr.open_mfdataset(paths,
                                           parallel=True,
                                           )
        else:
            print('Create and export coarse data')
            self.ds_LR = self.create_LR_data(export=True)

    def create_LR_data(self, export=True):

        data_LR = []
        for key in self.data_vars:
            filtered = ndf.gaussian_filter(
                self.ds_HR[key].fillna(0.0).data,
                sigma=self.sigma)
            data_LR.append(self.regridder(filtered))

        da_list = [
            xr.DataArray(data,
                         dims=['time', 'latitude', 'longitude'],
                         coords={'time': self.ds_HR.time,
                                 'latitude': self.grid_LR['lat'][..., 0],
                                 'longitude': self.grid_LR['lon'][0,],
                                 },
                         name=name,
                         attrs=self.ds_HR[name].attrs,
                         )
            for data, name in zip(data_LR, self.data_vars)]

        self.ds_LR = xr.merge(da_list)

        # chunk only in time
        self.ds_LR = self.ds_LR.chunk({
            'time': 2976,
            'latitude': -1,
            'longitude': -1,
        })

        if export:
            tools.ds_to_netcdf(self.ds_LR,
                               path=self.coarse_data_files,
                               prefix=self.coarse_data_prefix,
                               )
        return self.ds_HR_LR

    def create_scalers(self, export=True):

        scalers = {}
        scalers['HR'] = tools.create_scaler(self.ds_HR,
                                            self.scaling_range)
        scalers['LR'] = tools.create_scaler(self.ds_LR,
                                            self.scaling_range)

        if export:
            with open(self.scalers_file, 'wb') as file:
                dill.dump(scalers, file)

        return scalers

    def preprocess(self, ds):
        """ select datavars and cropping """
        ds_out = ds[self.data_vars]
        ds_out = ds_out.isel(
            latitude=self.lat_crop,
            longitude=self.lon_crop,
        )
        return ds_out

    def crop(self, input_field):
        """crop fields to 64 x 128 (assuming we're getting 69 x 129)"""
        return input_field[..., self.lat_crop, self.lon_crop]


dmgr_cmems = DataManagerCMEMS()
dmgr_cmems.load_HR_data()
dmgr_cmems.load_grid()
dmgr_cmems.load_LR_data(force_rebuild=False)
dmgr_cmems.create_scalers()
