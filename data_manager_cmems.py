import os
import dill
import dask_image.ndfilters as ndf
import tools
import xesmf as xe
import importlib
import xarray as xr
from dask.diagnostics import ProgressBar
import data_manager_base

importlib.reload(tools)
importlib.reload(data_manager_base)


class DataManagerCMEMS(data_manager_base.DataManagerBase):
    """ Managing input and output """

    def __init__(
            self,
            experiment_id="test",
            testing=False,
            force_rebuild=False,
            config_name='data_config_cmems',
    ):
        super().__init__()
        tools.load_config(self, config_name=config_name)
        self.config_name = config_name
        self.force_rebuild = force_rebuild
        self.dirs, self.files = self.setup_directories(
            experiment_id=experiment_id)

        # setup test mode
        self.testing = testing
        if self.testing:
            self.time_range = self.time_range_testing

        self.load_grid()
        self.load_scalers()

        # create_training_data needs to be called
        self.ready = False

    def create_training_data(self):
        print('load HR data')
        self.load_HR_data()
        print('load LR data')
        self.load_LR_data(force_rebuild=self.force_rebuild)
        self.create_ranges()
        self.ready = True

    def create_ranges(self):
        T = len(self.ds_LR.time)
        self.split_index = int(self.split_factor * T)
        self.train_range = slice(0, self.split_index)
        self.test_range = slice(self.split_index, T)

    def load_grid(self):
        # 3d mask
        self.mask = self.crop(xr.open_dataset(self.bathy_file).mask)

        self.grid_HR = tools.build_grid(self.mask.latitude,
                                        self.mask.longitude)

        self.grid_LR = tools.create_coarse_grid(self.mask.latitude,
                                                self.mask.longitude,
                                                self.coarsening_factor)

        self.bilin_downsampler = xe.Regridder(self.grid_HR,
                                              self.grid_LR,
                                              "bilinear",
                                              extrap_method="inverse_dist")

        self.bilin_upsampler = xe.Regridder(self.grid_LR,
                                            self.grid_HR,
                                            "bilinear",
                                            extrap_method="inverse_dist")

    def load_coords(self):
        coords = xr.open_dataset(self.coords_file)
        l_ = [self.crop(coords[var]) for var in coords]
        coords = xr.merge(l_)
        return coords

    def load_HR_data(self):
        self.ds_HR = xr.open_zarr(self.data_files, consolidated=True)
        self.ds_HR = self.process_ds(self.ds_HR)

    def load_LR_data(self, force_rebuild=False):

        path = self.coarse_data_file
        if not force_rebuild and os.path.exists(path):
            self.ds_LR = xr.open_zarr(path)
        else:
            print('Create and export coarse data')
            self.ds_LR = self.create_LR_data(export=True)

        # restrict to time range
        self.ds_LR = self.ds_LR.sel(time=self.time_range)

    def create_LR_data(self, export=True):

        data_LR = []
        for key in self.data_vars:
            filtered = ndf.gaussian_filter(
                self.ds_HR[key].fillna(0.0).data,
                sigma=self.sigma)
            data_LR.append(self.bilin_downsampler(filtered))

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
            'time': 64,
            'latitude': -1,
            'longitude': -1,
        })

        if export:
            encoding = {var: {"compressor": None}
                        for var in self.ds_LR.data_vars}
            with ProgressBar():
                self.ds_LR.to_zarr(self.coarse_data_file,
                                   encoding=encoding,
                                   consolidated=True,
                                   mode='w')

        return self.ds_LR

    def create_scalers(self, export=True):
        """for the HR set (2 years) this should take about 5 minutes"""
        print('creating scalers...')
        self.create_training_data()
        scalers = {}
        scalers['HR'] = tools.create_scaler(self.ds_HR,
                                            self.scaling_range,
                                            self.scaling_type)
        scalers['LR'] = tools.create_scaler(self.ds_LR,
                                            self.scaling_range,
                                            self.scaling_type)

        if export:
            with open(self.scalers_file, 'wb') as file:
                dill.dump(scalers, file)

        return scalers

    def load_scalers(self):
        self.scalers = None
        if not os.path.exists(self.scalers_file) or self.force_rebuild:
            print('Creating scalers')
            self.create_scalers(export=True)

        with open(self.scalers_file, 'rb') as file:
            self.scalers = dill.load(file)
        return self.scalers

    def process_ds(self, ds):
        """ select datavars and cropping """
        ds_out = ds[self.data_vars]
        ds_out = ds_out.isel(
            latitude=self.lat_crop,
            longitude=self.lon_crop,
        )
        ds_out = ds_out.sel(time=self.time_range)
        return ds_out

    def crop(self, input_field):
        """crop fields to 64 x 128 (assuming we're getting 69 x 129)"""
        return input_field[..., self.lat_crop, self.lon_crop]
