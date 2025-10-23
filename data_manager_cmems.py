import dask
import dask_image.ndfilters as ndf
import tools
import importlib
import xarray as xr
from scipy.ndimage import gaussian_filter
from data_manager_base import DataManagerBase

importlib.reload(tools)


class DataManagerCMEMS(DataManagerBase):

    def __init__(self):
        super().__init__()
        tools.load_config(self, config_name='data_config_cmems')

    def create_training_data(self):
        pass

    def load_uv_data(self):
        # restrict to chosen time range
        self.uv_data_files = \
            tools.apply_time_range(self.uv_data_files, self.time_range)
        self.uv_ds = xr.open_mfdataset(self.uv_data_files,
                                       parallel=False,
                                       preprocess=self.uv_preprocess)

    def load_grid(self):
        self.mask = self.crop(xr.open_dataset(self.bathy_file).mask)
        self.grid_HR = tools.build_grid(self.mask.latitude,
                                        self.mask.longitude)
        breakpoint()
        self.grid_LR = tools.build_grid(self.mask.latitude,
                                        self.mask.longitude)

    def create_coarse_uv_data(self):
        breakpoint()
        # - apply gaussian filter
        # - apply downsampling
        # - store results

        pass  # # todo

    def uv_preprocess(self, ds):
        """ select datavars and cropping """
        ds_out = ds[['uo', 'vo']]
        ds_out = ds_out.isel(
            latitude=self.lat_crop,
            longitude=self.lon_crop,
        )
        return ds_out

    def crop(self, input_field):
        """crop fields to 64 x 128 (assuming we're getting 69 x 129)"""
        return input_field[..., self.lat_crop, self.lon_crop]


dmgr_cmems = DataManagerCMEMS()
dmgr_cmems.load_uv_data()
dmgr_cmems.load_grid()
