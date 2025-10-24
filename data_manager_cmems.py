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

    def load_uv_data(self):
        # restrict to chosen time range
        self.uv_data_files = \
            tools.apply_time_range(self.uv_data_files, self.time_range)
        self.uv_ds = xr.open_mfdataset(self.uv_data_files,
                                       parallel=False,
                                       preprocess=self.uv_preprocess,
                                       )

        self.uv_ds = self.uv_ds.chunk({
            'time': 192,
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

        self.regridder = xe.Regridder(self.grid_HR, self.grid_LR,
                                      "bilinear",
                                      extrap_method="inverse_dist")

    def create_coarse_uv_data(self):
        uo_filtered = ndf.gaussian_filter(
            self.uv_ds.uo.fillna(0.0).data,
            sigma=self.sigma)

        vo_filtered = ndf.gaussian_filter(
            self.uv_ds.vo.fillna(0.0).data,
            sigma=self.sigma)

        uo_LR = self.regridder(uo_filtered)
        vo_LR = self.regridder(vo_filtered)

        da_list = [
            xr.DataArray(data,
                         dims=['time', 'latitude', 'longitude'],
                         coords={'time': self.uv_ds.time,
                                 'latitude': self.grid_LR['lat'][..., 0],
                                 'longitude': self.grid_LR['lon'][0,],
                                 },
                         name=name,
                         )
            for data, name in zip([uo_LR, vo_LR], ['uo', 'vo'])]

        self.uv_ds_LR = xr.merge(da_list)

        tools.ds_to_netcdf(self.uv_ds_LR, self.coarse_data_files)

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
dmgr_cmems.create_coarse_uv_data()

# with ProgressBar():
#     dmgr_cmems.uv_ds_LR.compute()
