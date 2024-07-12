import numpy as np
import xarray as xr
import xesmf as xe

data_dir = 'data'

HR_data_file = (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
                f'uo-vo_4.23E-7.78E_56.81N-58.70N_2023-01-01-2023-05-01.nc')

HR_bathy_file = (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
                 f'static_multi-vars_4.23E-7.78E_56.81N-58.70N_0.49-643.57m.nc')

LR_data_file = (f'{data_dir}/cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i_'
                f'uo-vo_4.22E-7.78E_56.80N-58.67N_2023-01-01-2023-05-01.nc')

def build_grid(ds, mask=None):
    lat_arr = ds.latitude
    lon_arr = ds.longitude
    Nlat = lat_arr.shape[0]
    Nlon = lon_arr.shape[0]
    lat_grid = np.tile(lat_arr, (Nlon,1)).T
    lon_grid = np.tile(lon_arr, (Nlat,1))
    grid = {}
    grid['N'] = Nlat
    grid['M'] = Nlon
    grid['lat'] = np.ascontiguousarray(lat_grid)
    grid['lon'] = np.ascontiguousarray(lon_grid)
    if mask is not None:
        grid['mask'] = mask
    return grid

def load_data():
    bt_HR = xr.open_dataset(HR_bathy_file)
    ds_HR = xr.open_dataset(HR_data_file)
    ds_LR = xr.open_dataset(LR_data_file)

    grid_HR = build_grid(ds_HR, bt_HR.mask[0,:,:])
    grid_LR = build_grid(ds_LR)

    interp_HR_LR = xe.Regridder(grid_HR, grid_LR, "bilinear",
                                extrap_method="inverse_dist")
    interp_LR_HR = xe.Regridder(grid_LR, grid_HR, "bilinear",
                                extrap_method="inverse_dist")

    da_HR = ds_HR.uo.rename({'longitude':'lon',
                             'latitude':'lat'})

    da_HR = da_HR.fillna(0.0)

    da_HR_LR = interp_HR_LR(da_HR.values)
    da_HR_LR_HR_tmp = interp_LR_HR(da_HR_LR)

    da_HR_LR = xr.DataArray(da_HR_LR, dims=['time','lat','lon'],
                            coords={'time':ds_HR.time,
                                    'lat':ds_LR.latitude.values,
                                    'lon':ds_LR.longitude.values})

    da_HR_LR_HR = xr.zeros_like(da_HR)
    da_HR_LR_HR[:,:,:] = da_HR_LR_HR_tmp
    da_LR = da_HR_LR_HR.fillna(0.0)

    return da_HR, da_LR

da_HR, da_LR = load_data()
