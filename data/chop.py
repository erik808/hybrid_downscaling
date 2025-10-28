import xarray as xr
import os
from dask.diagnostics import ProgressBar

# convert big file to monthly files, adapted from the
# save_mfdataset example

# compression level (0-9)
complevel = 0

# filename = 'cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_uo-vo_4.23E-7.78E_56.81N-58.70N_2023-01-01-2023-12-31.nc'
filename = 'cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_uo-vo-zos_4.22E-7.78E_56.81N-58.69N_2022-12-01-2025-09-30.nc'
basename = f'{filename[:-3]}_zarr'

cmd = f'mkdir -p {basename}'
print(cmd)
os.system(cmd)

ds = xr.open_dataset(filename, chunks={'time': 288})  # 3 days
ds.to_zarr(f'{basename}/data.zarr')

# def nested_groupby(ds):
#     datasets = []
#     keys = []
#     # def export_coarse_data(ds):
#     for year_key, ds_year in ds.groupby('time.year'):
#         for month_key, ds_month in ds_year.groupby('time.month'):
#             keys.append([year_key, month_key])
#             datasets.append(ds_month)

#     return keys, datasets


# keys, datasets = nested_groupby(ds)
# # zip(*ds.groupby(['time.year', 'time.month']))

# paths = [f"{basename}/{y:04d}-{m:02d}.nc" for y, m in keys]

# options = {"zlib": True, "complevel": complevel}

# encoding = {var : options for var in list(ds.keys())}

# with ProgressBar():
#     xr.save_mfdataset(datasets, paths, encoding=encoding)
