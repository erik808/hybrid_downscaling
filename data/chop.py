import dask
import xarray as xr
import os
from dask.diagnostics import ProgressBar

# convert big yearly file to monthly files, adapted from the
# save_mfdataset example

# compression level (0-9)
complevel = 5

filename = 'cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_uo-vo_4.23E-7.78E_56.81N-58.70N_2023-01-01-2023-12-31.nc'
basename = f'{filename[:-19]}_test'

cmd = f'mkdir -p {basename}'
print(cmd)
os.system(cmd)

ds = xr.open_dataset(filename, chunks={'time' : 'auto'})

months, datasets = zip(*ds.groupby('time.month'))

paths = [f"{basename}/{basename}{m:02d}.nc" for m in months]

options = {"zlib": True, "complevel": complevel}

encoding = {var : options for var in list(ds.keys())}

with ProgressBar():
    xr.save_mfdataset(datasets, paths, encoding=encoding)
