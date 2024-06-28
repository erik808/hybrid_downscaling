import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xesmf as xe
import time
from xmovie import Movie
from multiprocess import Pool
from dask.diagnostics import ProgressBar
from importlib import reload
from ESN.ESN import ESN

# Load training and test data

HR_data_file = ('cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
                'uo-vo_4.23E-7.78E_56.81N-58.70N_2023-01-01-2023-02-01.nc')

HR_bathy_file = ('cmems_mod_nws_phy_anfc_0.027deg-3D_'
                 'static_multi-vars_4.23E-7.78E_56.81N-58.70N_0.49-643.57m.nc')

LR_data_file = ('cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i_'
                'uo-vo_4.22E-7.78E_56.80N-58.67N_2023-01-01-2023-02-01.nc')

bt_HR = xr.open_dataset(HR_bathy_file)

ds_HR = xr.open_dataset(HR_data_file)
ds_LR = xr.open_dataset(LR_data_file)

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

grid_HR = build_grid(ds_HR, bt_HR.mask[0,:,:])
grid_LR = build_grid(ds_LR)

tic = time.time()
interp_HR_LR = xe.Regridder(grid_HR, grid_LR, "bilinear",
                            extrap_method="inverse_dist")
interp_LR_HR = xe.Regridder(grid_LR, grid_HR, "bilinear",
                            extrap_method="inverse_dist")

da_HR = ds_HR.uo.rename({'longitude':'lon',
                         'latitude':'lat'}).fillna(0.0)

da_HR_LR = interp_HR_LR(da_HR.values)
da_HR_LR_HR_tmp = interp_LR_HR(da_HR_LR)

da_HR_LR = xr.DataArray(da_HR_LR, dims=['time','lat','lon'],
                        coords={'time':ds_HR.time,
                                'lat':ds_LR.latitude.values,
                                'lon':ds_LR.longitude.values})

da_HR_LR_HR = xr.zeros_like(da_HR)
da_HR_LR_HR[:,:,:] = da_HR_LR_HR_tmp

Nt, Nlat, Nlon = da_HR_LR_HR.shape
Nt, Nlat_LR, Nlon_LR = da_HR_LR.shape
T = int(Nt * 3 /4.)
train_range_m = range(0,T-1)
train_range = range(1,T)
train_range_p = range(2,T+1)
init_idx = train_range[-1]
test_range = range(init_idx+1, Nt)

# Create simple convolutional network
import torch.nn as nn
class ConvNN(nn.Module):

    def __init__(self):

        super(ConvNN, self).__init__()

        self.relu = nn.LeakyReLU(0.01)
        self.kernel_size = (3,3)
        self.conv1 = nn.Conv2d(1,32,stride=(1,1),
                               kernel_size=self.kernel_size,
                               padding=1)
        self.conv2 = nn.Conv2d(32,1,stride=(1,1),
                               kernel_size=self.kernel_size,
                               padding=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigm(x)
        return x

model = ConvNN()

# Create data loader for pytorch using xbatcher
import xbatcher

xb_HR = xbatcher.BatchGenerator(da_HR,
                                input_dims = {'lat':69,
                                              'lon':129},
                                input_overlap = {'time' : 2},
                                batch_dims = {'time' : 10})

xb_HR_LR_HR = xbatcher.BatchGenerator(da_HR_LR_HR,
                                      input_dims = {'lat':69,
                                                    'lon':129},
                                      input_overlap = {'time' : 2},
                                      batch_dims = {'time' : 10})


# define loss, optimizer all that stuff TODO

# train  TODO
for xb, yb in zip(xb_HR, xb_HR_LR_HR):
    x = xb.astype('float32')\
          .expand_dims(dim={'c':1},axis=1)\
          .torch.to_tensor()
    y = yb.astype('float32')\
          .expand_dims(dim={'c':1},axis=1)\
          .torch.to_tensor()

    
    # z = model(x)

    breakpoint()
