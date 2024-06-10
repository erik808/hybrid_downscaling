import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xesmf as xe

HR_data_file = ('cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
                'uo-vo_4.23E-7.78E_56.81N-58.70N_2023-01-01-2023-02-01.nc')
LR_data_file = ('cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i_'
                'uo-vo_4.22E-7.78E_56.80N-58.67N_2023-01-01-2023-02-01.nc')

ds_HR = xr.open_dataset(HR_data_file)
ds_LR = xr.open_dataset(LR_data_file)

def build_grid(ds):
    lat_arr = ds.latitude
    lon_arr = ds.longitude
    Nlat = lat_arr.shape[0]
    Nlon = lon_arr.shape[0]
    lat_grid = np.tile(lat_arr, (Nlon,1)).T
    lon_grid = np.tile(lon_arr, (Nlat,1))
    grid = {}
    grid['N'] = Nlat
    grid['M'] = Nlon
    grid['lat'] = lat_grid
    grid['lon'] = lon_grid
    return grid

grid_HR = build_grid(ds_HR)
grid_LR = build_grid(ds_LR)

plt.scatter(grid_HR['lon'], grid_HR['lat'], s=0.01)

tstep=400
ufield_HR = ds_HR.uo[tstep,:,:]

interp_u_HR = LinearNDInterpolator( (grid_HR['lon'].flatten(),
                                     grid_HR['lat'].flatten() ), ufield_HR.values.flatten())

u_HR_LR = interp_u_HR( grid_LR['lon'].flatten(),
                       grid_LR['lat'].flatten() )

u = xr.zeros_like(ds_LR.uo)
u[tstep,:,:] = np.reshape(u_HR_LR, (grid_LR['N'], grid_LR['M']))


plt.close('all');
plt.subplot(2,1,1)
u[tstep,:,:].plot.imshow()
plt.subplot(2,1,2)
ds_LR.uo[tstep+1,:,:].plot.imshow()
plt.pause(1)













