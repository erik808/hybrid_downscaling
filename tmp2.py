import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xesmf as xe
import time
from xmovie import Movie
from multiprocess import Pool
from dask.diagnostics import ProgressBar
from ESN import ESN

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

ufield_HR = ds_HR.uo.rename({'longitude':'lon',
                             'latitude':'lat'})

u_HR_LR = interp_HR_LR(ufield_HR.values)
u_HR_LR_HR = interp_LR_HR(u_HR_LR)

da_HR_LR = xr.DataArray(u_HR_LR, dims=['time','lat','lon'],
                        coords={'time':ds_HR.time,
                                'lat':ds_LR.latitude.values,
                                'lon':ds_LR.longitude.values})

da_HR_LR_HR = xr.zeros_like(ufield_HR)
da_HR_LR_HR[:,:,:] = u_HR_LR_HR











raise Exception('exceptional')


def plot_frame(i):

    def plot_wrapper(ds, i):
        ds[i,:,:].plot.pcolormesh(vmin=ds.vmin, vmax=ds.vmax,
                                  center=0, cmap='RdBu',
                                  extend='both')
        plt.gca().set_title('')

    plt.clf()
    plt.subplot(2,2,1)
    plot_wrapper(ds1, i)
    plt.subplot(2,2,2)
    plot_wrapper(ds2, i)
    plt.subplot(2,2,3)
    plot_wrapper(ds3, i)
    plt.subplot(2,2,4)
    plot_wrapper(ds4, i)
    # plt.tight_layout()
    plt.suptitle(f"{np.datetime64(ds1.time[i].values, 'h')}")
    frame_name = f'output/frame-{i:06d}.png'
    plt.savefig(frame_name)

# xmovie parallel movie creation is about 5 times slower than

# timeslice=slice('2023-01-01','2023-01-08',2)
# tic = time.time()    
# mov = Movie(ufield_HR.chunk({'time':1})\
#             .sel(time=timeslice),
#             vmin=-1, vmax=1)

# mov.save('movie.mov',
#          overwrite_existing=True,
#          parallel=True)

# toc = time.time()
# print(f'elapsed: {toc-tic:02f}')

fig, axs = plt.subplots(2, 2, figsize=(16, 12),
                        sharex=True, sharey=True)

tic = time.time()
timeslice=slice('2023-01-01','2023-01-08',2)
ds1 = ufield_HR.sel(time=timeslice)
ds1['vmin'] = -1; ds1['vmax'] = 1
ds2 = da_HR_LR.sel(time=timeslice)
ds2['vmin'] = -1; ds2['vmax'] = 1
ds3 = da_HR_LR_HR.sel(time=timeslice)
ds3['vmin'] = -1; ds3['vmax'] = 1
ds4 = (da_HR_LR_HR-ufield_HR).sel(time=timeslice)
ds4['vmin'] = -1/3.; ds4['vmax'] = 1/3.

with Pool(8) as p:
    p.map(plot_frame, range(len(ds1.time)))

movie_name = 'movie.mov'
framerate=24
sys_cmd = ( f"ffmpeg -r {framerate} -f image2 -pattern_type glob -i "
            f"'output/frame-*.png' "
            f"-vcodec libx264 -crf 25  -pix_fmt yuv420p -y "
            f"output/{movie_name}" )

print(sys_cmd)
os.system(sys_cmd)
sys_cmd = ( f"rm output/frame-*.png")
print(sys_cmd)
os.system(sys_cmd)

toc = time.time()
print(f'elapsed: {toc-tic:02f}')
