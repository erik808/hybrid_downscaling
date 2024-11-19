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

X_HR = da_HR.fillna(0.0).values.reshape(Nt,-1, order='C')
X_LR = da_HR_LR_HR.fillna(0.0).values.reshape(Nt,-1, order='C')

model_type = 'ESNc'
control_scaling = 1

sec_pred = lambda x_HR, x_HR_m: 2*x_HR-x_HR_m

if (model_type == 'DMDc' or
    model_type == 'ESNc'):
    secPred = sec_pred(X_HR[train_range,:],
                       X_HR[train_range_m,:]) - X_LR[train_range_p,:]
    
    trainU = np.hstack((X_HR[train_range,:]-X_LR[train_range,:],
                        (secPred) * control_scaling))
elif (model_type == 'DMD' or
      model_type == 'ESN'):
    raise Exception('not implemented')
      # trainU = X_HR[train_range,:]
elif model_type == 'corr_only':
    raise Exception('not implemented')
     # trainU = X_LR[train_range_p,:]

trainY = X_HR[train_range_p,:] - X_LR[train_range_p,:]

esn_pars = {}
if (model_type == 'DMD' or
    model_type == 'DMDc' or
    model_type == 'corr_only'):
    esn_pars['dmdMode'] = True
else:
    esn_pars['dmdMode'] = False
    
if (model_type == 'DMD' or
    model_type == 'DMDc' or
    model_type == 'corr_only' or
    model_type == 'ESNc'):
    esn_pars['feedThrough'] = True
else:
    esn_pars['feedThrough'] = False

if model_type == 'ESNc':
    esn_pars['ftRange'] = range(X_HR.shape[1],
                                X_HR.shape[1] + X_LR.shape[1])
    
esn_pars['scalingType']        = 'none'
esn_pars['Nr']                 = 10000
esn_pars['rhoMax']             = 0.4
esn_pars['alpha']              = 0.2
esn_pars['entriesPerRow']      = 3
esn_pars['tikhonov_lambda']    = 1e-3
esn_pars['squaredStates']      = 'even'
esn_pars['inputMatrixType']    = 'balancedSparse'
esn_pars['fCutoff']            = 0.01

esn = ESN(esn_pars['Nr'], trainU.shape[1], trainY.shape[1])
esn.setPars(esn_pars)
esn.initialize()
esn.train(trainU, trainY)

Npred = Nt-T
N = Nlat * Nlon
predY = np.zeros((Npred, N))
esn_state = esn.X[-1,:].copy()

# initial state for the predictions
xk_HR = X_HR[init_idx, :]
xk_HR_m = X_HR[init_idx-1, :]

for i in range(4*24*4):
    xk_LR = X_LR[init_idx+i, :]
    xk_LR_p = X_LR[init_idx+i+1, :]
    dxk = xk_HR - xk_LR    
    Pdxk = (sec_pred(xk_HR, xk_HR_m) - xk_LR_p) \
        * control_scaling
    if (model_type == 'DMDc' or
        model_type == 'ESNc' ):
        u_in = np.append(dxk.squeeze(), Pdxk.squeeze())
    elif model_type == 'DMD':
        raise Exception('not implemented')
        # u_in = yk.squeeze()
    elif model_type == 'corr_only':
        raise Exception('not implemented')
        # u_in = Pyk.squeeze()

    u_in       = np.expand_dims(u_in, axis=0)
    u_in       = esn.scaleInput(u_in)
    esn_state  = esn.update(esn_state, u_in)
    u_out      = esn.apply(esn_state, u_in)
    u_out      = np.expand_dims(u_out, axis=0)
    yk         = esn.unscaleOutput(u_out)

    xk_HR_m = xk_HR
    xk_HR = xk_LR_p + yk

    predY[i,:] = yk
    print(f'{i}')

pmodel = xr.zeros_like(da_HR[test_range,:,:])
pmodel[:,:,:] = np.reshape(predY[:,:], (-1, Nlat, Nlon))

dsX_HR = xr.zeros_like(da_HR[test_range,:,:])
dsX_HR[:,:,:] = np.reshape(X_HR[test_range,:], (-1, Nlat, Nlon))

dsX_LR = xr.zeros_like(da_HR[test_range,:,:])
dsX_LR[:,:,:] = np.reshape(X_LR[test_range,:], (-1, Nlat, Nlon))

dmodel  = dsX_HR - pmodel - dsX_LR

ds = {}
ds['1'] = dsX_HR.rename('X_HR')
ds['2'] = dsX_LR.rename('X_LR')
ds['3'] = (dsX_LR + pmodel).rename(f'p{model_type} + X_LR')
ds['4'] = pmodel.rename(f'p{model_type}')
ds['5'] = dmodel.rename(f'd{model_type}')
ds['6'] = (dsX_HR-dsX_LR).rename(f'X_HR - X_LR')

ds['1']['vmin'] = -1;   ds['1']['vmax'] = 1
ds['2']['vmin'] = -1;   ds['2']['vmax'] = 1
ds['3']['vmin'] = -1;   ds['3']['vmax'] = 1
ds['4']['vmin'] = -1/3; ds['4']['vmax'] = 1/3
ds['5']['vmin'] = -1/3; ds['5']['vmax'] = 1/3
ds['6']['vmin'] = -1/3; ds['6']['vmax'] = 1/3

def plot_frame(i):

    def plot_wrapper(ds, i):
        ds[i,:,:].plot.pcolormesh(vmin=ds.vmin, vmax=ds.vmax,
                                  center=0, cmap='RdBu',
                                  extend='both',
                                  cbar_kwargs={'label':''})
        plt.gca().set_title(ds.name)
        plt.gca().set_xlabel('')
        plt.gca().set_ylabel('')

    plt.clf()
    for p in range(0,6):
        plt.subplot(3,2,p+1)
        plot_wrapper(ds[f'{p+1}'], i)

    plt.suptitle(f"{np.datetime64(ds['1'].time[i].values, 'h')}")
    frame_name = f'output/frame-{i:06d}.png'
    plt.savefig(frame_name)

fig, axs = plt.subplots(2, 2, figsize=(16, 12),
                        sharex=True, sharey=True)
tic = time.time()

with Pool(8) as p:
    p.map(plot_frame, range(0,4*24*4,1))

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
movie_name = f'movie_{model_type}_{timestamp}.mov'
framerate = 24
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
print('\a')
