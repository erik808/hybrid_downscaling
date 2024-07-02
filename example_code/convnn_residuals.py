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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import xbatcher

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
da_HR_LR_HR = da_HR_LR_HR.fillna(0.0)

Nt, Nlat, Nlon = da_HR_LR_HR.shape
Nt, Nlat_LR, Nlon_LR = da_HR_LR.shape
T = int(Nt * 3 /4.)
train_range_m = range(0,T-1)
train_range = range(1,T)
init_idx = train_range[-1]
test_range = range(init_idx+1, Nt)

sec_pred = lambda x_HR, x_HR_m: 2*x_HR-x_HR_m

# residual, secant prediction
res_secp_da = xr.zeros_like(da_HR)
res_secp_arr = sec_pred(da_HR[1:,:,:].values,
                        da_HR[:-1,:,:].values)\
                        - da_HR_LR_HR[1:,:,:].values
res_secp_da[1:,:,:] = res_secp_arr

# true residual
res_true_da = da_HR - da_HR_LR_HR

def normalize(x):
    minval = np.nanmin(x)
    maxmin = np.nanmax(x) - minval
    out = ((x - minval) / maxmin)
    return out, minval, maxmin

res_secp_da, minval_rp, maxmin_rp = normalize(res_secp_da)
res_true_da, minval_rt, maxmin_rt = normalize(res_true_da)

# Create simple convolutional network
class ConvNN(nn.Module):

    def __init__(self):

        super(ConvNN, self).__init__()

        self.relu = nn.LeakyReLU(0.01)
        self.kernel_size = (3,3)
        self.conv1 = nn.Conv2d(1,32,stride=(1,1),
                               kernel_size=self.kernel_size,
                               padding=1)

        self.conv2 = nn.Conv2d(32,64,stride=(1,1),
                               kernel_size=self.kernel_size,
                               padding=1)

        self.conv3 = nn.Conv2d(64,64,stride=(1,1),
                               kernel_size=self.kernel_size,
                               padding=1)

        self.convT1 = nn.ConvTranspose2d(64,32,stride=(1,1),
                                         kernel_size=self.kernel_size,
                                         padding=1)

        self.convT2 = nn.ConvTranspose2d(32,1, stride=(1,1),
                                         kernel_size=self.kernel_size,
                                         padding=1)
        self.sigm = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.convT1(x)
        x = self.relu(x)
        x = self.convT2(x)
        x = self.sigm(x)
        return x

model = ConvNN()
# model.load_state_dict(torch.load('ConvNN.pth'))

# Create data loaders for pytorch using xbatcher
# truth, training period
yb_train = xbatcher.BatchGenerator(res_true_da[train_range,:,:],
                                   input_dims = {'lat':69,
                                                 'lon':129},
                                   input_overlap = {'time' : 2},
                                   batch_dims = {'time' : 10})

# input, training period
xb_train = xbatcher.BatchGenerator(res_secp_da[train_range,:,:],
                                   input_dims = {'lat':69,
                                                 'lon':129},
                                   input_overlap = {'time' : 2},
                                   batch_dims = {'time' : 10})

yb_test = xbatcher.BatchGenerator(res_true_da[test_range,:,:],
                                   input_dims = {'lat':69,
                                                 'lon':129},
                                   input_overlap = {'time' : 0},
                                   batch_dims = {'time' : 1})

# input, testing period
xb_test = xbatcher.BatchGenerator(res_secp_da[test_range,:,:],
                                   input_dims = {'lat':69,
                                                 'lon':129},
                                   input_overlap = {'time' : 0},
                                   batch_dims = {'time' : 1})

# conversion of xbatcher batch to pytorch tensor
# this should be part of the xbatcher object... somehow TODO #FIXME
def to_tensor(xb):
    x = xb.astype('float32')\
          .expand_dims(dim={'c':1},axis=1)\
          .torch.to_tensor()
    return x

# construct the optimizer
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# define loss function
loss_fn = F.mse_loss
start_time = time.time()

# logging
log_dict = {'training loss per batch' : [],
            'training loss per epoch' : []}

epochs = 1
verbosity = 20
do_training=True
# model.load_state_dict(torch.load('ConvNN.pth'))
if do_training:
    for epoch_i in range(epochs):

        # put the model in train mode
        model.train()

        for batch_i, (xb, yb) in enumerate(zip(xb_train, yb_train)):
            x = to_tensor(xb)
            y = to_tensor(yb)
            o = model(x)

            # zero the gradients in the optimizer
            optimizer.zero_grad()

            # compute the loss, model output: o, truth: y
            loss = loss_fn(o, y, reduction='sum')

            # compute gradients
            loss.backward()

            # update model parameters
            optimizer.step()

            # do some logging:
            log_dict['training loss per batch'].append(loss.item())

            if not batch_i % verbosity:
                print(f'Epoch: {epoch_i}/{epochs} |'
                      f' Batch {batch_i}/{len(xb_train)} |'
                      f' Loss {loss}')
        print(f'time elapsed: {(time.time()-start_time)/60}m')
    print(f'total time elapsed: {(time.time()-start_time)/60}m')
    torch.save(model.state_dict(), 'ConvNN.pth')
else:
    model.load_state_dict(torch.load('ConvNN.pth'))



for batch_i, (xb, yb) in enumerate(zip(xb_test, yb_test)):
    x = to_tensor(xb)
    y = to_tensor(yb)
    o = model(x)
    # compute the loss, model output: o, truth: y
    loss = loss_fn(o, y, reduction='sum')
    print(loss)
    break
    
    
with torch.set_grad_enabled(False):
    model.eval()

    init_idx = train_range[-1]
    xk_HR = da_HR[init_idx, :, :].values
    xk_HR_m = da_HR[init_idx-1, :, :].values

    predY = xr.zeros_like(da_HR[test_range,:,:])

    for i in range(4*24*4):

        xk_LR = da_HR_LR_HR[init_idx+i, :, :].values
        xk_LR_p = da_HR_LR_HR[init_idx+i+1, :, :].values
        xk_HR_p = da_HR[init_idx+i+1, :, :].values

        res_secp = sec_pred(xk_HR, xk_HR_m) - xk_LR_p
        res_true = xk_HR_p - xk_LR_p

        res_secp = (res_secp - minval_rt) / maxmin_rt
        res_true = (res_true - minval_rt) / maxmin_rt

        x = torch.tensor(np.expand_dims(res_secp.astype('float32'),
                                        axis=(0,1)))
        y = torch.tensor(np.expand_dims(res_true.astype('float32'),
                                        axis=(0,1)))
        o = model(x)
        loss = loss_fn(o, y, reduction='sum')
        print(loss)

        o_out = o.detach().numpy().squeeze() * maxmin_rt + minval_rt

        xk_HR_m = xk_HR
        xk_HR = xk_LR_p + o_out
        predY[i,:,:] = o_out


model_type='ConvNN'
pmodel = xr.zeros_like(da_HR[test_range,:,:])
pmodel[:,:,:] = predY

dsX_HR = da_HR[test_range,:,:]
dsX_LR = da_HR_LR_HR[test_range,:,:]

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
        
