from importlib import reload

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xesmf as xe
import os
import time
from multiprocess import Pool
from dask.diagnostics import ProgressBar
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

import xbatcher

import data_manager as dm
reload(dm)

scaler = MinMaxScaler(feature_range=(0,1))
data_HR = scaler.fit_transform(dm.da_HR.values.reshape(dm.Nt, -1))\
                .reshape(dm.Nt, dm.Nlat, dm.Nlon)
data_LR = scaler.transform(dm.da_LR.values.reshape(dm.Nt, -1))\
                .reshape(dm.Nt, dm.Nlat, dm.Nlon)

T = int(dm.Nt * 4 / 5.)
look_back = 3
train_range_m = range(look_back,T-1)
train_range = range(look_back+1,T)
init_idx = train_range[-1]
test_range = range(init_idx+1, dm.Nt)

def create_lookback(da, look_back):
    Tdim = da.shape[0]
    da_with_lookback = da.expand_dims(dim={'lookback':look_back},
                                      axis=1).copy()

    for i in range(look_back):
        da_with_lookback[look_back:,i,:,:] = \
            da[(look_back-i):Tdim-i,:,:].values

    return da_with_lookback


fun_secant = lambda x_HR, x_HR_m, x_LR_p: 2 * x_HR - x_HR_m - x_LR_p

R_secant = xr.zeros_like(dm.da_HR)

da_LR = xr.zeros_like(dm.da_LR)
da_LR[:,:,:] = data_LR


R_secant_values = fun_secant(data_HR[1:,:,:],
                             data_HR[:-1,:,:],
                             data_LR[1:,:,:])

R_secant[1:,:,:] = R_secant_values
R_res = xr.zeros_like(dm.da_HR)
R_res[:,:,:] = data_HR - data_LR

R_res = create_lookback(R_res, look_back)
R_secant = create_lookback(R_secant, look_back)

batch_size = 100
overlap = 5

Y_train = xbatcher.BatchGenerator(R_res[train_range,:,:],
                                  input_dims = {'lat':69,
                                                'lon':129},
                                  input_overlap = {'time' : overlap},
                                  batch_dims = {'time' : batch_size})

U_train = xbatcher.BatchGenerator(R_res[train_range_m,:,:],
                                  input_dims = {'lat':69,
                                                'lon':129},
                                  input_overlap = {'time' : overlap},
                                  batch_dims = {'time' : batch_size})

# conversion of xbatcher batch to pytorch tensor
# this should be part of the xbatcher object... somehow TODO #FIXME
def to_tensor(xb):
    x = xb.astype('float32')\
          .torch.to_tensor()
    return x

class MyLSTM(nn.Module):
    def __init__(self, input_size=1):
        super(MyLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = 1000
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = MyLSTM(input_size=dm.Nlat*dm.Nlon)

learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# define loss function
loss_fn = F.mse_loss

# logging
log_dict = {'training loss per batch' : []}

epochs = 5
verbosity = 10

start_time=time.time()

print('start training')
for epoch_i in range(epochs):
    model.train()
    for batch_i, (ub, yb) in enumerate(zip(U_train, Y_train)):
        ub = to_tensor(ub).view(batch_size, look_back, -1)
        yb = to_tensor(yb).view(batch_size, look_back, -1)
        ob = model(ub)

        optimizer.zero_grad()

        loss = loss_fn(ob, yb, reduction='sum')

        loss.backward()

        optimizer.step()

        log_dict['training loss per batch'].append(loss.item())
        if not batch_i % verbosity:
            print(f'Epoch: {epoch_i+1}/{epochs} |'
                  f' Batch {batch_i+1}/{len(U_train)} |'
                  f' Loss {loss}')
    print(f'time elapsed: {(time.time()-start_time)/60}m')
print(f'total time elapsed: {(time.time()-start_time)/60}m')


pred_steps = len(test_range)
with torch.set_grad_enabled(False):
    model.eval()

    predY = xr.zeros_like(dm.da_HR[test_range,:,:])

    init_range = range(init_idx-look_back+1,
                       init_idx+1)
    init_range_m = range(init_idx-look_back,
                         init_idx)
    xk_HR = data_HR[init_range, :, :]
    xk_HR_m = data_HR[init_range_m, :, :]

    for i in range(pred_steps):
        xk_LR = data_LR[init_idx+i, :, :]
        xk_LR_p = data_LR[init_idx+i+1, :, :]
        xk_HR_p = data_HR[init_idx+i+1, :, :]

        res_secant = fun_secant(xk_HR, xk_HR_m, xk_LR_p)
        res_old  = xk_HR - xk_LR
        res_true = xk_HR_p - xk_LR_p        
        ub = torch.tensor(np.expand_dims(res_old.squeeze().astype('float32'),
                                         axis=(0,1))).view(1,look_back,-1)
        yb = torch.tensor(np.expand_dims(res_true.squeeze().astype('float32'),
                                         axis=(0,1))).view(1,1,-1)
        ob = model(ub)
        loss = loss_fn(ob[:,-1,:], yb[:,-1,:], reduction='sum')

        norm_resold = np.linalg.norm(res_old)
        norm_restrue = np.linalg.norm(res_true)
        print(f'{i}: |res_true|: {norm_restrue}, '
              f'|res_old|: {norm_resold}, loss: {loss}')
        
        yk = ob.view(1,look_back,dm.Nlat,dm.Nlon)\
               .detach().numpy().squeeze()

        predY[i,:,:] = scaler.inverse_transform(yk[-1,:,:].reshape(1,dm.Nlat*dm.Nlon))\
                             .reshape(dm.Nlat,dm.Nlon)
        xk_HR_m = xk_HR
        xk_HR = np.concatenate((xk_HR[1:,:,:],
                                np.expand_dims(xk_LR_p + yk[-1,:,:],
                                               axis=0)))

# EXPORT
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

model_type='LSTM'
pmodel = xr.zeros_like(dm.da_HR[test_range,:,:])
pmodel[:,:,:] = predY

dsX_HR = dm.da_HR[test_range,:,:]
dsX_LR = dm.da_LR[test_range,:,:]

dmodel  = dsX_HR - pmodel - dsX_LR

ds = {}
ds['1'] = dsX_HR.rename('X_HR')
ds['2'] = dsX_LR.rename('X_LR')
ds['3'] = (dsX_LR + pmodel).rename(f'p{model_type} + X_LR')
ds['4'] = pmodel.rename(f'p{model_type}')
ds['5'] = dmodel.rename(f'd{model_type}')
ds['6'] = (dmodel-(dsX_HR-dsX_LR)).rename(f'd{model_type} - (X_HR - X_LR)')

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

fig, axs = plt.subplots(2, 2, figsize=(10, 12),
                        sharex=True, sharey=True)

tic = time.time()

make_movie=True
if make_movie:
    with Pool(8) as p:
        p.map(plot_frame, range(0,10,1))

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


def plot_hovm(mode='horizontal'):
    plt.close('all')
    fig, axs = plt.subplots(3, 2, figsize=(10, 12),
                            sharex=True, sharey=True)
    for p in range(0,6):
        plt.subplot(3,2,p+1)
        this_ds = ds[f'{p+1}']
        if mode == 'horizontal':
            hovm_data = this_ds.sel(lat=57.9, method='nearest')
        elif mode == 'vertical':
            hovm_data = this_ds.sel(lon=5.3, method='nearest')
            
        hovm_data.plot.pcolormesh(vmin=this_ds.vmin,
                                  vmax=this_ds.vmax,
                                  center=0, cmap='RdBu',
                                  extend='both',
                                  cbar_kwargs={'label':''})

        plt.gca().set_title(this_ds.name)
        plt.gca().set_ylabel('')
        plt.gca().set_xlabel('')


    frame_name = f'output/hovm_{mode}_{model_type}_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(frame_name)

make_hovm=True
if make_hovm:
    plot_hovm(mode='horizontal')
    plot_hovm(mode='vertical')

toc = time.time()
print(f'elapsed: {toc-tic:02f}')
# print('\a')

# plt.close('all')
# plt.plot(log_dict['training loss per batch'])
# plt.pause(1)
