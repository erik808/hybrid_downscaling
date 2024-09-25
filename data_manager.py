import os
import numpy as np
import xarray as xr
import xesmf as xe
import torch
import keras
import dill
import time
import pytide
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from multiprocess import Pool

data_dir      = 'data'
transect_dir  = f'{data_dir}/transects'
HR_data_files = (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
                 f'uo-vo_4.23E-7.78E_56.81N-58.70N_2023-/*.nc')

HR_bathy_file = (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
                 f'static_multi-vars_4.23E-7.78E_56.81N-58.70N_0.49-643.57m.nc')

coords_file = (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
               f'static_e1t-e2t-e3t_4.23E-7.78E_56.81N-58.70N_0.49-643.57m.nc')

LR_data_file = (f'{data_dir}/cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i_'
                f'uo-vo_4.22E-7.78E_56.80N-58.67N_2023-01-01-2023-05-01.nc')

def build_grid(ds=None, mask=None):
    if mask == None:
        lat_arr = ds.latitude
        lon_arr = ds.longitude
    elif ds == None:
        lat_arr = mask.latitude
        lon_arr = mask.longitude
    else:
        raise Exception('Either ds or mask should be given')

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

def crop(input_field):
    return input_field[...,3:-2,:-1]

def regrid_to_transect(tpicker, resolution=1e2):

    print('Create transect regridder')
    mask = crop(xr.open_dataset(HR_bathy_file).mask)
    grid_orig = build_grid(mask)

    lons = grid_orig['lon'][0,:]
    lats = grid_orig['lat'][:,0]
    dlon = float((lons[1:]-lons[:-1]).mean())
    dlat = float((lats[1:]-lats[:-1]).mean())
    grid_aspect = dlon / dlat
    print(f' grid aspect ratio: {grid_aspect}')

    lon_start = lons[tpicker.x_trans[0]]
    lon_end   = lons[tpicker.x_trans[-1]]
    lat_start = lats[tpicker.y_trans[0]]
    lat_end   = lats[tpicker.y_trans[-1]]

    resolution = int(resolution)
    lat_arr = np.linspace(lat_start, lat_end, resolution)
    lon_arr = np.linspace(lon_start, lon_end, resolution)

    grid_upscale = {}
    grid_upscale['N'] = resolution
    grid_upscale['M'] = resolution
    lat_mat = np.tile(lat_arr, (resolution,1)).T
    lon_mat = np.tile(lon_arr, (resolution,1))
    grid_upscale['lat'] = np.ascontiguousarray(lat_mat)
    grid_upscale['lon'] = np.ascontiguousarray(lon_mat)
    grid_upscale['mask'] = np.identity(resolution)

    interp_to_transect = xe.Regridder(grid_orig, grid_upscale,
                                      method="bilinear",
                                      extrap_method="inverse_dist")
    return interp_to_transect

def load_u_data():
    bt_HR = xr.open_dataset(HR_bathy_file)
    ds_HR = xr.open_mfdataset(HR_data_files, parallel=True)
    ds_LR = xr.open_dataset(LR_data_file)

    mask = bt_HR.mask[0,:,:]
    grid_HR = build_grid(ds_HR, mask)
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

    return da_HR, da_LR, mask

def get_grid():
    " load grid, crop and return "
    coords = xr.open_dataset(coords_file)
    mask = crop(xr.open_dataset(HR_bathy_file).mask[0,:,:])
    l = [crop(coords[var]) for var in coords]
    coords = xr.merge(l)
    return coords, mask

def load_uv_data(coarsen_in_time=False,
                 detide=False,
                 differences=False,
                 coarsening_method='gaussian_filter'):

    bt_HR = xr.open_dataset(HR_bathy_file)
    ds_HR = xr.open_mfdataset(HR_data_files, parallel=True)
    ds_LR = xr.open_dataset(LR_data_file)

    mask = bt_HR.mask[0,:,:]
    grid_HR = build_grid(ds_HR, mask)
    grid_LR = build_grid(ds_LR)


    interp_HR_LR = xe.Regridder(grid_HR, grid_LR, "bilinear",
                                extrap_method="inverse_dist")
    interp_LR_HR = xe.Regridder(grid_LR, grid_HR, "bilinear",
                                extrap_method="inverse_dist")

    da_HR_uo = ds_HR.uo.rename({'longitude':'lon',
                                'latitude':'lat'})\
                                .fillna(0.0)

    da_HR_vo = ds_HR.vo.rename({'longitude':'lon',
                                'latitude':'lat'})\
                                .fillna(0.0)

    def detide_da(da):
        da.load()
        wt = pytide.WaveTable(["M2", "S2", "N2", "K1",
                               "O1", "Q1", "M4",
                               "K2", "P1", "Mf", "Mm" ])

        dates = da.time.values
        f, vu = wt.compute_nodal_modulations(dates)
        latlons = np.where(mask==1)
        ind_range = range(len(latlons[0]))
        pb = keras.utils.Progbar(len(ind_range))

        def detide_point(i):
            if not i % 200:
                pb.update(i)

            vals = da[:, latlons[0][i], latlons[1][i]].values
            waves = wt.harmonic_analysis(vals, f, vu)
            vals_tide = wt.tide_from_tide_series(dates, waves)
            vals_detide = vals - vals_tide
            return vals_detide

        print(f'Detiding:')
        with Pool(4) as p:
            results = p.map(detide_point, ind_range)

        pb.update(ind_range.stop, finalize=True)

        pb = keras.utils.Progbar(len(ind_range))
        da_dt = xr.zeros_like(da)
        print('Filling data array:')
        for i in ind_range:
            da_dt[:, latlons[0][i], latlons[1][i]] = results[i]
            pb.add(1)

        return da_dt

    if detide:
        da_HR_uo = detide_da(da_HR_uo)
        da_HR_vo = detide_da(da_HR_vo)

    if differences:
        print('Replace data with forward differences')
        da_HR_uo = da_HR_uo.diff('time')
        da_HR_vo = da_HR_vo.diff('time')

    def create_da_LR(da_HR,
                     coarsen_in_time=False,
                     coarse_time_freq='2h'):

        print('Regridding HR to LR')
        da_HR_LR = interp_HR_LR(da_HR.values)
        da_HR_LR = xr.DataArray(da_HR_LR, dims=['time','lat','lon'],
                                coords={'time':da_HR.time,
                                        'lat':ds_LR.latitude.values,
                                        'lon':ds_LR.longitude.values})
        if coarsen_in_time:
            da_HR_LR_resamp = da_HR_LR.resample(time=coarse_time_freq)\
                                      .first()
            da_HR_LR = da_HR_LR_resamp.interp(time=da_HR_LR.time,
                                              method='cubic')

        print('Regridding LR to HR')
        da_HR_LR_HR = xr.zeros_like(da_HR)
        da_HR_LR_HR_tmp = interp_LR_HR(da_HR_LR.values)
        da_HR_LR_HR[:,:,:] = da_HR_LR_HR_tmp

        # remove nans
        da_LR = da_HR_LR_HR.fillna(0.0)

        return da_LR


    def filter_HR_data(da, sigma):
        tic = time.time()
        print(f'Loading dataset {da.name}... ', end='')
        da.load()
        toc = time.time()
        print(f'done ({toc-tic:.1f}s)')
        print(f'Apply Gaussian filter with '
              f'sigma={sigma} to {da.name}...', end='')
        out_da = xr.zeros_like(da)
        # assume 3D
        out_da[:,:,:] = gaussian_filter(da.values, sigma=sigma)
        toc = time.time()

        mask_ = mask.rename({'latitude':'lat',
                            'longitude':'lon'})\
                    .assign_coords({'lat':out_da.lat,
                                    'lon':out_da.lon})

        out_da = out_da.where(mask_ == 1)
        out_da = out_da.fillna(0.0)
        print(f' done ({toc-tic:.1f}s)')
        return out_da

    if coarsening_method == 'regridding':
        da_LR_uo = create_da_LR(da_HR_uo, coarsen_in_time)
        da_LR_vo = create_da_LR(da_HR_vo, coarsen_in_time)

    elif coarsening_method == 'gaussian_filter':
        sigma = [1,1,1]
        da_LR_uo = filter_HR_data(da_HR_uo, sigma)
        da_LR_vo = filter_HR_data(da_HR_vo, sigma)
    else:
        raise Exception('invalid coarsening_method {coarsening_method}')

    # Crop data
    da_HR = {'uo': crop(da_HR_uo),
             'vo': crop(da_HR_vo)}
    da_LR = {'uo': crop(da_LR_uo),
             'vo': crop(da_LR_vo)}

    mask = crop(mask)

    return da_HR, da_LR, mask


class CustomScaler():

    def __init__(self, scaling_type = 'minmax_per_feature'):
        self.scaling_type = scaling_type
        self.shift = None
        self.scale = None
        self.fitted = False

    def fit(self, data):
        if self.scaling_type == 'standardize_per_feature':
            self.shift = np.mean(data, axis=0)
            self.scale = np.mean(data, axis=0)

        elif self.scaling_type == 'standardize_over_all_features':
            self.shift = np.mean(data)
            self.scale = np.mean(data)

        elif self.scaling_type == 'minmax_per_feature':
            self.scale = 1.0 / (np.max(data, axis=0) - np.min(data, axis=0))
            self.shift = np.min(data, axis=0)

        elif self.scaling_type == 'minmax_over_all_features':
            self.scale = 1.0 / (np.max(data) - np.min(data))
            self.shift = np.min(data)

        self.fitted = True

    def transform(self, data):
        if not self.fitted: raise Exception('scaler not fitted')
        return (data - self.shift) * self.scale

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if not fitted: raise Exception('scaler not fitted')
        return (x / self.scale) + self.shift


def load_training_data(split_factor=4/5,
                       scaling_range=(0,1),
                       coarsen_in_time=False,
                       detide=False,
                       differences=False,
                       residual_mode=False):
    # assume everything has this shape
    params = {}
    data = {}

    da_HR, da_LR, da_mask = load_uv_data(coarsen_in_time=coarsen_in_time,
                                         detide=detide,
                                         differences=differences)

    # create a torch mask
    params['mask'] = torch.tensor(da_mask.values)[None,:,:,None]

    # do the assembling into channels here
    data_HR = np.stack([da_HR['uo'].values,
                        da_HR['vo'].values], axis=3)
    data_LR = np.stack([da_LR['uo'].values,
                        da_LR['vo'].values], axis=3)

    # scaler = CustomScaler(scaling_type='minmax_per_feature')
    scalers = {}
    scalers['HR'] = MinMaxScaler(feature_range=scaling_range)
    scalers['R'] = MinMaxScaler(feature_range=scaling_range)

    Nt, Nlat, Nlon, num_channels = data_HR.shape


    data_HR = scalers['HR'].fit_transform(data_HR.reshape(Nt, -1))\
                           .reshape(Nt, Nlat, Nlon, num_channels)

    data_LR = scalers['HR'].transform(data_LR.reshape(Nt, -1))\
                           .reshape(Nt, Nlat, Nlon, num_channels)

    # import matplotlib.pyplot as plt
    # plt.close('all')
    # data_R  = (data_HR - data_LR)

    # plt.figure(figsize=(8,10))

    # plt.subplot(4,2,1)
    # a = plt.imshow(data_R[100,:,:,0])
    # plt.colorbar(a, shrink=0.5)

    # plt.subplot(4,2,2)
    # a = plt.imshow(data_R.min(axis=0)[:,:,0])
    # plt.colorbar(a, shrink=0.5)

    # plt.subplot(4,2,3)
    # a = plt.imshow(data_HR[100,:,:,0])
    # plt.colorbar(a, shrink=0.5)

    # plt.subplot(4,2,4)
    # a = plt.imshow(data_LR[100,:,:,0])
    # plt.colorbar(a, shrink=0.5)

    # plt.subplot(4,2,7)
    # plt.plot(np.sum(np.square(data_HR[:200,:,:,0]), axis=(1,2)))
    # plt.plot(np.sum(np.square(data_LR[:200,:,:,0]), axis=(1,2)))

    if residual_mode:
        # create residual and secant predictor data
        data_R  = (data_HR - data_LR)[2:,]
        secant  = 2*data_HR[1:-1,] - data_HR[:-2,]
        data_FT = secant - data_LR[2:,]
        Nt_R = data_R.shape[0]

        data_R = scalers['R'].fit_transform(data_R.reshape(Nt_R, -1))\
                             .reshape(Nt_R, Nlat, Nlon, num_channels)
        data_FT = scalers['R'].transform(data_FT.reshape(Nt_R, -1))\
                              .reshape(Nt_R, Nlat, Nlon, num_channels)
        # plt.subplot(4,2,5)
        # a = plt.imshow(data_R[100,:,:,0])
        # plt.colorbar(a, shrink=0.5)

        # plt.subplot(4,2,6)
        # a = plt.imshow(scalers['R'].min_\
        #                .reshape(Nlat, Nlon, num_channels)[:,:,0])
        # plt.colorbar(a, shrink=0.5)

        # plt.subplot(4,2,8)
        # a = plt.imshow(scalers['R'].scale_\
        #                .reshape(Nlat, Nlon, num_channels)[:,:,0])
        # plt.colorbar(a, shrink=0.5)

        # plt.tight_layout()
        # plt.pause(1)
        # breakpoint()


    Nt = Nt_R if residual_mode else Nt

    params.update({'Nt'   : Nt,
                   'Nlat' : Nlat,
                   'Nlon' : Nlon,
                   'num_channels' : num_channels})

    split = int(Nt*split_factor)
    train_range = range(0, split)
    spinup_range = range(split-10,split)
    test_range = range(split, Nt)

    if residual_mode:
        data['train'] = {'R'    : data_R[train_range,],
                         'LR'   : data_LR[2:,][train_range,],
                         'HR'   : data_HR[2:,][spinup_range,],
                         'time' : da_LR['uo'].time.values[2:][train_range]}

        data['test']  = {'R'    : data_R[test_range,],
#                         'FT'   : data_FT[test_range,],
                         'LR'   : data_LR[2:,][test_range,],
                         'HR'   : data_HR[2:,][test_range,],
                         'time' : da_LR['uo'].time.values[2:][test_range]}
    else:
        data['train'] = {'HR'   : data_HR[train_range,],
                         'LR'   : data_LR[train_range,],
                         'time' : da_LR['uo'].time.values[train_range]}

        data['test']  = {'HR'   : data_HR[test_range,],
                         'LR'   : data_LR[test_range,],
                         'time' : da_LR['uo'].time.values[test_range]}

    return data, params, scalers

def create_training_data(compute_data=True,
                         encoder=None,
                         residual_mode=False,
                         coarsen_in_time=False,
                         detide=False,
                         differences=False):

    postfix = '_detided' if detide else ''
    postfix += '_diff' if differences else ''
    postfix += '_residuals' if residual_mode else ''
    dill_file     = f'{data_dir}/ae_esn_training_data{postfix}.dill'
    dill_file_enc = f'{data_dir}/ae_esn_training_data{postfix}_encoded.dill'

    enc_data={}
    if compute_data:
        print('Create training data')
        orig_data, params, scalers  = \
            load_training_data(split_factor=4/5,
                               coarsen_in_time=coarsen_in_time,
                               detide=detide,
                               differences=differences,
                               residual_mode=residual_mode)

        container = {'data' : orig_data,
                     'params' : params,
                     'scalers' : scalers}

        print(f'writing to {dill_file}')
        with open(dill_file, 'wb') as file:
            dill.dump(container, file)

        if encoder != None:
            print('Create encoded train and test data...')
            enc_data = {}
            for period in ['train', 'test']:
                enc_data[period] = {}
                for resolution in ['HR', 'LR']:
                    print(f'{period}-{resolution}')
                    enc_data[period][resolution] = \
                        encoder.predict(orig_data[period][resolution])

            container_enc = {'data' : enc_data}

            print(f'writing to {dill_file_enc}')
            with open(dill_file_enc, 'wb') as file:
                dill.dump(container_enc, file)

    else:
        print('Load training data from file')
        print(f'  loading from {dill_file}')
        with open(dill_file, 'rb') as file:
            data = dill.load(file)

        orig_data = data['data']
        params = data['params']
        scalers = data['scalers']

        if encoder != None:
            print(f'  loading from {dill_file_enc}')
            with open(dill_file_enc, 'rb') as file:
                data_enc = dill.load(file)

            enc_data = data_enc['data']
        else:
            enc_data = None

    return orig_data, params, scalers, enc_data


def setup_directories(experiment_id, add_id):
    models_dir = f'experiments/{experiment_id}{add_id}/models'
    tuning_dir = f'experiments/{experiment_id}{add_id}/tuning'
    results_dir = f'experiments/{experiment_id}{add_id}/results'
    movie_dir = f'experiments/{experiment_id}{add_id}/movies'
    checkpoints_dir = f'experiments/{experiment_id}{add_id}/checkpoints'
    log_file = f'{models_dir}/log.txt'

    os.system(f'mkdir -p {models_dir}')
    os.system(f'mkdir -p {tuning_dir}')
    os.system(f'mkdir -p {movie_dir}')
    os.system(f'mkdir -p {results_dir}')
    os.system(f'mkdir -p {checkpoints_dir}')

    dirs = {'models'      : models_dir,
            'tuning'      : tuning_dir,
            'results'     : results_dir,
            'movies'      : movie_dir,
            'checkpoints' : checkpoints_dir}

    files = {'log' : log_file}

    return dirs, files
