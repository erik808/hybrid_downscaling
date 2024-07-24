import os
import numpy as np
import xarray as xr
import xesmf as xe
import torch
import dill
from sklearn.preprocessing import MinMaxScaler

data_dir      = 'data'
dill_file     = f'{data_dir}/ae_esn_training_data.dill'
dill_file_enc = f'{data_dir}/ae_esn_training_data_encoded.dill'


HR_data_files = (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
                 f'uo-vo_4.23E-7.78E_56.81N-58.70N_2023-/*.nc')

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

def load_uv_data():
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

    def create_da_LR(da_HR):
        print('Regridding HR to LR')
        da_HR_LR = interp_HR_LR(da_HR.values)
        print('Regridding LR to HR')
        da_HR_LR_HR_tmp = interp_LR_HR(da_HR_LR)

        da_HR_LR = xr.DataArray(da_HR_LR, dims=['time','lat','lon'],
                                coords={'time':ds_HR.time,
                                        'lat':ds_LR.latitude.values,
                                        'lon':ds_LR.longitude.values})

        da_HR_LR_HR = xr.zeros_like(da_HR)
        da_HR_LR_HR[:,:,:] = da_HR_LR_HR_tmp
        da_LR = da_HR_LR_HR.fillna(0.0)
        return da_LR

    da_LR_uo = create_da_LR(da_HR_uo)
    da_LR_vo = create_da_LR(da_HR_vo)

    da_HR = {'uo': da_HR_uo,
             'vo': da_HR_vo}
    da_LR = {'uo': da_LR_uo,
             'vo': da_LR_vo}

    return da_HR, da_LR, mask


def load_training_data(split_factor=4/5,
                       scaling_range=(0,1)):
    # assume everything has this shape
    params = {}
    data = {}

    da_HR, da_LR, da_mask = load_uv_data()

    # create a torch mask
    params['mask'] = torch.tensor(da_mask.values)[None,:,:,None]

    # do the assembling into channels here
    data_HR_stacked = np.stack([da_HR['uo'].values,
                                da_HR['vo'].values], axis=3)
    data_LR_stacked = np.stack([da_LR['uo'].values,
                                da_LR['vo'].values], axis=3)

    Nt, Nlat, Nlon, num_channels = data_HR_stacked.shape
    params.update({'Nt':Nt,
                   'Nlat':Nlat,
                   'Nlon':Nlon,
                   'num_channels':num_channels})

    # StandardScaler doesnt work that well
    scaler_HR = MinMaxScaler(feature_range=scaling_range)
    scalers = {}
    scalers['HR'] = scaler_HR

    data_HR = scaler_HR.fit_transform(data_HR_stacked.reshape(Nt, -1))\
                       .reshape(Nt, Nlat, Nlon, num_channels)
    data_LR = scaler_HR.transform(data_LR_stacked.reshape(Nt, -1))\
                       .reshape(Nt, Nlat, Nlon, num_channels)

    split = int(Nt*split_factor)
    train_range = range(0, split)
    test_range = range(split, Nt)

    data['train'] = {'HR'   : data_HR[train_range,:,:,:],
                     'LR'   : data_LR[train_range,:,:,:],
                     'time' : da_LR['uo'].time.values[train_range]}

    data['test']  = {'HR'   : data_HR[test_range,:,:,:],
                     'LR'   : data_LR[test_range,:,:,:],
                     'time' : da_LR['uo'].time.values[test_range]}

    return data, params, scalers

def create_training_data(compute_data=True, encoder=None):
    if compute_data:
        print('Create training data')
        orig_data, params, scalers  = load_training_data()

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
