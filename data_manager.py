import os
import numpy as np
import xarray as xr
import xesmf as xe
import torch
import keras
import dill
import time
import pytide
import scipy
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from multiprocess import Pool


class DataManager():

    def __init__(self):

        self.data_dir      = 'data'
        self.transect_dir  = f'{self.data_dir}/transects'
        self.HR_data_files = \
            (f'{self.data_dir}/cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
             f'uo-vo_4.23E-7.78E_56.81N-58.70N_2023-/*.nc')

        self.HR_bathy_file = \
            (f'{self.data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
             f'static_multi-vars_4.23E-7.78E_56.81N-58.70N_0.49-643.57m.nc')

        self.coords_file = \
            (f'{self.data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
             f'static_e1t-e2t-e3t_4.23E-7.78E_56.81N-58.70N_0.49-643.57m.nc')

        self.LR_data_file = \
            (f'{self.data_dir}/cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i_'
             f'uo-vo_4.22E-7.78E_56.80N-58.67N_2023-01-01-2023-12-31.nc')


    def build_grid(self, ds=[], mask=[]):
        assert (len(ds) > 0 or
                len(mask) > 0), 'Either ds or mask should be given'

        if len(mask) == 0:
            lat_arr = ds.latitude
            lon_arr = ds.longitude
        else:
            lat_arr = mask.latitude
            lon_arr = mask.longitude

        Nlat = lat_arr.shape[0]
        Nlon = lon_arr.shape[0]
        lat_grid = np.tile(lat_arr, (Nlon,1)).T
        lon_grid = np.tile(lon_arr, (Nlat,1))
        grid = {}
        grid['N'] = Nlat
        grid['M'] = Nlon
        grid['lat'] = np.ascontiguousarray(lat_grid)
        grid['lon'] = np.ascontiguousarray(lon_grid)
        if len(mask) > 0:
            grid['mask'] = mask
        return grid

    def crop(self, input_field):
        return input_field[...,3:-2,:-1]

    def regrid_to_transect(self, tpicker, resolution=1e2):

        print('Create transect regridder')
        mask = self.crop(xr.open_dataset(self.HR_bathy_file).mask)
        grid_orig = self.build_grid(mask)

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


    def create_regridders(self):
        print('Create regridders')
        bt_HR = xr.open_dataset(self.HR_bathy_file)
        ds_HR = xr.open_mfdataset(self.HR_data_files, parallel=True)
        ds_LR = xr.open_dataset(self.LR_data_file)

        mask = bt_HR.mask[0,:,:]
        grid_HR = self.build_grid(ds_HR, mask)
        grid_LR = self.build_grid(ds_LR)

        interp_HR_LR = xe.Regridder(grid_HR, grid_LR, "bilinear",
                                    extrap_method="inverse_dist")
        interp_LR_HR = xe.Regridder(grid_LR, grid_HR, "bilinear",
                                    extrap_method="inverse_dist")

        return interp_HR_LR, interp_LR_HR, mask


    def get_coarse_data(self, time_range, interpolate=False):

        if not interpolate:
            time_slice=slice(time_range[0],time_range[-1])
            ds_LR = xr.open_dataset(self.LR_data_file)\
                      .sel(time=time_slice)
        else:
                      #.fillna(0.0)\
            ds_LR = xr.open_dataset(self.LR_data_file)\
                      .interp(time=time_range, method='linear')

        r0, regridder, mask = self.create_regridders()
        da = np.stack([ds_LR['uo'].values,
                       ds_LR['vo'].values], axis=1)
        do=np.nan_to_num(self.crop(regridder(da)))

        return do.transpose(0,2,3,1)

    def get_grid(self):
        " load grid, crop and return "
        coords = xr.open_dataset(self.coords_file)
        mask = self.crop(xr.open_dataset(self.HR_bathy_file).mask[0,:,:])
        l = [self.crop(coords[var]) for var in coords]
        coords = xr.merge(l)
        return coords, mask


    def load_uv_data(self,
                     coarsen_in_time=False,
                     detide=False,
                     differences=False,
                     coarsening_method='gaussian_filter',
                     sigma=[1,1,1],
                     truncation=20):

        bt_HR = xr.open_dataset(self.HR_bathy_file)
        ds_HR = xr.open_mfdataset(self.HR_data_files, parallel=True)
        ds_LR = xr.open_dataset(self.LR_data_file)

        mask = bt_HR.mask[0,:,:]
        grid_HR = self.build_grid(ds_HR, mask)
        grid_LR = self.build_grid(ds_LR)

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
            tic = time.time()
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

        def reduced_basis(ds, truncation):
            print('Computing POD basis')
            orig_shape = list(ds.shape)
            data = ds.reshape(orig_shape[0], -1)
            data = data - np.mean(data, axis=0)
            U,_,_ = scipy.linalg.svd(data.T, False)
            new_shape = [truncation] + orig_shape[1:]
            U = U[:,:truncation].T.reshape(new_shape)
            return U

        def regrid_basis(U):

            interp_HR_LR, interp_LR_HR, mask = self.create_regridders()
            U_HR = interp_LR_HR(np.ascontiguousarray(U.transpose(0,3,1,2)))\
                .transpose(0,2,3,1)
            U_HR = np.nan_to_num(U_HR)

            ## renormalize
            orig_shape = U_HR.shape
            U_HR_MAT = U_HR.reshape(orig_shape[0], -1)
            norms = np.linalg.norm(U_HR_MAT, axis=1)
            U_HR = (U_HR.T / norms).T

            return U_HR

        def orth_project(U, da):
            tic = time.time()
            print(f'Loading dataset {da.name}... ', end='')
            da.load()
            toc = time.time()
            print(f'done ({toc-tic:.1f}s)')
            print(f'Creating approx. orthogonal projection... ', end='')
            tic = time.time()
            truncation = U.shape[0]
            orig_shape = da.shape
            out = xr.zeros_like(da)
            data = da.data.reshape(orig_shape[0], -1)
            U = U.reshape(truncation,-1)
            coords = (U @ data.T)
            da_LR = (U.T @ coords).T
            out[:,:,:] = da_LR.reshape(orig_shape)
            toc = time.time()
            print(f'done ({toc-tic:.1f}s)')
            return out, coords

        if coarsening_method == 'regridding':
            da_LR_uo = create_da_LR(da_HR_uo, coarsen_in_time)
            da_LR_vo = create_da_LR(da_HR_vo, coarsen_in_time)

        elif coarsening_method == 'gaussian_filter':
            da_LR_uo = filter_HR_data(da_HR_uo, sigma)
            da_LR_vo = filter_HR_data(da_HR_vo, sigma)

        elif coarsening_method == 'reduced_basis':

            data_LR = np.stack([ds_LR['uo'].fillna(0.0).values,
                                ds_LR['vo'].fillna(0.0).values], axis=3)

            U = reduced_basis(data_LR, truncation)
            U = regrid_basis(U)

            # da_LR_uo = filter_HR_data(da_HR_uo, sigma)
            # da_LR_vo = filter_HR_data(da_HR_vo, sigma)

            da_LR_uo, c_uo = orth_project(U[...,0], da_HR_uo)
            da_LR_vo, c_vo = orth_project(U[...,1], da_HR_vo)

            # plt.close('all')
            # plt.figure()
            # plt.imshow(da_HR_uo[0,:,:])
            # plt.pause(1)
            # plt.figure()
            # plt.imshow(da_LR_uo[0,:,:])
            # plt.pause(1)

            # plt.figure()
            # plt.loglog(np.mean(np.abs(c_uo),axis=1),'.-')
            # plt.loglog(np.mean(np.abs(c_vo),axis=1),'.-')
            # plt.pause(1)

        else:
            raise Exception('invalid coarsening_method {coarsening_method}')

        # Crop data
        da_HR = {'uo': self.crop(da_HR_uo),
                 'vo': self.crop(da_HR_vo)}
        da_LR = {'uo': self.crop(da_LR_uo),
                 'vo': self.crop(da_LR_vo)}

        mask = self.crop(mask)

        return da_HR, da_LR, mask



    def load_training_data(self,
                           split_factor=4/5,
                           scaling_range=(0,1),
                           coarsen_in_time=False,
                           coarsening_method='gaussian_filter',
                           detide=False,
                           differences=False,
                           sigma=[1,1,1],
                           truncation=20):

        # assume everything has this shape
        params = {}
        data = {}

        da_HR, da_LR, da_mask = \
            self.load_uv_data(coarsen_in_time=coarsen_in_time,
                              detide=detide,
                              differences=differences,
                              sigma=sigma,
                              coarsening_method=coarsening_method,
                              truncation=truncation)

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
        scalers['LR'] = MinMaxScaler(feature_range=scaling_range)
        scalers['R']  = MinMaxScaler(feature_range=scaling_range)

        Nt, Nlat, Nlon, num_channels = data_HR.shape
        data_HR_orig = data_HR
        data_LR_orig = data_LR

        data_HR = scalers['HR'].fit_transform(data_HR.reshape(Nt, -1))\
                               .reshape(Nt, Nlat, Nlon, num_channels)

        use_same_scaler = True
        if use_same_scaler:
            scalers['LR'] = scalers['HR']
            data_LR = scalers['LR'].transform(data_LR.reshape(Nt, -1))\
                                   .reshape(Nt, Nlat, Nlon, num_channels)
        else:
            data_LR = scalers['LR'].fit_transform(data_LR.reshape(Nt, -1))\
                                   .reshape(Nt, Nlat, Nlon, num_channels)

        scale = scalers['HR'].scale_.reshape(Nlat,Nlon,num_channels)

        params.update({'Nt'   : Nt,
                       'Nlat' : Nlat,
                       'Nlon' : Nlon,
                       'num_channels' : num_channels})

        split = int(Nt*split_factor)
        self.train_range = range(0, split)
        self.test_range = range(split, Nt)

        data['train'] = {'HR'   : data_HR[self.train_range,],
                         'LR'   : data_LR[self.train_range,],
                         'time' : da_LR['uo'].time.values[self.train_range]}

        data['test']  = {'HR'   : data_HR[self.test_range,],
                         'LR'   : data_LR[self.test_range,],
                         'time' : da_LR['uo'].time.values[self.test_range]}

        return data, params, scalers


    def create_training_data(self,
                             compute_data=True,
                             encoder=None,
                             coarsen_in_time=False,
                             coarsening_method='gaussian_filter',
                             detide=False,
                             differences=False,
                             sigma=[1,1,1],
                             truncation=20):

        postfix =  ''
        postfix =  '_detided' if detide else ''
        postfix += '_diff' if differences else ''

        if (coarsening_method == 'gaussian_filter' and
            len(sigma) > 0):
            sigma_str = str(sigma).replace(', ', '-').replace('.','_')
            postfix += f'_blur_{sigma_str}'
        elif (coarsening_method == 'reduced_basis'):
            postfix += f'_reduced_basis_tr{truncation}'
        else:
            postfix += ''

        dill_file     = f'{self.data_dir}/ae_esn_training_data{postfix}.dill'
        dill_file_enc = f'{self.data_dir}/ae_esn_training_data{postfix}_encoded.dill'

        enc_data = {}
        if compute_data:
            print('Create training data')
            orig_data, params, scalers  = \
                self.load_training_data(split_factor=4/5,
                                        coarsen_in_time=coarsen_in_time,
                                        coarsening_method=coarsening_method,
                                        detide=detide,
                                        differences=differences,
                                        sigma=sigma,
                                        truncation=truncation)

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


    def create_lookback(self, data, lookback):

        print('Create input data with lookback')
        # append train and test data along samples axis
        orig_train_length = data['train']['time'].shape[0]
        orig_test_length = data['test']['time'].shape[0]

        train_range = range(0, orig_train_length - lookback)
        test_range = range(train_range.stop,
                           train_range.stop + orig_test_length)

        for key in ['HR', 'LR', 'time']:
            X = np.append(data['train'][key],
                          data['test'][key],
                          axis=0)

            n_samples = X.shape[0]
            fields = list()
            for lb in range(lookback+1):
                # lookback field
                lb_field = X[lookback-lb:n_samples-lb,]
                fields.append(lb_field)

            tic = time.time()
            print(f'Stacking lookback fields: {key}')
            data = np.stack(fields, axis=1)
            data['train'][key] = data[train_range,]
            data['test'][key] = data[test_range,]
            toc = time.time()
            print(f'done ({toc-tic:.1f}s)')

        return orig_data


    def setup_directories(self, experiment_id, add_id):
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



class DataGenerator(keras.utils.PyDataset):

    def __init__(self, x, y,
                 ft_type,
                 batch_size,
                 shuffle=False,
                 lookback=0,
                 **kwargs):

        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.ft_type = ft_type
        self.shuffle = shuffle
        self.lookback = lookback
        self.__setup_data(x, y)
        

    def __setup_data(self, x, y):
        assert len(x) == 2
        assert len(y) == 1
        self.indices = np.arange(self.lookback, x[0].shape[0])
        self.n = len(self.indices)
        self.__do_shuffle()

        if self.ft_type == 'hybrid':
            self.x = x
        elif self.ft_type == 'only':
            self.x = [x[1]]
        else:
            self.x = [x[0]]
        self.y = y
        

    def __len__(self):
        # number of batches
        return int(np.ceil(self.n / self.batch_size))
    

    def __getitem__(self, index):
        low  = index * self.batch_size
        high = np.min([low + self.batch_size, self.n])
        inds = self.indices[low:high]

        batch_x = self.__create_lookback(inds, self.x)
        batch_y = [y[inds,] for y in self.y]
        return (batch_x, batch_y)
    

    def __do_shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


    def __create_lookback(self, inds, data):

        batch = list()
        for var in data:
            lb_fields = list() # lookback fields
            for lb in range(self.lookback+1):
                lb_field = var[inds-lb,]
                lb_fields.append(lb_field)

            batch.append(np.stack(lb_fields, axis=1))

        return batch


    def on_epoch_end(self):
        self.__do_shuffle()


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
