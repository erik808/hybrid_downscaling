import xarray as xr
import keras
import re
import pandas as pd
import glob
import os
import numpy as np
import sys
from datetime import datetime
import importlib
import xesmf as xe
from dask.diagnostics import ProgressBar
from sklearn.preprocessing import MinMaxScaler


class CustomScaler():

    def __init__(self, scaling_type='disabled'):
        self.scaling_type = scaling_type
        self.shift = None
        self.scale = None
        self.fitted = False

    def fit(self, data):
        if self.scaling_type == 'standardize_per_feature':
            self.shift = np.nanmean(data, axis=0)
            self.scale = np.nanmean(data, axis=0)

        elif self.scaling_type == 'standardize_over_all_features':
            self.shift = np.nanmean(data)
            self.scale = np.nanmean(data)

        elif self.scaling_type == 'minmax_per_feature':
            self.scale = 1.0 / (np.nanmax(data, axis=0) -
                                np.nanmin(data, axis=0))
            self.shift = np.nanmin(data, axis=0)

        elif self.scaling_type == 'minmax_over_all_features':
            self.scale = 1.0 / (np.nanmax(data) - np.nanmin(data))
            self.shift = np.nanmin(data)

        elif self.scaling_type == 'disabled':
            self.shift = 0.0
            self.scale = 1.0

        self.fitted = True

    def transform(self, data):
        if not self.fitted:
            raise Exception('scaler not fitted')
        return (data - self.shift) * self.scale

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if not self.fitted:
            raise Exception('scaler not fitted')
        return (data / self.scale) + self.shift


class Tee:
    """ Used to redirect and duplicate output """

    def __init__(self, file_name, mode="w"):
        print(f'Tee: redirecting output to {file_name}')
        self.file = open(file_name, mode)
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


# decorator to make sure functions clean on ending
def clean_on_end(func):
    def wrapper(*args, **kwargs):
        exp_name = func(*args, **kwargs)
        cleanup(exp_name)
    return wrapper


def cleanup(exp_name):
    print(f'cleanup {exp_name}:')

    today = datetime.now().strftime('%Y%m%d')

    files = []
    for ext in ['.dill', '.keras']:
        files.extend(
            glob.glob(os.path.join(f'experiments/{exp_name}', "**", f"*{ext}"),
                      recursive=True)
        )

    for dfile in files:
        if today in dfile:
            print(f'deleting {dfile}')
            os.remove(dfile)


def load_config(obj, config_name):
    # Load a config that lives in the <configs> dir: config_file =
    # <configs>/<config_name>.py. Overwrite class members and
    # create new ones according to what is present in
    # config_file. Exclude "__" members and functions.

    config_file = f'configs.{config_name}'
    print(f'Load config: {config_file}')

    try:
        config = importlib.import_module(config_file)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(config_file)

    importlib.reload(config)

    module_vars = vars(config)

    # load variables
    config_vars = {
        key: value for key, value in module_vars.items()
        if not key.startswith("__") and not callable(value)
    }

    for (key, value) in config_vars.items():
        setattr(obj, key, value)


def build_grid(lat_arr, lon_arr):

    Nlat = lat_arr.shape[0]
    Nlon = lon_arr.shape[0]

    lat_grid = np.tile(lat_arr, (Nlon, 1)).T
    lon_grid = np.tile(lon_arr, (Nlat, 1))

    grid = {}
    grid['N'] = Nlat
    grid['M'] = Nlon
    grid['lat'] = np.ascontiguousarray(lat_grid)
    grid['lon'] = np.ascontiguousarray(lon_grid)

    return grid


def create_coarse_grid(
        lats_HR,
        lons_HR,
        coarsening_factor,
):

    N_lat = len(lats_HR)
    N_lon = len(lons_HR)
    lat_start, lat_end = lats_HR[[0, -1]]
    lon_start, lon_end = lons_HR[[0, -1]]
    dlat = np.diff(lats_HR[[0, 1]])[0]
    dlon = np.diff(lons_HR[[0, 1]])[0]

    lats_LR = np.linspace(
        lat_start + dlat / 2,
        lat_end - dlat / 2,
        int(N_lat / coarsening_factor))

    lons_LR = np.linspace(
        lon_start + dlon / 2,
        lon_end - dlon / 2,
        int(N_lon / coarsening_factor))

    grid_LR = build_grid(lats_LR, lons_LR)

    return grid_LR


def regrid_to_transect(grid_orig,
                       lon_start,
                       lon_end,
                       lat_start,
                       lat_end,
                       resolution=1e2):

    resolution = int(resolution)
    lat_arr = np.linspace(lat_start, lat_end, resolution)
    lon_arr = np.linspace(lon_start, lon_end, resolution)

    grid_upscale = {}
    grid_upscale['N'] = resolution
    grid_upscale['M'] = resolution
    lat_mat = np.tile(lat_arr, (resolution, 1)).T
    lon_mat = np.tile(lon_arr, (resolution, 1))
    grid_upscale['lat'] = np.ascontiguousarray(lat_mat)
    grid_upscale['lon'] = np.ascontiguousarray(lon_mat)
    # grid_upscale['mask'] = np.identity(resolution)

    interp_to_transect = xe.Regridder(grid_orig, grid_upscale,
                                      method="bilinear",
                                      extrap_method="inverse_dist")
    return interp_to_transect


def apply_time_range(globstr, time_range):
    files = sorted(glob.glob(globstr))
    matches = [re.search(r'\/[0-9].*.nc', f).group()
               for f in files]
    matches = [pd.to_datetime(m[1:-3]) for m in matches]

    start = pd.to_datetime(time_range.start)
    end = pd.to_datetime(time_range.stop)

    keep_files = []
    for f, m in zip(files, matches):
        if m >= start and m <= end:
            keep_files.append(f)

    return keep_files


def check_time_overlap(ds, time_range):
    start = pd.to_datetime(time_range.start)
    end = pd.to_datetime(time_range.stop)
    time_min, time_max = ds.time[0].values, ds.time[-1].values
    overlap = (time_max >= start) and (time_min <= end)
    return overlap


def ds_to_netcdf(ds, path, prefix=""):
    """ splits a ds up in monthly files and saves to <path>"""

    keys, datasets = nested_groupby(ds)
    paths = [f"{path}/{prefix}{y:04d}-{m:02d}.nc" for y, m in keys]
    options = {"zlib": True, "complevel": 5}

    keys, datasets = nested_groupby(ds)
    paths = coarse_data_paths(ds, path, prefix, keys=keys)

    encoding = {var : options for var in list(ds.keys())}

    with ProgressBar():
        xr.save_mfdataset(datasets, paths, encoding=encoding)

    return paths


def coarse_data_paths(ds, path, prefix="", keys=None):
    if keys is None:
        keys, _ = nested_groupby(ds)
    paths = [f"{path}/{prefix}{y:04d}-{m:02d}.nc" for y, m in keys]
    return paths


def nested_groupby(ds):
    datasets = []
    keys = []
    # def export_coarse_data(ds):
    for year_key, ds_year in ds.groupby('time.year'):
        for month_key, ds_month in ds_year.groupby('time.month'):
            keys.append([year_key, month_key])
            datasets.append(ds_month)

    return keys, datasets


def create_scaler(ds, scaling_range):
    """we do a partial fit per chunk to avoid loading everything in memory

    """

    da = ds.to_array()
    scaler = MinMaxScaler(feature_range=scaling_range)

    # make sure only time is chunked
    N_vars = len(da.variable)
    da = da.chunk({'variable': N_vars})

    # correct ordering
    da = da.transpose('time',
                      'latitude',
                      'longitude',
                      'variable')

    # here we do not want nans
    da = da.fillna(0.0)

    print('Create chunks')
    chunks = da.data.to_delayed().ravel()

    pb_i = keras.utils.Progbar(len(chunks), interval=0.5)
    print('Creating scaler')
    for chunk in chunks:
        pb_i.add(1)
        chunk = chunk.compute()
        chunk = chunk.reshape(chunk.shape[0], -1)
        scaler.partial_fit(chunk)

    return scaler
