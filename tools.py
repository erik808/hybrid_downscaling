import glob
import os
import numpy as np
import sys
from datetime import datetime
import importlib
import xesmf as xe


class CustomScaler():

    def __init__(self, scaling_type='minmax_per_feature'):
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
