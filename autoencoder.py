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


import data_manager as dm
reload(dm)

scaler = MinMaxScaler(feature_range=(0,1))
data_HR = scaler.fit_transform(dm.da_HR.values.reshape(dm.Nt, -1))\
                .reshape(dm.Nt, dm.Nlat, dm.Nlon)
data_LR = scaler.transform(dm.da_LR.values.reshape(dm.Nt, -1))\
                .reshape(dm.Nt, dm.Nlat, dm.Nlon)


