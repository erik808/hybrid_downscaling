import copernicusmarine as cm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 3d box for open boundary in the Channel
box_Channel = {}
box_Channel['min_lon'] = -4.3
box_Channel['max_lon'] = -3.9
box_Channel['min_lat'] = 48.64
box_Channel['max_lat'] = 50.36
box_Channel['min_dep'] = 0.4940253794193268
box_Channel['max_dep'] = 155.85072326660156


# box for the Norwegian coastal current test area
box_NwCC = {}
box_NwCC['min_lon'] = 4.2
box_NwCC['max_lon'] = 7.8
box_NwCC['min_lat'] = 56.8
box_NwCC['max_lat'] = 58.7
box_NwCC['min_dep'] = 0.4940253794193268
box_NwCC['max_dep'] = 643.5668334960938

dataset_id = "cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i"
dataset_id = "cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i"
variables = ["uo", "vo"]
time_start = "2023-01-01T23:00:00"
time_end = "2023-02-01T23:00:00"

out = cm.subset(
    dataset_id=dataset_id,
    variables=variables,
    minimum_longitude=box_NwCC['min_lon'],
    maximum_longitude=box_NwCC['max_lon'],
    minimum_latitude=box_NwCC['min_lat'],
    maximum_latitude=box_NwCC['max_lat'],
    start_datetime=time_start,
    end_datetime=time_end,
    minimum_depth=box_NwCC['min_dep'],
    maximum_depth=box_NwCC['max_dep'],
    force_download=False,
    netcdf_compression_enabled=True,
    # output_filename="data.nc",
    overwrite_output_data=True,
    overwrite_metadata_cache=False,
    netcdf_compression_level=0,
)

path = str(out)
ds=xr.open_dataset(path)
Nlat = len(ds.latitude)
Nlon = len(ds.longitude)
Ntime = len(ds.time)
Npoints = Nlat*Nlon
u = np.reshape(ds.uo.values, (Ntime, Npoints)).T
v = np.reshape(ds.vo.values, (Ntime, Npoints)).T

reorder = np.vstack([np.arange(Npoints),
                     np.arange(Npoints, 2*Npoints)]).reshape((-1,),
                                                             order='F')


X = np.vstack([u,v])
X = X[reorder,:]


Xkp1 = X[:,2:]
Xk = X[:,1:-1]
Xkm1 = X[:,:-2]

# secant predictor
Uk = 2*Xk - Xkm1

u_pr,v = get_uv(Uk)
u_tr,v = get_uv(Xkp1)
diff = np.abs(u_tr-u_pr)


plt.close('all');
plt.subplot(2,2,1)
plt.imshow(u_pr[100,:,:]);
plt.gca().invert_yaxis();
plt.subplot(2,2,2)
plt.imshow(u_tr[100,:,:]);
plt.gca().invert_yaxis();
plt.subplot(2,2,3)
plt.imshow(diff[100,:,:]);
plt.gca().invert_yaxis();
plt.pause(1);


def get_uv(X):
    Nt = X.shape[1]
    u = np.reshape((X[0::2, :]).T, (Nt, Nlat, Nlon))
    v = np.reshape((X[1::2, :]).T, (Nt, Nlat, Nlon))    
    return u,v














