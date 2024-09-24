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

time_start = "2023-01-01T00:00:00"
time_end = "2023-12-31T23:00:00"

fetch = 'coords'
print(f'fetch {fetch}')

def fetch_wrapper(box, **kwargs):
    out = cm.subset(
        minimum_longitude=box['min_lon'],
        maximum_longitude=box['max_lon'],
        minimum_latitude=box['min_lat'],
        maximum_latitude=box['max_lat'],
        minimum_depth=box['min_dep'],
        maximum_depth=box['max_dep'],
        force_download=True,
        netcdf_compression_enabled=True,
        # output_filename="data.nc",
        overwrite_output_data=True,
        overwrite_metadata_cache=False,
        netcdf_compression_level=0,
        **kwargs
    )
    return out


if fetch == 'uv':

    dataset_id_HR = "cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i"
    dataset_id_LR = "cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i"
    variables = ["uo", "vo"]

    for ds_id in [dataset_id_HR, dataset_id_LR]:
        out = fetch_wrapper(box_NwCC,
                            dataset_id=ds_id,
                            variables=variables,
                            start_datetime=time_start,
                            end_datetime=time_end)


elif fetch == 'bathy':

    ds_id = "cmems_mod_nws_phy_anfc_0.027deg-3D_static"
    dataset_part="bathy"
    variables=["deptho", "deptho_lev", "mask"]

    out = fetch_wrapper(box_NwCC,
                        dataset_id=ds_id,
                        dataset_part=dataset_part,
                        variables=variables,
                        start_datetime=time_start,
                        end_datetime=time_end)
    
elif fetch == 'coords':
    ds_id = "cmems_mod_nws_phy_anfc_0.027deg-3D_static"
    dataset_part="coords"
    variables=["e1t", "e2t", "e3t"]

    out = fetch_wrapper(box_NwCC,
                        dataset_id=ds_id,
                        dataset_part=dataset_part,
                        variables=variables,
                        start_datetime=time_start,
                        end_datetime=time_end)
