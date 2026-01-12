# --------------------------------------------------------------------
# Data configuration specific for the synthetic CMEMS NWS set
# --------------------------------------------------------------------
compute_data = False
split_factor = 4 / 5
time_range = slice('2023-01-01', '2025-07-01')

# short range for integration tests
# time_range_testing = slice('2023-09-01', '2023-12-31')
time_range_testing = slice('2023-06-01', '2023-12-01')
time_range_testing = slice('2023-09-01', '2023-09-05')

# tmp
# split_factor = 24 / 25
# time_range = slice('2023-01-01', '2025-02-01')

# overlap between training and testing dataset (in samples)
overlap = 5

# important directories and files
data_dir = 'data'
transect_dir = f'{data_dir}/transects'
data_files = \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-2D_PT1H-m_'
     'uo-vo-zos_4.22E-7.78E_56.81N-58.69N_'
     '2022-12-01-2025-09-30_zarr_nocons/data.zarr')
coarse_data_folder = \
    (f'{data_dir}/coarse_data')
bathy_file = \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
     f'static_multi-vars_4.22E-7.78E_56.81N-58.69N_0.49-643.57m.nc')
coords_file =  \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
     f'static_e1t-e2t-e3t_4.22E-7.78E_56.81N-58.69N_0.49-643.57m.nc')

# keys to use from cmems dataset
data_vars = ['uo', 'vo', 'zos']

# cropping of 2d fields
lat_crop = slice(3, -2)
lon_crop = slice(0, -1)

# Coarsening
# parameters for Gaussian filter
sigma = [2, 2, 2]
# coarsening factor for latitude and longitude
coarsening_factor = 16
# file manip
coarse_data_prefix = \
    (f"data_LR_r{coarsening_factor}_sigm"
     f"{str(sigma).replace(', ','_')}_")
coarse_data_file = \
    (f'{coarse_data_folder}/'
     f'{coarse_data_prefix}data.zarr')

# scalers for both coarse and HR data
scaling_range = (0, 1)
scaling_type = 'minmax'
scalers_file =  \
    (f'{data_dir}/scalers_{time_range.start}_{time_range.stop}_'
     f'{coarse_data_prefix}_{scaling_type}.dill')


# analysis parameters
window_size = 10 * 24
