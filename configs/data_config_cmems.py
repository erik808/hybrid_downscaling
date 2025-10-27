# --------------------------------------------------------------------
# Data configuration specific for the synthetic CMEMS NWS set
# --------------------------------------------------------------------
compute_data = False
scaling_range = (0, 1)
split_factor = 4 / 5
# time_range = slice('2023-01-01', '2024-12-31')
time_range = slice('2023-01-01', '2024-12-31')

# important directories and files
data_dir = 'data'
transect_dir = f'{data_dir}/transects'
data_files = \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
     'uo-vo-zos_4.22E-7.78E_56.81N-58.69N_2022-12-01-2025-09-30/*.nc')
coarse_data_files = \
    (f'{data_dir}/coarse_data')
bathy_file = \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
     f'static_multi-vars_4.22E-7.78E_56.81N-58.69N_0.49-643.57m.nc')
coords_file =  \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
     f'static_e1t-e2t-e3t_4.22E-7.78E_56.81N-58.69N_0.49-643.57m.nc')

# keys to use
data_vars = ['uo', 'vo', 'zos']

# cropping of 2d fields
lat_crop = slice(3, -2)
lon_crop = slice(0, -1)

# Coarsening
# parameters for Gaussian filter
sigma = [1, 1.5, 1.5]
coarsening_factor = 4
coarse_data_prefix = \
    (f"data_LR_r{coarsening_factor}_sigm"
     f"{str(sigma).replace(', ','_')}_")

# scalers
scalers_file =  \
    (f'{data_dir}/scalers_{time_range.start}_{time_range.stop}_'
     f'{coarse_data_prefix}.dill')
