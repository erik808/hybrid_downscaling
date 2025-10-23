# --------------------------------------------------------------------
# Data configuration specific for the synthetic CMEMS NWS set
# --------------------------------------------------------------------
compute_data = False
scaling_range = (0, 1)
split_factor = 4 / 5
time_range = slice('2023-01-01', '2024-12-31')

# important directories and files
data_dir = 'data'
transect_dir = f'{data_dir}/transects'
uv_data_files = \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
     'uo-vo-zos_4.22E-7.78E_56.81N-58.69N_2022-12-01-2025-09-30/*.nc')
bathy_file = \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
     f'static_multi-vars_4.22E-7.78E_56.81N-58.69N_0.49-643.57m.nc')
coords_file =  \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
     f'static_e1t-e2t-e3t_4.22E-7.78E_56.81N-58.69N_0.49-643.57m.nc')

# cropping of 2d fields
lat_crop = slice(3, -2)
lon_crop = slice(0, -1)

# method to 'destroy' small scales in both space and time
coarsening_method = 'gaussian_filter'
sigma = [1, 1.5, 1.5]
