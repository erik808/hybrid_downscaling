# --------------------------------------------------------------------
# Data configuration specific for the synthetic CMEMS NWS set
# --------------------------------------------------------------------
compute_data = False
scaling_range = (0, 1)
split_factor = 4 / 5

# important directories and files
data_dir = 'data'
transect_dir = f'{data_dir}/transects'
uv_data_files = \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-2D_PT15M-i_'
     f'uo-vo_4.23E-7.78E_56.81N-58.70N_2023-/*.nc')
bathy_file = \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
     f'static_multi-vars_4.23E-7.78E_56.81N-58.70N_0.49-643.57m.nc')
coords_file =  \
    (f'{data_dir}/cmems_mod_nws_phy_anfc_0.027deg-3D_'
     f'static_e1t-e2t-e3t_4.23E-7.78E_56.81N-58.70N_0.49-643.57m.nc')

# method to 'destroy' small scales in both space and time
coarsening_method = 'gaussian_filter'
sigma = [1, 1.5, 1.5]
