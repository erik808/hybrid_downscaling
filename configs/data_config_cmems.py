# --------------------------------------------------------------------
# Data configuration specific for the synthetic CMEMS NWS set
# --------------------------------------------------------------------
compute_data = False
scaling_range=(0, 1)
split_factor = 4 / 5

# method to 'destroy' small scales in both space and time
coarsening_method = 'gaussian_filter'
sigma = [1, 1.5, 1.5]

truncation = 100  # deprecated
coarsen_in_time = False  # deprecated
differences = False  # deprecated
detide = False  # disabled due to package issues
