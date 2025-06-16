# --------------------------------------------------------------------
# Data configuration specific for the synthetic CMEMS NWS set
# --------------------------------------------------------------------
detide = False  # implementation disabled due to package issues
compute_data = False
coarsen_in_time = False
coarsening_method = 'gaussian_filter'
sigma = [1, 1.5, 1.5]
truncation = 100
