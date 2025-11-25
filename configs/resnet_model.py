# learning rate in Adam optimizer
learning_rate = 5e-5

# upsampling method: 'subpixel', 'bilinear'
upsampling_method = 'bilinear'

# activation function  (except output sigmoid)
activation = 'prelu'
activation_out = 'tanh_scaled'

# number of residual blocks
residual_blocks = 6

# standard number of filters
num_filters = 64

# number of filters for hybridization layer
num_filters_hybrid = 64

# number of additional layers in output block (not including output
# sigmoid)
num_output_layers = 2

# number of necessary upsampling blocks is inferred from LR and HR grids
