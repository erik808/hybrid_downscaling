# learning rate in Adam optimizer
learning_rate = 1e-4

# upsampling method: 'subpixel', 'bilinear'
upsampling_method = 'subpixel'

# activation function  (except output sigmoid)
# activation = 'leaky_relu'
activation = 'prelu'
activation_out = 'tanh_scaled'

# number of residual blocks
residual_blocks = 6

# standard number of filters
num_filters = 64

# coupling layer
enable_coupling_layer = False
# number of filters for coupling layer
num_filters_coupling = 64

# number of additional layers in output block (not including output
# sigmoid)
num_output_layers = 0

# number of necessary upsampling blocks is inferred from LR and HR grids
