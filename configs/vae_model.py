# learning rate in Adam optimizer
learning_rate = 5e-5

# weight on KL loss
beta = 1e-5

# weight on reconstruction loss
gamma = 1

#
upsampling_method = 'bilinear'

# activation type ('prelu', 'relu', 'leaky_relu', 'elu')
activation = 'leaky_relu'
activation_out = 'tanh_scaled'

# number of filters in conv layers
input_filters = 64
filters = 64  # tested: 64
num_filters_hybrid = 64

# number of down and upsampling convolutions
num_layers = 4  # tested: 4

# deterministic mode bypasses the sampling layer and uses the mean
# only
deterministic_mode = False

# bypass the whole vae
bypass_vae = False
