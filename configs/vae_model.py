# learning rate in Adam optimizer
learning_rate = 1e-4

# weight on KL loss
beta = 0.0

# weight on reconstruction loss
gamma = 1

# weight on latent space size
alpha_ls = 1

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
deterministic_mode = True

# bypass the whole vae
bypass_vae = False

# select sampling: 'spatial', 'dense'
sampling_type = 'spatial'

# latent space size
dense_units = 64  # used in dense mode
latent_space = 32  # used in dense mo
