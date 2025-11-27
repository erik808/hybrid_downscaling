# learning rate in Adam optimizer
learning_rate = 1e-4

# weight on KL loss
beta = 0.0

# weight on reconstruction loss
gamma = 1

# weight on latent space size
alpha_ls = 0.0

#
upsampling_method = 'bilinear'

# activation type ('prelu', 'relu', 'leaky_relu', 'elu')
activation = 'relu'
activation_out = 'tanh_scaled'

# number of down and upsampling convolutions
num_layers = 5  # tested: 4

kernel_size = 3

# number of filters in input
input_filters = 64
# filters in downsampling (and reversed in upsampling)
filters = [64, 64, 128, 256, 512, 512]
# filters in the coupling with resnet
num_filters_hybrid = 64


# deterministic mode bypasses the sampling layer and uses the mean
# only
deterministic_mode = True

# bypass the whole vae
bypass_vae = False

# select sampling: 'spatial', 'dense'
sampling_type = 'dense'

# latent space size
# dense_units = 4096  # used in dense mode
latent_space = 2048  # used in dense mode
