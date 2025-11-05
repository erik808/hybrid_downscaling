# learning rate in Adam optimizer
learning_rate = 5e-5

# weight on KL loss
beta = 1e-4

# weight on reconstruction loss
gamma = 1

# multiplying factors number of filters
filter_mult_start = 8
filter_mult_rest = 2

# activation type ('relu', 'leaky_relu', 'elu')
activation='leaky_relu'

# dense dimension
dense_dim = 256

# latent space dimension
latent_space_dim = 64

# deterministic mode bypasses the sampling layer and uses the mean
# only
deterministic_mode = False
