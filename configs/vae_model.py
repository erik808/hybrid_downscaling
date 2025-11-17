# learning rate in Adam optimizer
learning_rate = 2e-3

# weight on KL loss
beta = 1e-5

# weight on reconstruction loss
gamma = 1

# activation type ('prelu', 'relu', 'leaky_relu', 'elu')
activation = 'leaky_relu'

# number of filters in conv layers
filters = 128  # tested: 64

# number of down and upsampling convolutions
num_layers = 1  # tested: 4

# deterministic mode bypasses the sampling layer and uses the mean
# only
deterministic_mode = True
