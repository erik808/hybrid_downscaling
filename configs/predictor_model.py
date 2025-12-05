# Set True if the encoder and decoder weights are allowed to get
# updated during training
trainable_encoder = True
trainable_decoder = True

# learning rate
learning_rate = 1e-4

alpha_outer = 1.0
alpha_inner = 1.0

# weight on KL loss
beta = 0.0

# weight on reconstruction loss
gamma = 1

# model that does the actual prediction in the latent space
# options: 'identity', 'simpleRNN',
#          'dense', 'lstm', 'conv3d',
#          'convlstm', 'DMD', 'DMDc'
predictor = 'ESN'

esn_dmd_bias = False,

esn_dmd_pars = {
    'Nr': 5000,
    'rhoMax': 0.8,
    'entriesPerRow': 3,
    'alpha': 0.2,
    'tikhonov_lambda': 1.0e-5,
    'fCutoff': 0.1,
    'squaredStates': 'even',
}

output_filters = 256

convlstm_filters = 64

activation = 'leaky_relu'

kernel_regularizer = None  # {'L2': 1e-2}
recurrent_regularizer = None  # {'L2': 1e-2}
recurrent_dropout = 0.4  # 0.4
dense_units = 4
