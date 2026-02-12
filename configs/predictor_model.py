# Set True if the encoder and decoder weights are allowed to get
# updated during training
trainable_encoder = True
trainable_decoder = True

# learning rate
learning_rate = 1e-4

# select losses to include: 'outer_pred', 'inner_pred',
# 'reconstruction', 'KL', 'ls_size'
loss_list = ['outer_pred']  # , 'inner_pred', 'ls_size', 'reconstruction']

alpha_outer = 1.0
alpha_inner = 1.0
alpha_ls = 1e-4

# weight on KL loss
beta = 1e-3

# weight on reconstruction loss
gamma = 1e1

# model that does the actual prediction in the latent space
# options: 'identity', 'simpleRNN',
#          'dense', 'lstm', 'conv3d',
#          'convlstm', 'DMD', 'DMDc', 'ESN', 'ESNc'
predictor = 'ESNc'


esn_dmd_pars = {
    'Nr': 10000,
    'rhoMax': 1.0,
    'entriesPerRow': 3,
    'alpha': 1.0,
    'tikhonov_lambda': 1e-2,
    'fCutoff': 0.0,
    'squaredStates': 'even',
    'keep_samples': 5000,
}

esn_dmd_bias = False,
output_filters = 256

convlstm_filters = 64

activation = 'leaky_relu'

kernel_regularizer = None  # {'L2': 1e-2}
recurrent_regularizer = None  # {'L2': 1e-2}
recurrent_dropout = 0.4  # 0.4
dense_units = 4
