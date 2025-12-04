# Set True if the encoder and decoder weights are allowed to get
# updated during training
trainable_encoder = True
trainable_decoder = True

# learning rate
learning_rate = 1e-4

alpha = 1.0
alpha_ls = 1.0

# weight on KL loss
beta = 0.0

# weight on reconstruction loss
gamma = 1

# model that does the actual prediction in the latent space
# options: 'identity', 'simpleRNN',
#          'dense', 'lstm', 'conv3d',
#          'convlstm', 'DMD', 'DMDc'
predictor = 'dense'

lambdaDMD = 0.0
cutoffDMD = 0.1
alphaDMD = 1.0
biasDMD = True

output_filters = 128

convlstm_filters = 64

activation = 'leaky_relu'

kernel_regularizer = None  # {'L2': 1e-2}
recurrent_regularizer = None  # {'L2': 1e-2}
recurrent_dropout = 0.4  # 0.4
dense_units = 2048
