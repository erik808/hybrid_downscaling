# Set True if the encoder and decoder weights are allowed to get
# updated during training
trainable_encoder = True
trainable_decoder = True

# learning rate
learning_rate = 1e-4

# weight on KL loss
beta = 1e-5

# weight on reconstruction loss
gamma = 1

# model that does the actual prediction in the latent space
# options: 'simpleRNN', 'dense', 'lstm', 'conv3d', 'convlstm'
predictor = 'lstm'

activation = 'leaky_relu'

kernel_regularizer = None  # {'L2': 1e-2}
recurrent_regularizer = None  # {'L2': 1e-2}
recurrent_dropout = 0.5  # 0.4
dense_units = 8
