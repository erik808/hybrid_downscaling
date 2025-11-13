# Set True if the encoder and decoder weights are allowed to get
# updated during training
trainable_encoder = True
trainable_decoder = True

# learning rate
learning_rate = 1e-4

# weight on KL loss
beta = 1e-4

# weight on reconstruction loss
gamma = 1

# model that does the actual prediction in the latent space
# options: 'simpleRNN', 'dense', 'conv3d', 'convlstm'
predictor = 'conv3d'
