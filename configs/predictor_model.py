# Set True if the encoder and decoder weights are allowed to get
# updated during training
trainable_encoder = False
trainable_decoder = False

# learning rate
learning_rate = 1e-4

# model that does the actual prediction in the latent space
# options: 'simpleRNN', 'dense', 'conv3d', 'convlstm'
predictor = 'conv3d'
