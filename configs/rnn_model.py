# Set True if the encoder and decoder weights are allowed to get
# updated during RNN training
trainable_VAE = False

# learning rate
learning_rate = 1e-4

# model that does the actual prediction in the latent space
# options: 'simpleRNN', 'dense', 'conv3d'
predictor = 'conv3d'
