# Set True if resnet and/or predictor weights are allowed to get
# updated during training
trainable_resnet = True
trainable_predictor = True

# learning rate
learning_rate = 1e-4

# weight on prediction loss
alpha = 1

# weight on prediction loss in the latent space
alpha_ls = 1

# weight on VAE KL loss
beta = 1e-5

# weight on VAE reconstruction loss
gamma = 1

# hybridization: 'product', 'concat', 'add'
hybridization = 'product'

# activation after hybridization
activation = 'leaky_relu'
