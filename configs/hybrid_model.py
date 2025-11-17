# Set True if resnet and/or predictor weights are allowed to get
# updated during training
trainable_resnet = True
trainable_predictor = True

# select losses to include
# loss_list = ['outer_pred', 'inner_pred', 'reconstruction', 'KL']
loss_list = ['outer_pred']  # , 'reconstruction']

# learning rate
learning_rate = 2e-3

# weight on outer prediction loss
alpha = 1

# weight on prediction loss in the latent space (inner prediction)
alpha_ls = 1e-3

# weight on VAE KL loss
beta = 1e-5

# weight on VAE reconstruction loss
gamma = 1e-3

# hybridization: 'product', 'concat', 'add'
hybridization = 'product'
