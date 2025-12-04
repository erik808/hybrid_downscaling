# Set True if resnet and/or predictor weights are allowed to get
# updated during training
trainable_resnet = True
trainable_predictor = True

# select losses to include: 'outer_pred', 'inner_pred',
# 'reconstruction', 'KL', 'ls_size'
loss_list = ['outer_pred']  #, 'inner_pred', 'ls_size', 'reconstruction']

# learning rate
learning_rate = 1e-4

# weight on outer prediction loss
alpha_outer = 1

# weight on prediction loss in the latent space (inner prediction)
alpha_inner = 1e-1

# weight on latent space size
alpha_ls = 1e-4

# weight on VAE KL loss  (betaVAE)
beta = 1e-3

# weight on VAE reconstruction loss
gamma = 1e-1

# hybridization: 'product', 'concat', 'add'
hybridization = 'add'
