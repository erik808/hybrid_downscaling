import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# hyperparameters

random_seed = 49
learning_rate = 0.0005
batch_size = 256
num_epochs = 30
num_classes = 10

def get_dataloaders_mnist(batch_size, num_workers=0,
                          train_transforms=None,
                          test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)

    valid_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=test_transforms)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader

train_loader, valid_loader, test_loader = \
    get_dataloaders_mnist(batch_size=batch_size,
                          num_workers=2)

# visualize data
# train_iter = iter(train_loader)
# images, labels = next(train_iter)

# plt.close('all')
# plt.pcolormesh(images[0,0,:,:], cmap='Greys')
# print(labels[0].numpy())
# plt.gca().invert_yaxis()
# plt.pause(1)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1,1),
                      kernel_size=(3,3),
                      padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(32,64, stride=(2,2),
                      kernel_size=(3,3),
                      padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(64,64, stride=(2,2),
                      kernel_size=(3,3),
                      padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,64, stride=(2,2),
                               kernel_size=(3,3),
                               padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,32,stride=(2,2),
                               kernel_size=(3,3),
                               padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, stride=(1,1),
                               kernel_size=(3,3),
                               padding=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoder()

# Move the model to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'model lives on the {device}')
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_autoencoder(num_epochs, model, optimizer,
                      train_loader, loss_fn=None,
                      logging_interval=100,
                      skip_epoch_stats=False,
                      save_model=None):
    
    log_dict = {'train_loss_per_batch' : [],
                'train_loss_per_epoch' : []}

    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):
        # put the model in train mode
        model.train()

        for batch_idx, (features, _) in enumerate(train_loader):
            # forward and back prop:
            logits = model(features)

            # zero the gradients
            optimizer.zero_grad()

            # compute loss
            loss = loss_fn(logits, features)

            # compute gradients
            loss.backward()

            # update model parameters:
            optimizer.step()

            # do some logging
            log_dict['train_loss_per_epoch'].append(loss.item())

            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1}/{num_epochs} |'
                      f' Batch {batch_idx}/{len(train_loader)} |'
                      f' Loss {loss}')

        if not skip_epoch_stats:

            # put model in evaluation mode
            model.eval()

            # this saves memory during inference (same as torch.no_grad())
            with torch.set_grad_enabled(False):
                train_loss = compute_epoch_loss_autoencoder


            


def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0

    with torch.no_grad():
        for (features, _) in data_loader:
            features = features.to(device)
            logits = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples

        return curr_loss
            
