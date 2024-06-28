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
torch.manual_seed(49)
learning_rate = 0.0005
batch_size = 256
num_epochs = 1
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
train_iter = iter(train_loader)
images, labels = next(train_iter)

plt.close('all')
# plt.pcolormesh(images[0,0,:,:], cmap='Greys')
# print(labels[0].numpy())
# plt.gca().invert_yaxis()
# plt.pause(1)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.relu = nn.LeakyReLU(0.01)
        self.kernel_size = (3,3)

        self.conv1 = nn.Conv2d(1, 32, stride=(1,1),
                               kernel_size=self.kernel_size,
                               padding=1)

        self.conv2 = nn.Conv2d(32,64, stride=(2,2),
                               kernel_size=self.kernel_size,
                               padding=1)

        self.conv3 = nn.Conv2d(64,64, stride=(2,2),
                               kernel_size=self.kernel_size,
                               padding=1)

        self.conv4 = nn.Conv2d(64,64, stride=(1,1),
                               kernel_size=self.kernel_size,
                               padding=1)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3136, 2)
        self.linear2 = nn.Linear(2, 3136)

        self.convTrans1 = nn.ConvTranspose2d(64,64, stride=(1,1),
                                             kernel_size=self.kernel_size,
                                             padding=1)

        self.convTrans2 = nn.ConvTranspose2d(64,64, stride=(2,2),
                                             kernel_size=self.kernel_size,
                                             padding=1)

        self.convTrans3 = nn.ConvTranspose2d(64,32,stride=(2,2),
                                             kernel_size=self.kernel_size,
                                             padding=0)

        self.convTrans4 = nn.ConvTranspose2d(32, 1, stride=(1,1),
                                             kernel_size=self.kernel_size,
                                             padding=0)

        self.sigm = nn.Sigmoid()

    def encoder(self, x):
        breakpoint()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x

    def decoder(self, x):
        x = self.linear2(x)
        # reshape
        x = x.view(-1, 64, 7, 7)
        x = self.convTrans1(x)
        x = self.relu(x)
        x = self.convTrans2(x)
        x = self.relu(x)
        x = self.convTrans3(x)
        x = self.relu(x)
        x = self.convTrans4(x)
        # trimming
        x = x[:,:,:-1, :-1]
        x = self.sigm(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# class Reshape(nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         self.shape = args

#     def forward(self, x):
#         return x.view(self.shape)

# class Trim(nn.Module):
#     def __init__(self, *args):
#         super().__init__()

#     def forward(self, x):
#         return x[:, :, :-1, :-1]

# class AutoEncoder2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3),  padding=1),
#             nn.LeakyReLU(0.01),
#             nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
#             nn.LeakyReLU(0.01),
#             nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
#             nn.LeakyReLU(0.01),
#             nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
#             nn.Flatten(),
#             nn.Linear(3136, 2)
#         )
#         self.decoder = nn.Sequential(
#             torch.nn.Linear(2, 3136),
#             Reshape(-1, 64, 7, 7),
#             nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
#             nn.LeakyReLU(0.01),
#             nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
#             nn.LeakyReLU(0.01),
#             nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
#             nn.LeakyReLU(0.01),
#             nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
#             Trim(),  # 1x29x29 -> 1x28x28
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         print(x.norm())
#         print(x.norm())
#         return x


model = AutoEncoder()
model.load_state_dict(torch.load('conv_autoencoder.pth'))


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
                train_loss = compute_epoch_loss_autoencoder(model,
                                                            train_loader,
                                                            loss_fn,
                                                            device)
                print(f' *** Epoch: {epoch+1}/{num+_epochs} '
                      f'| Loss {train_loss}')
                log_dict['train_loss_per_epoch'].append(train_loss.item())

        print(f'Time elapsed: {(time.time()-start_time)/60}')
    print(f'Total training time: {(time.time()-start_time)/60}')

    if save_model is not None:
        torch.save(model.state_dict(), save_model)

    return log_dict




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

log_dict = train_autoencoder(num_epochs=1,
                             model=model,
                             optimizer=optimizer,
                             train_loader=train_loader,
                             skip_epoch_stats=True,
                             logging_interval=50)

# torch.save(model.state_dict(), 'conv_autoencoder.pth')
