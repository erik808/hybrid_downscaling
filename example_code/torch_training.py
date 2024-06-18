import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

training_set = torchvision.datasets.FashionMNIST('./data', train=True,
                                                 transform=transform,
                                                 download=True)

validation_set = torchvision.datasets.FashionMNIST('./data', train=False,
                                                   transform=transform,
                                                   download=True)

# shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=4,
                                              shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=4,
                                                shuffle=False)

# labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Report split sizes
print(f'Training set has {len(training_set)} instances')
print(f'Validation set has {len(validation_set)} instances')

#---------------------------------------------------------------
#---------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = GarmentClassifier()

# ------------------------------------------------------------------
loss_fn = torch.nn.CrossEntropyLoss()

# batches of 4
dummy_outputs = torch.rand(4,10)

# dummy classifications
dummy_labels = torch.tensor([1,5,3,7])

print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print(f'total loss {loss.item()}')

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# ------------------------------------------------------------------
def train_one_epoch(epoch_index, tb_writer):
    # tb_writer: tensorboard writer
    
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(training_loader):
        # data is a input + label pair
        inputs, labels = data

        # zero the gradients...
        optimizer.zero_grad()

        # predictions for this batch
        outputs = model(inputs)

        # Compute loss and its gradients:
        loss = loss_fn(outputs, labels)
        loss.backward()

        # adjust the weights
        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'  batch {i+1} loss {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
    
    return last_loss
    
# ------------------------------------------------------------------
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000

do_training=False
if do_training:
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch_number+1}')

        # enable gradient tracking
        model.train(True)
        # do a pass over the data
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0

        # set model to evaluation mode
        model.eval()
        # disable gradient computation
        with torch.no_grad():
            # evaluate model with validation data
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i+1)
        print(f'(LOSS train {avg_loss} valid {avg_vloss}')

        # do some logging
        writer.add_scalars('Training vs. validation loss',
                           {'Training':avg_loss,
                            'Validation':avg_vloss},
                           epoch_number+1)

        writer.flush()

        # track best performance

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # save the model state
            model_path = f'model_{timestamp}_{epoch_number}'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
else:    
    model = GarmentClassifier()
    model.load_state_dict(torch.load('model_20240618_100612_4'))  


import matplotlib.pyplot as plt
import numpy as np

def matplotlib_imshow(img, one_channel=False):
    plt.close('all')
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()

    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))

    plt.pause(1)
        
dataiter = iter(validation_loader)
images, labels = next(dataiter)

img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
