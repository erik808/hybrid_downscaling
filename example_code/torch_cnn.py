import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

loss = nn.MSELoss()
optimizer_class = optim.Adam

class CNN(nn.Module):    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv1 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))      

