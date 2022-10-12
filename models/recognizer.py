import numpy as np
import torch
from torch import nn, optim, functional as F

class Recognizer(nn.Module):

    def __init__(self, im_size):
        # Call the parent constructor
        super().__init__()
        
        # Initial size of the image
        self.im_size = im_size

        # Convolutional layers
        self.conv1  = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2  = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3  = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4  = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5  = nn.Conv2d(64, 64, 3, padding=1)
        # Fully connected layers
        self.fc1    = nn.Linear(64*(im_size//32)**2, 12)
        
        # Pooling
        self.pool   = nn.MaxPool2d(2, 2)
        # Activation functions
        self.relu   = nn.ReLU()
        # Dropout
        self.dropout    = nn.Dropout(p=0.5)
        self.dropout_fc = nn.Dropout(p=0.2)
        # Softmax
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(-1, 64*(self.im_size//32)*(self.im_size//32))
        
        # Fully connected layers
        x = self.fc1(x)

        # Divide into relevant features
        x = x.view(-1, 4, 3)

        # Softmax each feature
        x = self.softmax(x)

        return x
