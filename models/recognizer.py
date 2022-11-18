import numpy as np
import torch
from torch import functional as F
from torch import nn, optim


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
		self.conv4  = nn.Conv2d(64, 128, 3, padding=1)
		self.conv5  = nn.Conv2d(128, 256, 3, padding=1)
		self.conv6  = nn.Conv2d(256, 512, 3, padding=1)
		#self.conv7  = nn.Conv2d(512, 1024, 3, padding=1)
		# Fully connected layers
		self.fc1    = nn.Linear(512*(im_size//32)**2, 1024)
		self.fc2    = nn.Linear(1024, 12)
		self.fc 	= nn.Linear(512*(im_size//32)**2, 12)
		
		# Pooling
		self.pool   = nn.MaxPool2d(2, 2)
		# Activation functions
		self.relu   = nn.ReLU()
		# Dropout
		self.dropout 	= nn.Dropout(p=0.2)
		self.dropout_fc = nn.Dropout(p=0.5)
		# Softmax
		self.softmax = nn.Softmax(dim=2)
		# Batch normalization
		self.batch_norm_1 = nn.BatchNorm2d(16)
		self.batch_norm_2 = nn.BatchNorm2d(32)
		self.batch_norm_3 = nn.BatchNorm2d(64)
		self.batch_norm_4 = nn.BatchNorm2d(128)
		self.batch_norm_5 = nn.BatchNorm2d(256)
		self.batch_norm_6 = nn.BatchNorm2d(512)
		self.batch_norm_7 = nn.BatchNorm2d(1024)

	def forward(self, x, train = True):
		# Convolutional layers
		x = self.pool(self.batch_norm_1(self.relu(self.conv1(x))))
		x = self.dropout(x)
		x = self.pool(self.batch_norm_2(self.relu(self.conv2(x))))
		x = self.pool(self.batch_norm_3(self.relu(self.conv3(x))))
		x = self.dropout(x)
		x = self.pool(self.batch_norm_4(self.relu(self.conv4(x))))
		x = self.pool(self.batch_norm_5(self.relu(self.conv5(x))))
		x = self.dropout(x)
		x = self.batch_norm_6(self.relu(self.conv6(x)))
		#x = self.batch_norm_7(self.relu(self.conv7(x)))
		
		# Flatten
		x = x.view(-1, 512*(self.im_size//32)**2)
		
		# Fully connected layers
		# x = self.relu(self.fc1(x))
		# x = self.dropout_fc(x)
		# x = self.fc2(x)
		x = self.fc(x)

		# Divide into relevant features
		x = x.view(-1, 4, 3)

		# Softmax each feature
		x = self.softmax(x)

		return x




class ResBlock_Classic(nn.Module):

	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()

		if out_channels % in_channels != 0:
			raise Exception("ResBlock Error: number of output channels {} not divisible by the number of input channels {}!".format(out_channels, in_channels))

		# Convolitional layers
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1)

		# Batch normalizations
		self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
		self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)

		# ReLU's
		self.relu1 = nn.ReLU()
		self.relu2 = nn.ReLU()

	def forward(self, x : torch.Tensor):
		residue = x.clone()
		x = self.conv1(x)
		x = self.batch_norm1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.batch_norm2(x)
		
		# If the number of channels has changes, expand the residue
		if residue.shape[1] != x.shape[1]:
			residue = residue.repeat(1,x.shape[1]//residue.shape[1],1,1)
		return self.relu2(residue+x)