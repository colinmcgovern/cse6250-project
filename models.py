import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import time 

from torch.nn.functional import sigmoid
from torch.nn import Linear
from torch.nn import Conv1d
from torch.nn import MaxPool1d
from torch.nn import Dropout
from torch.nn.functional import relu

from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

import re
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from collections import Counter
import sys
	
import torchbearer
import torchbearer.callbacks as callbacks
from torchbearer import Trial
from torchbearer.callbacks import L2WeightDecay, ExponentialLR

# SVM ##################################################################################
# class MySVM(nn.Module):
# 	"""Support Vector Machine"""

# 	def __init__(self, train_data_x, kernel, num_classes=5,
# 				gamma_init=1.0, train_gamma=True):
# 		super(MySVM,self).__init__()

# 		self._train_data_x = train_data_x

# 		if kernel == 'linear':
# 			self._kernel = self.linear
# 			self._num_c = train_data_x.size(1) * 300
# 		elif kernel == 'rbf':
# 			self._kernel = self.rbf
# 			self._num_c = train_data_x.size(0)
# 			self._gamma = torch.nn.Parameter(torch.FloatTensor([gamma_init]),
# 											 requires_grad=train_gamma)

# 		self._w = torch.nn.Linear(in_features=self._num_c, out_features=num_classes)

# 	def rbf(self, x, gamma=1):
# 		y = self._train_data_x.repeat(self._train_data_x.size(0), 1, 1)
# 		return torch.exp(-self._gamma*((x[:,None]-y)**2).sum(dim=2))

# 	@staticmethod
# 	def linear(x):
# 		return x

# 	def forward(self, x):
# 		# h = x.matmul(self.w.t()) + self.b
# 		print('x')
# 		print(x)
# 		print(x.size())
# 		y = self._kernel(x)
# 		y = self._w(y)
# 		return y

# def hinge_loss(y_pred, y_true):
# 	# return torch.mean(torch.clamp(1 - y_pred.t() * y_true, min=0))
# 	return torch.max(torch.zeros_like(y), 1-y_pred.t()*y_true).mean()

# def mypause(interval):
# 	backend = plt.rcParams['backend']
# 	if backend in matplotlib.rcsetup.interactive_bk:
# 		figManager = matplotlib._pylab_helpers.Gcf.get_active()
# 		if figManager is not None:
# 			canvas = figManager.canvas
# 			if canvas.figure.stale:
# 				canvas.draw_idle()
# 			canvas.start_event_loop(interval)
# 			return

# @callbacks.on_start
# def scatter(_):
# 	plt.figure(figsize=(5, 5))
# 	plt.ion()
# 	plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)


# @callbacks.on_step_training
# def draw_margin(state):
# 	if state[torchbearer.BATCH] % 10 == 0:
# 		w = state[torchbearer.MODEL].w[0].detach().to('cpu').numpy()
# 		b = state[torchbearer.MODEL].b[0].detach().to('cpu').numpy()

# 		z = (w.dot(xy) + b).reshape(x.shape)
# 		z[np.where(z > 1.)] = 4
# 		z[np.where((z > 0.) & (z <= 1.))] = 3
# 		z[np.where((z > -1.) & (z <= 0.))] = 2
# 		z[np.where(z <= -1.)] = 1

# 		if CONTOUR in state:
# 			for coll in state[CONTOUR].collections:
# 				coll.remove()
# 			state[CONTOUR] = plt.contourf(x, y, z, cmap=plt.cm.jet, alpha=0.5)
# 		else:
# 			state[CONTOUR] = plt.contourf(x, y, z, cmap=plt.cm.jet, alpha=0.5)
# 			plt.tight_layout()
# 			plt.show()

# 		mypause(0.001)

# CNN ##################################################################################
class MyCNN(nn.Module):
	def __init__(self, num_classes=5, window_sizes=(3,4,5)):
		super(MyCNN, self).__init__()

		self.convs = nn.ModuleList([
			nn.Conv2d(1, 100, [window_size, 300], padding=(window_size - 1, 0))
			for window_size in window_sizes
		])

		self.drop = Dropout(0.3)
	
		self.fc = nn.Linear(100 * len(window_sizes), num_classes)

	def forward(self, x):

		x = torch.unsqueeze(x, 1)
		xs = []
		for conv in self.convs:
			x2 = torch.relu(conv(x))
			x2 = torch.squeeze(x2, -1)
			x2 = F.max_pool1d(x2, x2.size(2))
			xs.append(x2)
		x = torch.cat(xs, 2)

		x = self.drop(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

# CNN ##################################################################################
class MyFFNN(nn.Module):
	def __init__(self, num_classes=5, buffer_size=-1):
		super(MyFFNN, self).__init__()
		self.layers = nn.Sequential(
            nn.Linear(300*buffer_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
		
	def forward(self, x):
		x = x.view(x.size(0), -1) 
		x = self.layers(x)
		return x