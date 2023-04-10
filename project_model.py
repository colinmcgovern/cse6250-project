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

PATH_OUTPUT = "."
NUM_EPOCHS = 50
BATCH_SIZE = 4
USE_CUDA = True
NUM_WORKERS = 0
save_file = 'MyCNN.pth'

# file_location = "/FileStore/tables/500_Reddit_users_posts_labels.csv" #DATABRICKS
file_location = "500_Reddit_users_posts_labels.csv" #LOCAL
string_to_num = {"Supportive": 0 , "Indicator": 1, "Ideation": 2,
 					"Behavior": 3, "Attempt": 4}

d = pd.read_hdf('mini.h5')
words = list(d.index)
key_to_index = {}
iter = 0
for word in words:
	key_to_index[word] = iter
	iter = iter + 1

model = Word2Vec()
model.wv.key_to_index = key_to_index
model.wv.vectors = d.values
del d

df=pd.read_csv(file_location)
regex = re.compile('[^a-zA-Z\s]')
for i in range(0,len(df)):
	df.iloc[i][1] = regex.sub(' ', df.iloc[i][1]).replace('gt','').split()
df = df.sample(frac=1)[0:200] #For testing

#for testing
# label_counts = Counter(df.iloc[:,2].values)
# smallest_count = min(label_counts.values())
# all_labels = label_counts.keys()
# new_df = pd.DataFrame()
# for label in all_labels:
# 	to_add = df[df['Label']==label][0:smallest_count]
# 	if(len(new_df)==0):
# 		new_df = to_add
# 	else:
# 		new_df = new_df.append(to_add, ignore_index = True)
# df = new_df.sample(frac=1)
#for testing

longest_post = 0
for i in range(0,len(df)):
	if(len(df.iloc[i][1])>longest_post):
		longest_post = len(df.iloc[i][1])

def convert_posts(posts):
	new_posts = [[0]*300]*longest_post
	for i in range(0,len(posts)):
		key = '/c/en/' + posts[i].lower()
		if key in key_to_index.keys():
			new_posts[i] = model.wv[key]
	return np.array(new_posts)

class MyDataset():
	def __init__(self):
		x=df.iloc[:,1].map(convert_posts)
		x2 = []
		for i in x:
			x2.append(torch.from_numpy(i))
		# self.x_train=torch.cat(x2, dim=2)
		self.x_train=torch.stack((x2))
		y=df.iloc[:,2].values
		y=[string_to_num[i] for i in y]
		self.y_train=torch.tensor(y,dtype=torch.float64)
	def __len__(self):
		return len(self.y_train)
	def __getitem__(self,idx):
		return self.x_train[idx],self.y_train[idx]

def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():
		BATCH_SIZE = target.size(0)
		_, pred = output.max(1)
		correct = pred.eq(target).sum()
		return correct * 100.0 / BATCH_SIZE

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def train(model_param, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()
	model_param.train()
	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)
		optimizer.zero_grad()
		output = model_param(input.float())

		print("train")
		print("input {}".format(input))
		print("output {}".format(output))
		print(type(output))
		print(list(map(lambda x: x.index(max(x)),output.tolist())))
		print("target {}".format(target.long()))
		print(type(target))

		loss = criterion(output, target.long())
		print("loss")
		print(loss)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
		loss.backward()
		optimizer.step()
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))
	return losses.avg, accuracy.avg


def evaluate(model_param, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()
	results = []
	model_param.eval()
	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):
			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)
			output = model_param(input.float())

			print("evaluate")
			print("input {}".format(input))
			print("output {}".format(output))
			print(type(output))
			print(list(map(lambda x: x.index(max(x)),output.tolist())))
			print("target {}".format(target))
			print(type(target))

			loss = criterion(output, target.long())
			print("loss")
			print(loss)
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))
			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))
			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))
	return losses.avg, accuracy.avg, results

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig("losses.png",pad_inches=0)
	plt.clf() 
	plt.cla() 
	plt.figure()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig("accuracies.png",pad_inches=0)
	pass

def plot_confusion_matrix(results, class_names):
	plt.clf() 
	plt.cla() 
	plt.rc('font', size=8)		  # controls default text sizes
	plt.rc('axes', titlesize=8)	 # fontsize of the axes title
	plt.rc('axes', labelsize=8)	# fontsize of the x and y labels
	plt.rc('xtick', labelsize=8)	# fontsize of the tick labels
	plt.rc('ytick', labelsize=8)	# fontsize of the tick labels
	plt.rc('legend', fontsize=8)	# legend fontsize
	plt.rc('figure', titlesize=8)  # fontsize of the figure title
	out = list(zip(*results))
	confusion_matrix = metrics.confusion_matrix(list(out[0]),list(out[1]),normalize='true')
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_names)
	cm_display.plot()
	plt.savefig("confusion.png",pad_inches=0, dpi=199)
	pass

class MyCNN(nn.Module):
	def __init__(self, num_classes=5, window_sizes=(3,4,5)):
		super(MyCNN, self).__init__()

		self.convs = nn.ModuleList([
			nn.Conv2d(1, 100, [window_size, 300], padding=(window_size - 1, 0))
			for window_size in window_sizes
		])

		self.drop = Dropout(0.3)
	
		self.fc = nn.Linear(100 * len(window_sizes), num_classes)

		# self.fc1 = nn.Linear(300,100)
		# self.fc2 = nn.Linear(100,100)
		# self.fc3 = nn.Linear(100,100)
		# self.fc4 = nn.Linear(100,num_classes)

	def forward(self, x):

		x = torch.unsqueeze(x, 1)
		xs = []
		for conv in self.convs:
			x2 = torch.relu(conv(x))
			x2 = torch.squeeze(x2, -1)
			x2 = F.max_pool1d(x2, x2.size(2))
			xs.append(x2)
		x = torch.cat(xs, 2)

		# x = torch.unsqueeze(x, 1)
		# xs = []
		# for conv in self.convs:
		# 	x2 = torch.tanh(conv(x))
		# 	x2 = torch.squeeze(x2, -1)
		# 	x2 = F.max_pool1d(x2, x2.size(2))
		# 	xs.append(x2)
		# x = torch.cat(xs, 2)

		x = self.drop(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		# LEAVE OFF!!!
		# x = F.softmax(x, dim = 1)

<<<<<<< HEAD
		# x = self.fc1(x)
		# x = self.fc2(x)
		# x = self.fc3(x)
		# x = self.fc4(x)
=======
		# probs = F.softmax(logits, dim = 1)
>>>>>>> dfc1e8abe51cc4dc96645cf1d0abcb455616cc97

		return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

CNN_model = MyCNN()
print('CNN_model')
print(CNN_model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN_model.parameters())

print("CNN_model.parameters()")
print(CNN_model.parameters())

CNN_model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

VALID_FRACTION = 4
valid_size = int(len(df)/VALID_FRACTION)
test_size = int(len(df)/VALID_FRACTION)
train_size = len(df)-valid_size-test_size

print('valid_size')
print(valid_size)
print('test_size')
print(test_size)
print('train_size')
print(train_size)

train_loader, valid_loader, test_loader = torch.utils.data.random_split(MyDataset(), [train_size,valid_size,test_size])
train_loader = DataLoader(train_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# train_size = len(df)
# # train_loader, valid_loader, test_loader = torch.utils.data.random_split(MyDataset(), [train_size,0,0])
# train_loader = DataLoader(MyDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# valid_loader = train_loader
# test_loader = train_loader



# a, b = MyDataset()
# c = TensorDataset(a,b)
# train_loader = DataLoader(MyDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# valid_loader = DataLoader(MyDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# test_loader = DataLoader(MyDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

#del
# print('train_loader')
# for data, targets in train_loader:
# 	print('data')
# 	print(data)
# 	print('targets')
# 	print(targets)

# print('valid_loader')
# for data, targets in valid_loader:
# 	print('data')
# 	print(data)
# 	print('targets')
# 	print(targets)

# print('test_loader')
# for data, targets in test_loader:
# 	print('data')
# 	print(data)
# 	print('targets')
# 	print(targets)

#del

print("train_loader")
for data, targets in train_loader:
	print('data')
	print(data)
	print('targets')
	print(targets)

print("valid_loader")
for data, targets in valid_loader:
	print('data')
	print(data)
	print('targets')
	print(targets)

for epoch in range(NUM_EPOCHS):
	print("epoch: {}".format(epoch))

	train_loss, train_accuracy = train(CNN_model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(CNN_model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(CNN_model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

best_CNN_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
test_loss, test_accuracy, test_results = evaluate(best_CNN_model, device, test_loader, criterion)

plot_confusion_matrix(test_results, string_to_num.keys())