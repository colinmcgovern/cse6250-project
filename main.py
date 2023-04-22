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
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

DATA_SIZE = 10
NUM_EPOCHS = 1
MODEL_CHOICE = "CNN"
LABEL_CHOICE = "3+1"
USE_CF = 1

if(len(sys.argv)!=1):
	DATA_SIZE = int(sys.argv[1])
	NUM_EPOCHS = int(sys.argv[2])
	MODEL_CHOICE = sys.argv[3]
	LABEL_CHOICE = sys.argv[4]
	if(sys.argv[5]=="USE_CF"):
		USE_CF = 1
	else:
		USE_CF = 0

if(len(sys.argv)!=5):
	print("WARNING")
	print("Correct input is: script.py DATA_SIZE NUM_EPOCHS [SVM-RBF|SVM-L|RF|FFNN|CNN] [5|4|3+1] [USE_CF|NO_CF]")
	print("Using default input parameters...")

fold = -1

RUN_FOLDER = "DATA_SIZE_{}_NUM_EPOCHS_{}_MODEL_CHOICE_{}_LABEL_CHOICE_{}_USE_CF_{}_fold/".format(DATA_SIZE,NUM_EPOCHS,MODEL_CHOICE,LABEL_CHOICE,USE_CF,fold)
PATH_OUTPUT = "output/" + RUN_FOLDER

print("The output will be: {}".format(PATH_OUTPUT))

if not os.path.exists(PATH_OUTPUT):
	os.makedirs(PATH_OUTPUT)

BATCH_SIZE = 4
if(MODEL_CHOICE=="FFNN"):
	BATCH_SIZE = 1
USE_CUDA = True
NUM_WORKERS = 0
save_file = 'model_trained.pth'.format(MODEL_CHOICE,LABEL_CHOICE,NUM_EPOCHS,DATA_SIZE)

# file_location = "500_Reddit_users_posts_labels.csv" #LOCAL
file_location = "reddit_data_with_cf.csv" #LOCAL

# Numbers to labels
string_to_num = {}
num_to_string = {}
if(LABEL_CHOICE=="5"):
	string_to_num = {"Supportive": 0 , "Indicator": 1, "Ideation": 2,
	 					"Behavior": 3, "Attempt": 4}
	num_to_string = {0: "Supportive",1:"Indicator",2:"Ideation",3:"Behavior",4:"Attempt"}
elif(LABEL_CHOICE=="4"):
	string_to_num = {"Supportive": -1 , "Indicator": 0, "Ideation": 1,
	 					"Behavior": 2, "Attempt": 3}
	num_to_string = {0:"Indicator",1:"Ideation",2:"Behavior",3:"Attempt"}
elif(LABEL_CHOICE=="3+1"):
	string_to_num = {"Supportive": 0 , "Indicator": 0, "Ideation": 1,
	 					"Behavior": 2, "Attempt": 3}
	num_to_string = {0:"Control",1:"Ideation",2:"Behavior",3:"Attempt"}
else:
	exit()

# Words to vector
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
	df['post'][i] = regex.sub(' ', df['post'][i]).replace('gt','').split()

if(LABEL_CHOICE=="4"):
	df=df[df['label']!='Supportive']

DATA_SIZE = int(DATA_SIZE)
if(DATA_SIZE>0):
	df = df.sample(frac=1)[0:DATA_SIZE]
	print(df)

longest_post_len = 0
for i in range(0,len(df)):
	if(len(df.iloc[i][1])>longest_post_len):
		longest_post_len = len(df.iloc[i][1])

print("longest_post_len: {}".format(longest_post_len))

def convert_posts(posts):
	new_posts = [[0]*300]*longest_post_len
	for i in range(0,len(posts)):
		key = '/c/en/' + posts[i].lower()
		if key in key_to_index.keys():
			new_posts[i] = model.wv[key]
	return np.array(new_posts)

class MyDataset():
	def __init__(self,start,length,remove_mode,include_cf):
		cut_df = df
		if(remove_mode):
			cut_df = df[0:start].append(df[start+length:len(df)])
		else:
			cut_df = df[start:start+length]
		x=cut_df.iloc[:,1].map(convert_posts)
		if(include_cf==1):
			for i in range(len(cut_df)):
				cf_vals = list(cut_df.iloc[i].iloc[3:len(cut_df.columns)])
				cf_vals = (cf_vals + 300 * [0])[:300]
				x.iloc[i] = np.vstack([x.iloc[i],cf_vals])
				del cf_vals
		x2 = []
		for i in x:
			x2.append(torch.from_numpy(i))
		# self.x_train=torch.cat(x2, dim=2)
		self.x_train=torch.stack((x2))
		self.y=cut_df.iloc[:,2].values
		self.y=[string_to_num[i] for i in self.y]
		# self.y=list(filter(lambda temp:temp>=0,self.y))
		self.y_train=torch.tensor(self.y,dtype=torch.float64)
		del x
		del x2
		del cut_df
	def __len__(self):
		return len(self.y_train)
	def __getitem__(self,idx):
		return self.x_train[idx],self.y_train[idx]
	def get_x(self):
		return self.x_train.detach().cpu().numpy()
	def get_y(self):
		return self.y

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
		# if(target.size()==torch.Size([1])):
		# 	target = torch.unsqueeze(target,1)
		loss = criterion(output, target.long())
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
			print("output")
			print(output.size())
			print("target")
			print(target.size())
			# if(target.size()==torch.Size([1])):
			# 	target = torch.unsqueeze(target,1)
			loss = criterion(output, target.long())
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


def plot_learning_curves(train_losses, test_losses, train_accuracies, test_accuracies):
	print("plot_learning_curves")
	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
	plt.plot(np.arange(len(test_losses)), test_losses, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig(os.path.join(PATH_OUTPUT,"losses.png"),pad_inches=0)
	plt.clf() 
	plt.cla() 
	plt.figure()
	plt.plot(np.arange(len(test_accuracies)), train_accuracies, label='Train')
	plt.plot(np.arange(len(test_accuracies)), test_accuracies, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig(os.path.join(PATH_OUTPUT,"accuracies.png"),pad_inches=0)
	print("plot_learning_curves saved")
	pass

def plot_confusion_matrix(results,path):
	print("plot_confusion_matrix")
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
	class_names = list(num_to_string.values())
	# normalized
	confusion_matrix = metrics.confusion_matrix(list(out[0]),list(out[1]),normalize='true')
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_names)
	cm_display.plot()
	plt.savefig(os.path.join(path,"confusion_NORMALIZED.png"),pad_inches=0, dpi=199)
	# non normalized
	confusion_matrix = metrics.confusion_matrix(list(out[0]),list(out[1]),normalize=None)
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_names)
	cm_display.plot()
	plt.savefig(os.path.join(path,"confusion.png"),pad_inches=0, dpi=199)
	print("plot_confusion_matrix saved")
	pass

class MyCNN(nn.Module):
	def __init__(self, num_classes, window_sizes=(3,4,5)):
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

class MyFFNN(nn.Module):
	def __init__(self, num_classes, buffer_size=-1):
		super(MyFFNN, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(300*buffer_size, 64),
			nn.ReLU(),
			nn.Linear(64, num_classes),
			nn.Sigmoid()
		)
	def forward(self, x):
		x = x.view(x.size(0), -1) 
		x = self.layers(x)
		return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
	print("C-C-CUDA ACTIVE!!!")
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

model_used = ""
if(MODEL_CHOICE=="SVM-RBF"):
	model_used = ""
elif(MODEL_CHOICE=="SVM-L"):
	model_used = ""
elif(MODEL_CHOICE=="RF"):
	model_used = ""
elif(MODEL_CHOICE=="FFNN"):
	if(LABEL_CHOICE=="5"):
		model_used = MyFFNN(5,longest_post_len+USE_CF)
	else:
		model_used = MyFFNN(4,longest_post_len+USE_CF)
elif(MODEL_CHOICE=="CNN"):
	if(LABEL_CHOICE=="5"):
		model_used = MyCNN(5)
	else:
		model_used = MyCNN(4)
else:
	print("BAD MODEL NAME")
	exit()

if(MODEL_CHOICE=='CNN' or MODEL_CHOICE=='FFNN'):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model_used.parameters(),lr=0.001)
	model_used.to(device)
	criterion.to(device)
	untrained_model = model_used

def fold_testing(fold=5):
	best_val_acc = 0.0
	train_losses, train_accuracies = [], []
	test_losses, test_accuracies = [], []
	test_size = int(len(df)/fold)
	train_size = len(df)-test_size
	print('test_size')
	print(test_size)
	print('train_size')
	print(train_size)
	precision_avg = 0
	recall_avg = 0
	f_score_avg = 0
	ord_error_avg = 0
	start = 0
	for i in range(fold):
		print("####### FOLD: {} #######".format(i))		
		test_len = int(len(df)/fold)
		if(start+test_len > len(df)):
			test_len = len(df)-start+test_len
		print("Making train_ds")
		train_ds = MyDataset(start,test_len,True,USE_CF)
		print("Making test_ds")
		test_ds = MyDataset(start,test_len,False,USE_CF)
		start = start + test_len
		if(MODEL_CHOICE=='CNN' or MODEL_CHOICE=='FFNN'):
			model_used = untrained_model
			print("train_ds to loader")
			train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
			print("test_ds to loader")
			test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
			for epoch in range(NUM_EPOCHS):
				print("epoch: {}".format(epoch))
				train_loss, train_accuracy = train(model_used, device, train_loader, criterion, optimizer, epoch)
				test_loss, test_accuracy, test_results = evaluate(model_used, device, test_loader, criterion)
				train_losses.append(train_loss)
				test_losses.append(test_loss)
				train_accuracies.append(train_accuracy)
				test_accuracies.append(test_accuracy)
				is_best = test_accuracy >= best_val_acc
				if is_best:
					best_val_acc = test_accuracy
					torch.save(model_used, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)
			plot_learning_curves(train_losses, test_losses, train_accuracies, test_accuracies)
			# best_model_used = torch.load(os.path.join(PATH_OUTPUT, save_file))
			best_model_used = model_used
			test_loss, test_accuracy, test_results = evaluate(best_model_used, device, test_loader, criterion)
			FOLD_SAVE_PATH = os.path.join(PATH_OUTPUT,"FOLD_{}".format(i))
			if not os.path.exists(FOLD_SAVE_PATH):
				os.makedirs(FOLD_SAVE_PATH)
				true_output = [i for (i, j) in test_results]
				pred_output = [j for (i, j) in test_results]
				actual_pred_df = pd.DataFrame({'true_output': true_output,'pred_output': pred_output})
				actual_pred_df.to_csv(os.path.join(FOLD_SAVE_PATH,"actual_pred_df.txt"), sep='\t')
				with open(os.path.join(FOLD_SAVE_PATH,"counts.txt"), 'w') as file:
					file.write(str(dict(Counter(true_output))))
					file.write(str(dict(Counter(pred_output))))
				with open(os.path.join(FOLD_SAVE_PATH,"stats.txt"), "w") as text_file:
					text_file.write("precision recall f_score ord_error\n")
					precision_val = precision(true_output,pred_output)
					recall_val = recall(true_output,pred_output)
					f_score_val = f_score(true_output,pred_output)
					ord_error_val = ord_error(true_output,pred_output)
					text_file.write("{} {} {} {}".format(precision_val,recall_val,f_score_val,ord_error_val))
				plot_confusion_matrix(test_results,FOLD_SAVE_PATH)
			del train_ds
			del test_ds
			del model_used
		elif(MODEL_CHOICE[0:3]=='SVM'):
			X = train_ds.get_x()
			X = X.reshape(len(X),(longest_post_len+USE_CF)*300)
			y = train_ds.get_y()
			if(MODEL_CHOICE=="SVM-L"):
				clf = svm.SVC(decision_function_shape='ovo',kernel='linear')
			elif(MODEL_CHOICE=="SVM-RBF"):
				clf = svm.SVC(decision_function_shape='ovo',kernel='rbf')
			else:
				exit()
			clf.fit(X, y)
			predictions = clf.predict(test_ds.get_x().reshape(len(test_ds.get_x()),(longest_post_len+USE_CF)*300))
			test_results = list(zip(test_ds.get_y(),predictions))
		elif(MODEL_CHOICE=='RF'):
			X = train_ds.get_x()
			X = X.reshape(len(X),(longest_post_len+USE_CF)*300)
			y = train_ds.get_y()
			rf_model = RandomForestClassifier()
			rf_model.fit(X, y)
			predictions = rf_model.predict(test_ds.get_x().reshape(len(test_ds.get_x()),(longest_post_len+USE_CF)*300))
			test_results = list(zip(test_ds.get_y(),predictions))
		else:
			exit()
		true_output = [i for (i, j) in test_results]
		pred_output = [j for (i, j) in test_results]
		precision_avg = precision_avg + precision(true_output,pred_output)
		recall_avg = recall_avg + recall(true_output,pred_output)
		f_score_avg = f_score_avg + f_score(true_output,pred_output)
		ord_error_avg = ord_error_avg + ord_error(true_output,pred_output)
		print("fold: {}".format(fold))
		print("precision: {}".format(precision(true_output,pred_output)))
		print("recall: {}".format(recall(true_output,pred_output)))
		print("f_score: {}".format(f_score(true_output,pred_output)))
		print("ord_error: {}".format(ord_error(true_output,pred_output)))
		actual_pred_df = pd.DataFrame({'true_output': true_output,'pred_output': pred_output})
		actual_pred_df.to_csv(os.path.join(PATH_OUTPUT,"actual_pred_df.txt"), sep='\t')
		with open(os.path.join(PATH_OUTPUT,"counts.txt"), 'w') as file:
			file.write(str(dict(Counter(true_output))))
			file.write(str(dict(Counter(pred_output))))
		plot_confusion_matrix(test_results,PATH_OUTPUT)
	precision_avg = precision_avg / fold
	recall_avg = recall_avg / fold
	f_score_avg = f_score_avg / fold
	ord_error_avg = ord_error_avg / fold
	print("{} {} {} {}".format(precision_avg,recall_avg,f_score_avg,ord_error_avg))
	with open(os.path.join(PATH_OUTPUT,"stats.txt"), "w") as text_file:
		text_file.write("precision recall f_score ord_error\n")
		text_file.write("{} {} {} {}".format(precision_avg,recall_avg,f_score_avg,ord_error_avg))


def FP(actual,predicted):
	out = 0
	for i in range(len(actual)):
		if(predicted[i]>actual[i]):
			out = out + 1
	return float(out)/float(len(actual))

def FN(actual,predicted):
	out = 0
	for i in range(len(actual)):
		if(actual[i]>predicted[i]):
			out = out + 1
	return float(out)/float(len(actual))

def TP(actual,predicted):
	return 1. - FP(actual,predicted)

def TN(actual,predicted):
	return 1. - FN(actual,predicted)

def precision(actual,predicted):
	if(TP(actual,predicted)+FP(actual,predicted)==0):
		print("TP")
		print(TP(actual,predicted))
		print("FP")
		print(FP(actual,predicted))
		return -1
	return TP(actual,predicted) / (TP(actual,predicted)+FP(actual,predicted))

def recall(actual,predicted):
	if(TP(actual,predicted)+FN(actual,predicted)==0):
		print("TP")
		print(TP(actual,predicted))
		print("FN")
		print(FN(actual,predicted))
		return -1
	return TP(actual,predicted) / (TP(actual,predicted)+FN(actual,predicted))

def f_score(actual,predicted):
	return (2 * TP(actual,predicted)) / ((2 * TP(actual,predicted)) + FP(actual,predicted)+FN(actual,predicted))

def ord_error(actual,predicted):
	out = 0
	for i in range(len(actual)):
		if( abs(actual[i]-predicted[i])>1 ):
			out = out + 1
	return float(out)/float(len(actual))

fold_testing()
