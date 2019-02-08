# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import numpy as np
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

data_transforms = {
	'train': transforms.Compose([
		transforms.Resize((256,256)),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	,
	'test': transforms.Compose([
		transforms.Resize((256,256)),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	]),
}

class Net(nn.Module):

	#Our batch shape for input x is (3, 256, 256)

	def __init__(self, n_classes):
		super(Net, self).__init__()

		#Input channels = 3, output channels = 32
		self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
		self.conv2 = nn.Conv2d(32, 32, kernel_size=9, stride=1, padding=4)
		self.mp1 = nn.MaxPool2d(2)

		self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
		self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
		self.mp2 = nn.MaxPool2d(2)

		self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
		self.mp3 = nn.MaxPool2d(2)

		self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
		self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.mp4 = nn.MaxPool2d(2)

		self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
		self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
		self.mp5 = nn.MaxPool2d(2)

		self.conv11 = nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1)
		self.mp6 = nn.MaxPool2d(2)

		self.fc1 = nn.Linear(768*4*4, 2048)
		self.fc2 = nn.Linear(2048, 2048)
		self.fc3 = nn.Linear(2048, n_classes)

	def forward(self, x):
		# in_size = x.size(0)

		print('x',x)

		#Size changes from (3, 256, 256) to (32, 256, 256)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		#Size changes from (32, 256, 256) to (32, 128, 128)
		x = self.mp1(x)

		#Size changes from (32, 128, 128) to (64, 128, 128)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		#Size changes from (64, 128, 128) to (64, 64, 64)
		x = self.mp2(x)

		#Size changes from (64, 64, 64) to (128, 64, 64)
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		#Size changes from (128, 64, 64) to (128, 32, 32)
		x = self.mp3(x)

		#Size changes from (128, 32, 32) to (256, 32, 32)
		x = F.relu(self.conv7(x))
		x = F.relu(self.conv8(x))
		#Size changes from (256, 32, 32) to (256, 16, 16)
		x = self.mp4(x)

		#Size changes from (256, 16, 16) to (512, 16, 16)
		x = F.relu(self.conv9(x))
		x = F.relu(self.conv10(x))
		#Size changes from (512, 16, 16) to (512, 8, 8)
		x = self.mp5(x)

		#Size changes from (512, 8, 8) to (768, 8, 8)
		x = F.relu(self.conv11(x)) #essa chamda da errado

		#Size changes from (768, 8, 8) to (768, 4, 4)
		x = self.mp6(x)
		#Size changes from (768, 4, 4) to (1,12288)
		x = x.view(-1,768*4*4)  # flatten the tensor

		#Size changes from (1, 12288) to (1, 2048)
		x = F.relu(self.fc1(x))

		x = F.relu(self.fc2(x))

		#Size changes from (1, 2048) to (1, total of classes)
		x = F.softmax(self.fc3(x), dim=1)

		return(x)

def train(epoch, train_loader, batch_size):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)

		# loss = F.nll_loss(output, target)
		criterion = nn.BCELoss()

		zeros = batch_size*[np.zeros(n_classes)]
		zeros = np.asarray(zeros)
		for i, t in enumerate(target):
			zeros[i][t.item()] = 1.0

		# print('zeros --> ',zeros)

		target = Variable(torch.from_numpy(zeros))

		print('output --> ',output)
		print('target --> ',target)
		print('output shape --> ',output.shape)
		print('target shape--> ',target.shape)
		print()

		loss = criterion(output,target.float())

		loss.backward()
		optimizer.step()
		# if batch_idx % 5 == 4:
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
			100. * (batch_idx+1) / len(train_loader), loss.item()))

		print('=================================================================')


def test(epoch, test_loader):
	model.eval()
	model.train(False)

	test_loss = 0
	correct = 0
	for data, target in test_loader:
		
		data, target = Variable(data), Variable(target)
		output = model(data)

		# sum up batch loss
		# test_loss += F.nll_loss(output, target).item()
		criterion = nn.BCELoss()
		
		zeros = 1*[np.zeros(n_classes)]
		zeros = np.asarray(zeros)
		for i, t in enumerate(target):
			zeros[i][t.item()] = 1.0

		target = Variable(torch.from_numpy(zeros))

		print('zeros --> ',zeros)
		print('output --> ',output)
		print('target --> ',target)

		test_loss += criterion(output,target.float())

		# get the index of the max log-probability
		pred = output.data.max(1, keepdim=True)[1]

		print('pred --> ',pred)
		# print('target[pred] --> ',target[pred].float())
		print()

		# correct += pred.eq(target.data.view_as(pred)).cpu().sum()

		for i, t in enumerate(target):
			correct += pred.eq(target[i][pred].long().data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)

	acc = 100. * correct / len(test_loader.dataset) 

	string = 'Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		epoch, test_loss, correct, len(test_loader.dataset), acc)

	print(string)
	print('=================================================================')
	return acc

# Training settings
batch_size = 1
# batch_size_test = 1
n_folds = 3
epochs = 2
data_dir = 'folds'
n_classes = 2
lr = 0.1

model = None
optimizer = None

handle = open('log', 'a')
handle.write('\n··· Informações de desempenho - {} ···\n\n'.format(time.ctime()))

accuracies = []

for fold in range(1, n_folds+1):
	
	test_idx = fold
	
	print('\n\t\tIteracao {} - Fold de teste: fold{}\n'.format(test_idx,test_idx))
	print('=================================================================')


	image_datasets_train = {'fold{}'.format(x): datasets.ImageFolder(os.path.join(data_dir, "fold{}".format(x)), data_transforms['train']) for x in range(1,n_folds+1) if x != test_idx}
	image_datasets_test = {'fold{}'.format(x): datasets.ImageFolder(os.path.join(data_dir, "fold{}".format(x)), data_transforms['test']) for x in range(test_idx,test_idx+1)}

	datasets_images = [image_datasets_train['fold{}'.format(i)] for i in range(1,n_folds+1) if i != test_idx]

	train_data = torch.utils.data.ConcatDataset(datasets_images)
	# Data Loader (Input Pipeline)
	train_loader = torch.utils.data.DataLoader(dataset=train_data,
											   batch_size=batch_size,
											   shuffle=False)
	test_loader = torch.utils.data.DataLoader(dataset=image_datasets_test['fold{}'.format(test_idx)],
											  batch_size= 1,
											  shuffle=False)

	model = Net(n_classes)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

	acc = None
	for epoch in range(1, epochs+1):
		acc = test(epoch, test_loader)
		train(epoch, train_loader, batch_size)
		print('\n-------------------TESTE-----------------------------\n')
	
	accuracies.append(acc)

	string = '\tIteracao {} - Acuracia: {}%\n'.format(fold,acc)
	handle.write(string)
	torch.save(model.state_dict(), 'model{}.pt'.format(fold))

mean = np.mean(accuracies)
std = np.std(accuracies)
string = '\nAcuracia Media: {}%, Desvio Padrao: {}'.format(acc,std)

handle.write(string)

handle.close()
