"""
Deep Learning Assignment 1
Semi-supervised learning on MNIST data set with Pytorch
"""
from __future__ import print_function
import pickle 
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from sub import subMNIST   # testing the subclass of MNIST dataset
import pickle
from sklearn.cluster import KMeans
import warnings


# Zoom function
def zoom_image(image):
    length, width = image.size()
    xbs = np.copy(image.numpy())
    zoom_rate = random.uniform(0.7,1.3)
    xbs_new = np.zeros((length, width))
    xbs_n = scipy.ndimage.zoom(xbs, zoom_rate)
    if zoom_rate == 1:
        pass
    elif zoom_rate < 1:
        for i in range(int(length*zoom_rate)):
            for j in range(int(width*zoom_rate)):
                xbs_new[i+abs(length/2-width/2*zoom_rate),j+abs(width/2-length/2*zoom_rate)]= xbs_n[i,j]
    else:
        for i in range(28):
            for j in range(28):
                xbs_new[i,j] = xbs_n[i+abs(length/2-width/2*zoom_rate),j+abs(width/2-width/2*zoom_rate)]
    return xbs_new


# Slide function
def slide_image(image):
    height, width = image.size()
    direction = random.randint(1,5)
    # 1: upper-left
    # 2: upper-right
    # 3: lower-left
    # 4: lower-right
    xbs = np.copy(image.numpy())
    h_dis = random.uniform(0,4)
    v_dis = random.uniform(0,4)
    xbs_new = np.zeros((height,width))
    if direction == 1:
        for i in range(height):
            for j in range(width):
                if i <= height-v_dis-1 and j <= width-h_dis-1:
                    xbs_new[i,j] = xbs[i+v_dis,j+h_dis]
    elif direction == 2:
        for i in range(height):
            for j in range(width):
                if i <= height-v_dis-1 and j >= h_dis-1:
                    xbs_new[i,j] = xbs[i+v_dis,j-h_dis]
    elif direction == 3:
        for i in range(height):
            for j in range(width):
                if i >= v_dis-1 and j <= width-h_dis-1:
                    xbs_new[i,j] = xbs[i-v_dis,j+h_dis]
    else:
        for i in range(height):
            for j in range(width):
                if i >= v_dis-1 and j >= h_dis-1:
                    xbs_new[i,j] = xbs[i-v_dis,j-h_dis]
    return xbs_new

# Declare filtering function, outputs are 5*5 patches (kernels) to be fit into first ConV layer of model
def image_patch(images, n=5, stride_col=3, stride_row=3, criterion=-1):    
    a = []

    idx_row = range(0,28,stride_row)
    idx_col = range(0,28,stride_col)
    
    for element in images:
        for i in idx_row:
            if i+n <= 28:
                for j in idx_col:
                    if j+n <= 28:
                        if element[i:i+n,j:j+n].sum() > criterion:
                            a.append(element[i:i+n,j:j+n])
    return np.array(a) 


# CPU only training
def train(epoch, pseudo_train_loss, lr_0):
    k = 0.998
    pi = 0.5
    pf = 0.99
    T = 500
    T1 = 100
    T2 = 600
    alpha_f = 3.00
    pf = 0.99
    model.train()
    lr = lr_0 * (k**epoch) 
    if epoch < 500:
        momentum = (epoch/T)*pf + (1-epoch/T)*pi
    else: 
        momentum = pf
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if epoch < T1:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        pseudo_train_loss.append(loss.data[0])
    else:
        if epoch < T2:
            alpha = (epoch-T1)/(T2-T1)*alpha_f
        else:
            alpha = alpha_f
        for (batch_idx, (data, target)),(unlabel_idx,(unlabel_data, unlabel_target)) in zip(enumerate(train_loader),enumerate(unlabel_loader)):
        
            data, target = Variable(data), Variable(target)
            unlabel_data, unlabel_target = Variable(unlabel_data), Variable(unlabel_target)
            optimizer.zero_grad()
            output = model(data)
            unlabel_output = model(unlabel_data)
            loss = F.nll_loss(output, target) + alpha * F.nll_loss(unlabel_output, unlabel_target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [label: {}/{} ({:.0f}%)]; unlabel: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), unlabel_idx * len(unlabel_data), 
                    len(unlabel_loader.dataset),  100. *unlabel_idx / len(unlabel_loader),
                    loss.data[0]))
        pseudo_train_loss.append(loss.data[0])

        
def test(epoch, valid_loader, unlabel_loader, pseudo_valid_loss, pseudo_valid_acc):
    model.eval()
    test_loss = 0
    correct = 0
    pseudo_labels = np.array([])
    for data, target in valid_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    pseudo_valid_loss.append(test_loss)
    pseudo_valid_acc.append(100. * correct / len(valid_loader.dataset))
    for data, target in unlabel_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        temp = output.data.max(1)[1].numpy().reshape(-1)
        pseudo_labels = np.concatenate((pseudo_labels, temp))
    pseudo_labels = torch.from_numpy(pseudo_labels)
    trainset_unlabel.train_labels = pseudo_labels.long()
    unlabel_loader = torch.utils.data.DataLoader(trainset_unlabel, batch_size=128, shuffle=False)


if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	# Transform setting for data import
	transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

	# Import data
	trainset_imoprt = pickle.load(open("train_labeled.p", "rb"))
	validset_import = pickle.load(open("validation.p", "rb"))
	trainset_unlabel = pickle.load(open("train_unlabeled.p", "rb"))

	## Initialize for data augmentation
	trainset_new = subMNIST(root='./data', train=True, download=True, transform=transform, k=12000)
	trainset_new.train_data = torch.cat([trainset_imoprt.train_data, trainset_imoprt.train_data, trainset_imoprt.train_data, trainset_imoprt.train_data])
	trainset_new.train_labels = torch.cat([trainset_imoprt.train_labels, trainset_imoprt.train_labels, trainset_imoprt.train_labels, trainset_imoprt.train_labels])
	
	# Add initial meaningless labels "-1" to the unlabeled data
	unlabeled_labels = torch.from_numpy(np.repeat(-1, trainset_unlabel.__len__()))
	trainset_unlabel.train_labels = unlabeled_labels

	# Data Augmentation Loops
	angle = np.random.uniform(-30,45,(3000)) 
	for i in range(trainset_imoprt.__len__()): 
		trainset_new.train_data[i+3000] = torch.from_numpy(ndimage.rotate(trainset_imoprt.train_data[i].numpy(),angle[i],reshape=False))
		trainset_new.train_data[i+6000] = torch.from_numpy(zoom_image(trainset_imoprt.train_data[i]))
		trainset_new.train_data[i+9000] = torch.from_numpy(slide_image(trainset_imoprt.train_data[i]))


	train_loader = torch.utils.data.DataLoader(trainset_new, batch_size=32, shuffle=True)	
	unlabel_loader = torch.utils.data.DataLoader(trainset_unlabel, batch_size=128, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)

	train_np = trainset_new.train_data.numpy()
	train_kmeans = image_patch(train_np, criterion=0.01).reshape(-1,25)

	# Run k-means to obtain initial weights
	kmeans = KMeans(n_clusters=10, random_state=0).fit(train_kmeans)

	#Normalize 
	initial_weights_np = kmeans.cluster_centers_
	initial_weights_np = (initial_weights_np - np.min(initial_weights_np))/(np.max(initial_weights_np) - np.min(initial_weights_np))
	initial_weights_tensor = torch.FloatTensor(initial_weights_np).float().resize_(10,1,5,5)

	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
			self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
			self.conv2_drop = nn.Dropout2d(p=0.2)
			self.fc1 = nn.Linear(320, 50)
			self.fc2 = nn.Linear(50, 10)

		def forward(self, x):
			x = F.relu(F.max_pool2d(self.conv1(x), 2))
			x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
			x = x.view(-1, 320)
			x = F.relu(self.fc1(x))
			x = F.dropout(x, training=self.training)
			x = F.relu(self.fc2(x))
			return F.log_softmax(x)

	model = Net()
	model.conv1.weight.data = initial_weights_tensor

	pseudo_train_loss = []
	pseudo_valid_loss = []
	pseudo_valid_acc = []
	for epoch in range(1, 500):
		train(epoch, pseudo_train_loss, lr_0 = 0.01)
		if epoch >= 99:
			test(epoch, valid_loader, unlabel_loader, pseudo_valid_loss, pseudo_valid_acc)

	pickle.dump(model, open("model.p", "wb" ))
	pickle.dump(pseudo_train_loss, open("pseudo_train_loss.p", "wb"))
	pickle.dump(pseudo_valid_loss, open("pseudo_validate_loss.p", "wb"))
	pickle.dump(pseudo_valid_acc, open("pseudo_validate_acc.p", "wb"))    
        
    

    

