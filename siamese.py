import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pickle as pkl
import random
from PIL import Image

BATCH_LEN = 10
NUM_STEPS = 100
PRINT_STEP = 1
LEARNING_RATE = 0.001
train_pkl = 'orl_train.pkl'
val_pkl = 'orl_val.pkl'
test_pkl = 'orl_test.pkl'
train_data = pkl.load(open(train_pkl, "rb"))
val_data = pkl.load(open(val_pkl, "rb"))
test_data = pkl.load(open(test_pkl, "rb"))

def num_flat_features(x):
	size = x.size()[1:]  # all dimensions except the batch dimension
	num_features = 1
	for s in size:
		num_features *= s
	return num_features

def get_im_data(path):
	pil_obj = Image.open(path)
	return torchvision.transforms.ToTensor()(pil_obj)

def gen_batch(dict_files, len_batch):
	dict_keys = dict_files.keys()
	pos_batch, neg_batch = [], []
	for i in range(len_batch):
		key = random.choice(dict_keys)
		im1, im2 = random.sample(dict_files[key],2)
		pos_batch.append([get_im_data(im1), get_im_data(im2)])
		key1, key2 = random.sample(dict_keys, 2)
		im1, im2 = random.choice(dict_files[key1]), random.choice(dict_files[key2])
		neg_batch.append([get_im_data(im1), get_im_data(im2)])
	return pos_batch, neg_batch

class SiameseNet(nn.Module):
	
	def __init__(self):
		super(SiameseNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 5)
		self.conv2 = nn.Conv2d(32, 16, 5)
		self.conv3 = nn.Conv2d(16, 8, 5)
		self.fc1 = nn.Linear(640, 16)
		self.fc2 = nn.Linear(16, 16)

	def forward_once(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = F.max_pool2d(F.relu(self.conv3(x)), 2)
		x = x.view(-1, num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def forward(self, x1, x2):
		return self.forward_once(x1), self.forward_once(x2)

def siamese_loss(model, batch, label = 1):
	"""
		x1 and x2 are the similar pairs (positive dataset)
		x3 and x4 are the dissimilar pairs (negative dataset)
	"""
	x1 = torch.stack([item[0] for item in batch])
	x2 = torch.stack([item[1] for item in batch])
	out1, out2 = model(x1, x2)
	similarity = F.cosine_similarity(out1, out2)
	if label == 1:
		return ((1.0-similarity)/2.0).mean()
	else:
		return ((similarity+1)/2.0).mean()

def accuracy(self, batch, threshold = 0.8, label = 1):
	"""
		x1 and x2 are the similar pairs (positive dataset)
		x3 and x4 are the dissimilar pairs (negative dataset)
	"""
	x1 = torch.stack([item[0] for item in batch])
	x2 = torch.stack([item[1] for item in batch])
	out1, out2 = model(x1, x2)
	similarity = F.cosine_similarity(out1, out2)
	if label == 1:
		return (torch.sign(similarity-threshold).mean()+1)/2.0
	else:
		return (torch.sign(threshold-similarity).mean()+1)/2.0


def train(model, train_dict, steps = NUM_STEPS, print_step = PRINT_STEP, lr = LEARNING_RATE, batch_size = BATCH_LEN):
	print("Hello")
	optimizer = optim.Adam(model.parameters(), lr=lr)
	best_val_acc = float('inf')
	for step in range(steps):
		optimizer.zero_grad()
		pos_batch, neg_batch = gen_batch(train_dict, batch_size)
		loss = siamese_loss(model, pos_batch, label = 1) + siamese_loss(model, neg_batch, label = -1)
		loss.backward()
		optimizer.step()
		if step%print_step==0:
			val_loss_avg = 0
			for i in range(5):
				val_accuracy_pos, val_accuracy_neg = eval_accuracy(model, val_data)
				pos_batch, neg_batch = gen_batch(val_data, batch_size)
				val_loss = siamese_loss(model, pos_batch, label = 1) + siamese_loss(model, neg_batch, label = -1)
				val_loss_avg += val_loss
			if val_loss_avg < best_val_acc:
				best_val_acc = val_loss_avg
				torch.save(model, 'best_model')
			val_loss_avg /= 10.0

			print("Step {} : Train loss {} Val - Loss {} PosAcc {} NegAcc {}".format(step, loss, val_loss, val_accuracy_pos, val_accuracy_neg))

def eval_accuracy(model, test_dict, batch_size = BATCH_LEN, steps = 10, threshold = 0.8):
	cum_pos_accuracy = 0.0
	cum_neg_accuracy = 0.0
	for step in range(steps):
		pos_batch, neg_batch = gen_batch(test_dict, batch_size)
		batch_pos_accuracy = accuracy(model, pos_batch, threshold, label = 1)
		batch_neg_accuracy = accuracy(model, neg_batch, threshold, label = -1)
		cum_pos_accuracy += batch_pos_accuracy
		cum_neg_accuracy += batch_neg_accuracy
	return cum_pos_accuracy/steps, cum_neg_accuracy/steps

model = SiameseNet()
train(model, train_data)
model = torch.load('best_model')
for i in range(10):
	eval_accuracy(model, val_data, threshold = i/10.0)
# val_accuracy = eval_accuracy(model, val_data)
# print("Accuracy on validation set = {}".format(val_accuracy))