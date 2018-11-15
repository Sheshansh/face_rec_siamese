import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pickle as pkl
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BATCH_LEN = 32
NUM_STEPS = 100
PRINT_STEP = 1
LEARNING_RATE = 0.01
ALPHA = 1.0

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
    r = torchvision.transforms.ToTensor()(pil_obj)
    pil_obj.close()
    return r

def gen_TripletBatch(dict_files, len_batch):
    dict_keys = list(dict_files.keys())
    anchor = torch.from_numpy(np.zeros((len_batch, 1, 112, 92))).float()
    posis = torch.from_numpy(np.zeros((len_batch, 1, 112, 92))).float()
    negis = torch.from_numpy(np.zeros((len_batch, 1, 112, 92))).float()
    for i in range(len_batch):
        key1, key2 = random.sample(dict_keys, 2)
        anchor[i] = get_im_data(random.choice(dict_files[key1]))
        posis[i] = get_im_data(random.choice(dict_files[key1]))
        negis[i] =get_im_data(random.choice(dict_files[key2]))
    return anchor, posis, negis

###### distFunc ######
def distFunc(a ,b): # 1st dimension is num of samples
    d = torch.from_numpy(np.zeros((a.shape[0]))).float()
    # d = F.cosine_similarity(a,b)
    for i in range(a.shape[0]):
        d[i] = F.mse_loss(a[i],b[i])
    return d


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
        dist = distFunc(self.forward_once(x1), self.forward_once(x2))
        return dist

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


lossCurve = [] ## for graph
def train(model, train_dict, steps = NUM_STEPS, print_step = PRINT_STEP, lr = LEARNING_RATE,alpha = ALPHA ,batch_size = BATCH_LEN):
    print("START")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        optimizer.zero_grad()

        anchor,posis,negis = gen_TripletBatch(train_dict, batch_size)

        pos_dists = model(anchor, posis)
        neg_dists = model(anchor, negis)

        loss = (F.relu(pos_dists - neg_dists + alpha)).mean()

        loss.backward()
        optimizer.step()

        lossCurve.append(loss)
        if step%print_step==0:
            print("Step {} :\tTrain loss {}".format(step, loss))

        if step%(print_step*20)==0:
            evaluate(model, val_data, name='ValdistPlot_'+str(step))
            evaluate(model, train_dict, name='TraindistPlot_'+str(step))


    evaluate(model, val_data, name='ValdistPlot_'+str(steps))    
    evaluate(model, train_dict, name='TraindistPlot_'+str(steps))

def evaluate(model, test_dict, test_size = 80, threshold = 0.5, name='distPlot'):
    anchor,posis,negis = gen_TripletBatch(test_dict, test_size)
    pos_dists = model(anchor, posis).detach().numpy()
    neg_dists = model(anchor, negis).detach().numpy()

    print(name+' pos_dists mean/std '+str(pos_dists.mean())+'\t'+str(pos_dists.std()))
    print(name+' neg_dists mean/std '+str(neg_dists.mean())+'\t'+str(neg_dists.std()))
    
    plt.clf()
    plt.scatter(pos_dists, [1]*pos_dists.shape[0], s=0.1, alpha=0.2)
    plt.scatter(neg_dists, [0]*neg_dists.shape[0], s=0.1, alpha=0.2)
    plt.savefig('plot/'+name+'.svg')
    # return cum_pos_accuracy/steps, cum_neg_accuracy/steps

def findThreshold(model, eval_dict, eval_size = 100):   # Approximates a optimal threshold for a set
    anchor,posis,negis = gen_TripletBatch(eval_dict, eval_size)
    pos_dists = model(anchor, posis).detach().numpy()
    neg_dists = model(anchor, negis).detach().numpy()
    ## assuming the dist follows a gaussian distribution, find the intersection of the 2 gauss
    m1 = pos_dists.mean()
    std1 =  pos_dists.std()
    m2 = neg_dists.mean()
    std2 = neg_dists.std()

    ## https://stackoverflow.com/questions/22579434/python-finding-the-intersection-point-of-two-gaussian-curves
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    r1,r2 = np.roots([a,b,c])
    thres = r1 if (m1<=r1<=m2) else r2

    print('\nfindThreshold:\tpos_dist.mean = '+str(m1)+'\tneg_dist.mean = '+str(m2))
    print('roots '+str(r1)+'\t'+str(r2)+' threshold : '+str(thres))
    return thres

def Calc_accuracy(model, test_dict, threshold, test_size = 100, steps = 10):
    pos_count = 0
    neg_count = 0
    for step in range(steps):
        anchor,posis,negis = gen_TripletBatch(test_dict, test_size)
        pos_dists = model(anchor, posis).detach().numpy()
        neg_dists = model(anchor, negis).detach().numpy()
        pos_count += np.sum(pos_dists < threshold)
        neg_count += np.sum(neg_dists > threshold)
    
    print('accuracy pos/neg '+str(pos_count/(test_size*steps))+'\t'+str(neg_count/(test_size*steps)))
    # return pos_count/(test_size*steps), neg_count/(test_size*steps)

model = SiameseNet()
# lossCurve = []
# train(model, train_data)
# torch.save(model, 'best_model')

# plt.clf()
# plt.plot(np.linspace(1, len(lossCurve), len(lossCurve)), lossCurve)
# plt.savefig('loss_vs_step.svg')

print(findThreshold(model, val_data))
print(findThreshold(model, val_data))

threshold = findThreshold(model, val_data)

Calc_accuracy(model, test_data, threshold)
Calc_accuracy(model, test_data, threshold)

# model = torch.load('best_model')
# for i in range(10):
#     eval_accuracy(model, val_data, threshold = i/10.0)
# val_accuracy = eval_accuracy(model, val_data)
# print("Accuracy on validation set = {}".format(val_accuracy))