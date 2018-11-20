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
from sklearn.linear_model import LogisticRegression

BATCH_LEN = 100
NUM_STEPS = 500
PRINT_STEP = 1
LEARNING_RATE = 0.01
ALPHA = 0.5
WEIGHT_DECAY = 0.0001

train_pkl = 'orl_train.pkl'
val_pkl = 'orl_val.pkl'
test_pkl = 'orl_test.pkl'
train_data = pkl.load(open(train_pkl, "rb"))
val_data = pkl.load(open(val_pkl, "rb"))
test_data = pkl.load(open(test_pkl, "rb"))

augment = True

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def get_im_data(path):
    pil_obj = Image.open(path)
    if augment:
        pil_obj = pil_obj.rotate(random.randint(-16,16))
        if random.randint(0,1) == 0:
            pil_obj = pil_obj.transpose(Image.FLIP_LEFT_RIGHT)
    r = torchvision.transforms.ToTensor()(pil_obj)
    pil_obj.close()
    return r

def train_lr(dict_files, len_batch):
    dict_keys = list(dict_files.keys())
    X = []
    Y = []
    for i in range(len_batch/2):
        key = random.choice(dict_keys)
        i1, i2 = random.sample(dict_files[key], 2)
        X.append((get_im_data(i1), get_im_data(i2)))

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
    return anchor.cuda(), posis.cuda(), negis.cuda()

###### distFunc ######
def distFunc(a ,b): # 1st dimension is num of samples
    d = torch.from_numpy(np.zeros((a.shape[0]))).float()
    for i in range(a.shape[0]):
        d[i] = F.mse_loss(a[i],b[i])
    return d


class SiameseNet(nn.Module):  
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 4)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(384, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm1d(384)
        self.bn4 = nn.BatchNorm1d(25)
        self.bn5 = nn.BatchNorm1d(25)
        self.dp = nn.Dropout(0.0)
    def forward_once(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 4)
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 3)
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, num_flat_features(x))
        x = self.bn3(x)
        x = self.dp(F.relu(self.fc1(x)))
        x = self.bn4(x)        
        x = self.dp(F.relu(self.fc2(x)))
        x = self.bn5(x)        
        x = self.dp(self.fc3(x))
        xn = torch.norm(x, p=2, dim=1)
        x = x.div(xn.view([-1,1]))
        return x
    def forward(self, x1, x2):
        dist = distFunc(self.forward_once(x1), self.forward_once(x2))
        if max(dist) > 4.0:
            import pdb; pdb.set_trace()
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
trainAccCurve = [] ## for graph
valAccCurve = [] ## for graph
def train(model, train_dict, steps = NUM_STEPS, print_step = PRINT_STEP, lr = LEARNING_RATE,alpha = ALPHA ,batch_size = BATCH_LEN):
    print("START")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = WEIGHT_DECAY)
    best_val_accuracy = 0.0
    for step in range(steps):
        optimizer.zero_grad()

        anchor,posis,negis = gen_TripletBatch(train_dict, batch_size)

        pos_dists = model(anchor, posis)
        neg_dists = model(anchor, negis)

        loss = (F.relu(pos_dists - neg_dists + alpha)).mean()
        # loss = (torch.log(pos_dists - neg_dists + 5.0)).mean()

        loss.backward()
        optimizer.step()

        augment = False
        model.eval()
        if step%print_step==0:
            print("Step {} :\tTrain loss {}".format(step, loss))
        if step%10*print_step==0:
            classification_model = train_logistic(model, train_data, steps = 2)
            trainAcc = Calc_accuracy(model, train_data, classification_model, steps = 2)
            valAcc = Calc_accuracy(model, val_data, classification_model, steps = 10)
            if valAcc > best_val_accuracy:
                best_val_accuracy = valAcc
                torch.save(model, 'best_model_valbased')
            trainAccCurve.append(trainAcc)
            valAccCurve.append(valAcc)
            print("TrainAcc {} valAcc {}".format(trainAcc, valAcc))

        lossCurve.append(loss)
        model.train()
        augment = True


def train_logistic(model,train_dict, batch_size = 100, steps = 100):
    # X = np.zeros((2*batch_size*steps,1))
    # Y = np.zeros((2*batch_size*steps))
    X = []
    Y = []
    for step in range(steps):
        anchor,posis,negis = gen_TripletBatch(train_dict, batch_size)
        # pos_dists = model.forward(anchor, posis).cpu().detach().numpy()
        # neg_dists = model(anchor, negis).detach().numpy()
        # X[step*2*batch_size:step*2*batch_size+batch_size,0] = pos_dists
        # X[step*2*batch_size+batch_size:step*2*batch_size+2*batch_size,0] = neg_dists
        # Y[step*2*batch_size:step*2*batch_size+batch_size] = 1
        # Y[step*2*batch_size+batch_size:step*2*batch_size+2*batch_size] = 0
        anchor_rep = model.forward_once(anchor).cpu().detach().numpy()
        posis_rep = model.forward_once(posis).cpu().detach().numpy()
        negis_rep = model.forward_once(negis).cpu().detach().numpy()
        X.append(np.abs((anchor_rep - posis_rep))**2)
        X.append(np.abs((anchor_rep - negis_rep))**2)
        Y.append([1]*batch_size)
        Y.append([-1]*batch_size)
    X = np.array(X).reshape([-1,X[0].shape[1]])
    Y = np.array(Y).reshape([-1])
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X,Y)
    
    return clf


def Calc_accuracy(model, test_dict, classification_model, batch_size = 100, steps = 10):
    # X = np.zeros((2*batch_size*steps,1))
    # Y = np.zeros((2*batch_size*steps))
    X = []
    Y = []
    for step in range(steps):
        anchor,posis,negis = gen_TripletBatch(test_dict, batch_size)
        # pos_dists = model(anchor, posis).detach().numpy()
        # neg_dists = model(anchor, negis).detach().numpy()
        # X[step*2*batch_size:step*2*batch_size+batch_size,0] = pos_dists
        # X[step*2*batch_size+batch_size:step*2*batch_size+2*batch_size,0] = neg_dists
        # Y[step*2*batch_size:step*2*batch_size+batch_size] = 1
        # Y[step*2*batch_size+batch_size:step*2*batch_size+2*batch_size] = 0
        anchor_rep = model.forward_once(anchor).cpu().detach().numpy()
        posis_rep = model.forward_once(posis).cpu().detach().numpy()
        negis_rep = model.forward_once(negis).cpu().detach().numpy()
        X.append(np.abs((anchor_rep - posis_rep))**2)
        X.append(np.abs((anchor_rep - negis_rep))**2)
        Y.append([1]*batch_size)
        Y.append([-1]*batch_size)
      
    X = np.array(X).reshape([-1,X[0].shape[1]])
    Y = np.array(Y).reshape([-1])
    return (classification_model.score(X, Y))
    
    # print('accuracy pos/neg '+str(pos_count/(test_size*steps))+'\t'+str(neg_count/(test_size*steps)))
    # return pos_count/(test_size*steps), neg_count/(test_size*steps)

model = SiameseNet().cuda()

train(model, train_data)
plt.clf()
plt.plot(np.linspace(1, len(lossCurve), len(lossCurve)), lossCurve)
plt.savefig('loss_vs_step.svg')
plt.clf()
plt.plot(np.linspace(1, len(trainAccCurve), len(trainAccCurve)), trainAccCurve)
plt.plot(np.linspace(1, len(valAccCurve), len(valAccCurve)), valAccCurve)
plt.savefig('accuracies.svg')
torch.save(model, 'best_model')

model = torch.load('best_model')
classification_model = train_logistic(model, train_data)
augment = False
model.eval()
print(Calc_accuracy(model, train_data, classification_model))
print(Calc_accuracy(model, val_data, classification_model))
print(Calc_accuracy(model, test_data, classification_model))

model = torch.load('best_model_valbased')
classification_model = train_logistic(model, train_data)
augment = False
model.eval()
print(Calc_accuracy(model, train_data, classification_model))
print(Calc_accuracy(model, val_data, classification_model))
print(Calc_accuracy(model, test_data, classification_model))
