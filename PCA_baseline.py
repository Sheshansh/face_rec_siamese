import pickle as pkl
import random
import numpy as np
import scipy
from PIL import Image
from sklearn.decomposition import PCA

def get_im_data_as_vec(path):
    pil_obj = Image.open(path)
    return np.array(pil_obj).flatten()

def gen_batch(dict_files, pca, len_batch):
    dict_keys = list(dict_files.keys())
    pos_batch, neg_batch = [0]*len_batch, [0]*len_batch
    for i in range(len_batch):
        key = random.choice(dict_keys)
        im1, im2 = random.sample(dict_files[key],2)
        pos_batch[i] = scipy.spatial.distance.cosine(pca.transform(get_im_data_as_vec(im1).reshape(1,-1)), pca.transform(get_im_data_as_vec(im2).reshape(1,-1)))
        key1, key2 = random.sample(dict_keys, 2)
        im1, im2 = random.choice(dict_files[key1]), random.choice(dict_files[key2])
        neg_batch[i] = scipy.spatial.distance.cosine(pca.transform(get_im_data_as_vec(im1).reshape(1,-1)), pca.transform(get_im_data_as_vec(im2).reshape(1,-1)))
    return pos_batch, neg_batch

train_pkl = 'orl_train.pkl'
val_pkl = 'orl_val.pkl'
test_pkl = 'orl_test.pkl'
train_data = pkl.load(open(train_pkl, "rb"))
val_data = pkl.load(open(val_pkl, "rb"))
test_data = pkl.load(open(test_pkl, "rb"))

key_list = list(train_data.keys())
a = len(key_list)
b = len(train_data[key_list[0]])
num_samples = a*b
num_featues = 10304
X = np.zeros((num_samples,num_featues))
i=0
for key in key_list:
    for file in train_data[key]:
        X[i,:] = get_im_data_as_vec(file)
        #print(image_vec.shape)
        i+=1
print(X.shape)
pca = PCA(n_components=20, svd_solver='full') #TODO Find optimal value of k using validation set.
X_new = pca.fit_transform(X) 
print(X_new.shape)
pos_batch,neg_batch = gen_batch(train_data,pca,100)
print(pos_batch)
print(neg_batch)
