import os
import random
import pickle as pkl

ORL_DIR = '/home/sheshansh/project/orl_faces/'
subjects = [dI for dI in os.listdir(ORL_DIR) if os.path.isdir(os.path.join(ORL_DIR,dI))]

def pkl_dump(subject_list, mode):
	"""
	subject list is the subset of subdirs we have to deal with
	mode is train/val/test
	"""
	files = {}
	for subject in subject_list:
		files[subject] = [ORL_DIR+subject+'/'+file for file in os.listdir(ORL_DIR+subject) if file.endswith('.pgm')]	
	with open("orl_{}.pkl".format(mode), "wb") as f:
		pkl.dump(files, f)

num_subjects = len(subjects)
train_subjects = random.sample(subjects, int(0.8*num_subjects))
val_subjects = random.sample(set(subjects)-set(train_subjects), int(0.1*num_subjects))
test_subjects = list(set(subjects)-set(train_subjects)-set(val_subjects))

pkl_dump(train_subjects, 'train')
pkl_dump(val_subjects, 'val')
pkl_dump(test_subjects, 'test')
