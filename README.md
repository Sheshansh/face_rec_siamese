This work was done as Course Project for CS725 - Fundamentals of Machine Learning.
## Instructions for running
1. Download the ORL dataset from http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z and extract it.
```bash
wget http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z
tar xvzf att_faces.tar.Z
```
2. Generate pickle files for train, dev and test fold.
```bash
python gen_orl_pkl.py orl_faces/
```
3. Train and evaluate a model.
```bash
python PCA_baseline.py
```
or
```bash
python siameseTripLet.py
```