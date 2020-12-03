import numpy as np
import dippykit as dip
from ComputeCoordinates import computeCoordinates
from EncodeVlad import encodeVlad
from TrainEncoder import trainEncoder

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

import os
import glob

# This script trains the SVM on the encoded data and runs it to evaluate


# Gets label from filename
def computeLabel(img_path):
    name = os.path.split(img_path)[1]
    return int(name[:2]) # First two numbers in file name represent category

dataset_path = './data/kth-tips/KTH_TIPS/'

imgs_all = glob.glob(dataset_path + '**/*.png', recursive=True)
num_imgs = len(imgs_all)
#num_imgs = 200
# Picks a random subset of images for training
# TODO: partition dataset into testing and training sets
unique_inds = np.random.choice(len(imgs_all), size=(num_imgs,), replace=False)
print("Evaluating on " + str(num_imgs) + " images")
#print(unique_inds)

n_fold = 10 # Cross Validation k-fold

#imgs = [imgs_all[int(i)] for i in subset]

# Define settings
blkRadii = 7
numPoints = 20
scale = 4
stride = 2

xi, yi = computeCoordinates(blkRadii, numPoints)

numDescriptors = 50000
numClusters = 128

accs = np.zeros((n_fold,))

for k in range(n_fold):
    test_inds = unique_inds[int(k*num_imgs/n_fold) : int((k+1)*num_imgs/n_fold)] # Contains num_imgs/n_fold images
    print(np.shape(test_inds))
    train_inds = np.concatenate((unique_inds[0:int(k*num_imgs/n_fold)], unique_inds[int((k+1)*num_imgs/n_fold):]))
    print(np.shape(train_inds))
    train_imgs = [imgs_all[int(i)] for i in train_inds]
    test_imgs = [imgs_all[int(i)] for i in test_inds]

    print(len(train_imgs))
    print("Training KMeans")
    kmeans = trainEncoder(train_imgs, numDescriptors, numClusters, xi, yi, scale, stride, blkRadii)

    X = np.zeros((128*numPoints*5, len(train_imgs)))
    #y = np.zeros((len(imgs)))
    y = np.array([computeLabel(train_imgs[i]) for i in range(len(train_imgs))])
    #print(y)

    print("Computing VLAD Encodings for Training")
    for i in range(len(train_imgs)):
        print(train_imgs[i])
        vlad_vec = encodeVlad(kmeans, train_imgs[i], xi, yi, scale, stride, blkRadii, numClusters)
        X[:, i] = vlad_vec
        #y[i] # Need to add labels and then test SVM

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='linear'))
    clf.fit(X.T, y)

    #joblib.dump(clf, filename='SVC.joblib')

    # Uses test images
    XTest = np.zeros((128*numPoints*5, len(test_imgs)))
    yTest = np.array([computeLabel(test_imgs[i]) for i in range(len(test_imgs))])
    print("Computing VLAD Encodings for Testing")
    for i in range(len(test_imgs)):
        print(test_imgs[i])
        vlad_vec = encodeVlad(kmeans, test_imgs[i], xi, yi, scale, stride, blkRadii, numClusters)
        XTest[:, i] = vlad_vec

    res = clf.predict(XTest.T)
    print(res)
    print(yTest)
    accs[k] = np.shape(np.where(res == yTest))[1]/len(test_imgs)
    print(accs[k])

print(accs)
print("Average Accuracy:")
print(np.average(accs))
print("Stdev Acc:")
print(np.std(accs))