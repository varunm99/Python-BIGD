import numpy as np
import dippykit as dip
from ComputeGradientDMD import computeGradientDMD

# Encodes VLAD for one image (given as path)

def encodeVlad(kmeans, im_path, xi, yi, scale, stride, blkRadii, numClusters):
    img = dip.im_read(im_path)
    if np.shape(img) != (200,200):
        img = dip.resize(img, (200,200)) # Resizes each image to be the same size
    features = computeGradientDMD(img, xi, yi, scale, stride, blkRadii)
    classes = kmeans.predict(features.T)
    #classes = np.array([kmeans.predict(features[:,i]) for i in range(np.shape(features)[1])])
    numSamp = np.shape(xi)[1]
    descr=np.zeros((len(kmeans.cluster_centers_)*numSamp*5,))
    for i in range(numClusters):
        center = kmeans.cluster_centers_[i]
        center = np.reshape(center,(np.shape(center)[0],1))
        inds = np.where(classes == i)[0]
        centermat = np.repeat(center, np.shape(inds)[0], axis=1)
        descr[i*numSamp*5:(i+1)*numSamp*5] = np.sum(centermat - features[:,inds], axis=1)
    
    return descr