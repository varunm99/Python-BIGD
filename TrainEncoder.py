import numpy as np
import dippykit as dip
from ComputeGradientDMD import computeGradientDMD
from sklearn.cluster import KMeans

# Trains encoder from a list of images (provided as array of paths)

def trainEncoder(images, numDescr, numClusters, xi, yi, scale, stride, blkRadii):
    numImages = len(images)
    numDescrPerImg = int(np.ceil(numDescr/numImages))

    descrs = np.zeros((np.shape(xi)[1]*5, numDescrPerImg*numImages))

    for i in range(numImages):
        print("Training on : " + images[i]) # Prints name of image
        im = dip.im_read(images[i])
        if np.shape(im) != (200,200):
            im = dip.resize(im, (200,200)) # Resizes each image to be the same size
        features = computeGradientDMD(im, xi, yi, scale, stride, blkRadii)

        subset = np.random.choice(np.shape(features)[1], size=int(numDescrPerImg), replace=False)
        '''
        if(i == numImages - 1):
            numRemaining = np.shape(descrs[:, i*numDescrPerImg : (i+1)*numDescrPerImg])[1]
            subset = subset[:numRemaining]
        '''
        descrs[:, i*numDescrPerImg : (i+1)*numDescrPerImg] = features[:,subset]

    print("Finished descriptors")
    # Kmeans learning to find cluster centers
    kmeans = KMeans(n_clusters=numClusters, random_state=0,max_iter=1, algorithm='full').fit(descrs.T)

    # Returns Kmeans object containing clusters and assignments
    return kmeans