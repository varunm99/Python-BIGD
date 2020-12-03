import numpy as np
import time
'''
Generates points from Gaussian Distribution to feed centers for block-difference
calculation. 

Returns x_i (indices of first point)  and y_i (indiceds of second point)
x_i, y_i are 2 x numPoints
'''
def computeCoordinates(blkRadii, numPoints):
    blkSize=2*blkRadii + 1
    x_i = np.full((2,numPoints), np.inf)
    y_i = np.full((2,numPoints), np.inf)
    while(len(np.where(x_i == np.inf)[0]) != 0):
        x_inds = np.where(x_i == np.inf)
        y_inds = np.where(y_i == np.inf)

        pts = np.round(np.random.normal(0, (blkSize**2)/25, size=(np.shape(x_inds))))
        x_i = replace_vecs(x_i, x_inds[1], pts)
        x_i[np.where(x_i > blkRadii)] = blkRadii
        x_i[np.where(x_i < -blkRadii)] = -blkRadii

        pts = np.round(np.random.normal(0, (blkSize**2)/25, size=np.shape(y_inds)))
        y_i = replace_vecs(y_i, y_inds[1], pts)
        y_i[np.where(y_i > blkRadii)] = blkRadii
        y_i[np.where(y_i < -blkRadii)] = -blkRadii

        overlaps = np.where(np.all(np.equal(x_i, y_i), axis=0))
        x_i[:,overlaps] = np.inf
        y_i[:,overlaps] = np.inf
    x_i = x_i.astype(int)
    y_i = y_i.astype(int)
    return (x_i, y_i)

def replace_vecs(mat, ind, mat_replace):
  for i, index in enumerate(ind):
    mat[:,index] = mat_replace[:,i]
  return mat