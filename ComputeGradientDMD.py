import numpy as np
import dippykit as dip
from scipy.ndimage import sobel

def computeGradientDMD(img, xi, yi, scale, stride, blkRadii):
    pts1 = xi
    pts2 = yi
    numSamp = np.shape(xi)[1]
    sampPerScale = numSamp/scale

    # Constrain points within appropriate bounds
    pts1[np.where(pts1 > blkRadii - scale + 1)] = blkRadii - scale +1
    pts1[np.where(pts1 < -blkRadii)] = -blkRadii
    pts2[np.where(pts2 > blkRadii - scale + 1)] = blkRadii - scale +1
    pts2[np.where(pts2 < -blkRadii)] = -blkRadii

    pts1 = pts1 + blkRadii + 1
    pts2 = pts2 + blkRadii + 1

    blkSize = 2*blkRadii + 1

    r, c = np.shape(img)
    effr = r - blkSize
    effc = c - blkSize

    IGrad_feat = 5  # 5 descriptors in BIGD vector
    Gx = sobel(img, axis=1)
    Gy = sobel(img, axis=0)

    featureimg_total = np.zeros((r,c,IGrad_feat))
    featureimg_total[..., 0] = img
    featureimg_total[..., 1] = Gx
    featureimg_total[..., 2] = np.abs(Gx)
    featureimg_total[..., 3] = Gy
    featureimg_total[..., 4] = np.abs(Gy)

    numFV = int((np.shape(img)[0]/stride - blkRadii) * (np.shape(img)[1]/stride - blkRadii))
    v_new = np.zeros((numFV, numSamp*IGrad_feat)) # Number of feature vectors by number of features per vector

    pts1= pts1.astype(int)
    pts2= pts2.astype(int)

    for j in range(IGrad_feat):
        ftimg = featureimg_total[..., j]
        v = np.zeros((numFV, numSamp)) # Number of feature vectors by number of sampled points in patch

        # Compute integral images
        itimg = np.cumsum(ftimg,0)
        itimg = np.cumsum(itimg,1)
        iimg = np.zeros((np.shape(itimg)[0] + 2, np.shape(itimg)[1] + 2))
        iimg[1:-1,1:-1] = itimg

        # Compute the normalization constant for each block 
        normMat = np.sqrt(dip.convolve2d(np.power(ftimg, 2),np.ones(((blkRadii*2) +1, (blkRadii*2) + 1))))
        normMat = normMat[2*blkRadii:-2*blkRadii,2*blkRadii:-2*blkRadii]
        normMat[np.where(normMat == 0)] = 1e-10

        for i in range(numSamp):
            mbSize = int(np.floor((i + sampPerScale)/sampPerScale))
            # Integral image coordinates for computing the sum of the pixel values
            # of size mbSize wrt pts1
            iiPt1 = iimg[pts1[0,i] + mbSize : pts1[0,i]+effr + mbSize + 1,
                pts1[1,i] + mbSize : pts1[1,i] + effc + mbSize + 1]
            iiPt2 = iimg[pts1[0,i] + mbSize : pts1[0,i] + effr + mbSize + 1, \
                pts1[1,i] : pts1[1,i]+effc + 1]
            iiPt3 = iimg[pts1[0,i] : pts1[0,i] + effr + 1, \
                pts1[1,i]+ mbSize : pts1[1,i]+effc+ mbSize + 1]
            iiPt4 = iimg[pts1[0,i] : pts1[0,i] + effr + 1,\
                pts1[1,i] : pts1[1,i] + effc + 1]

            blockSum1 = iiPt4 + iiPt1 - iiPt2 - iiPt3

            iiPt1 = iimg[pts2[0,i] + mbSize : pts2[0,i]+effr + mbSize + 1, \
                pts2[1,i] + mbSize : pts2[1,i]+effc + mbSize + 1]
            iiPt2 = iimg[pts2[0,i] + mbSize : pts2[0,i] + effr + mbSize + 1, \
                pts2[1,i] : pts2[1,i]+effc + 1]
            iiPt3 = iimg[pts2[0,i] : pts2[0,i] + effr + 1, \
                pts2[1,i]+ mbSize : pts2[1,i]+effc+ mbSize + 1]
            iiPt4 = iimg[pts2[0,i] : pts2[0,i] + effr + 1,\
                pts2[1,i] : pts2[1,i] + effc + 1]
            blockSum2 = iiPt4 + iiPt1 - iiPt2 - iiPt3
            # Average intensities
            blockSum1 = blockSum1/(mbSize*mbSize)
            blockSum2 = blockSum2/(mbSize*mbSize)
            
            #Block difference
            diffImg = ((blockSum1 - blockSum2)) / normMat
            # Samples at appropriate points
            selectedGrid = diffImg[0:-1:stride,0:-1:stride]
            v[:, i] = selectedGrid.flatten()
        v_new[:, j*numSamp:(j+1)*numSamp] = v
    return v_new.T

