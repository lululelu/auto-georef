#This code extracts the keypoints using SURF detector and descriptor, 
#FLANN (nearest-neighbor) matching algorithm

import cv2
import numpy
import time

start_time = time.time()

img1_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Working Codes\incheon.rising.310.jpg'
img2_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Working Codes\incheon.here.crop.jpg'

img1 = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

# build feature detector and descriptor extractor
hessian_threshold = 400
surf = cv2.SURF(hessian_threshold)
hkp, hdes = surf.detectAndCompute(img1, None)
nkp, ndes = surf.detectAndCompute(img2, None)

# extract vectors of size 64 from raw descriptors numpy arrays
rowsize = len(hdes) / len(hkp)
if rowsize > 1:
    hrows = numpy.array(hdes, dtype = numpy.float32).reshape((-1, rowsize))
    nrows = numpy.array(ndes, dtype = numpy.float32).reshape((-1, rowsize))
    #print hrows.shape, nrows.shape
else:
    hrows = numpy.array(hdes, dtype = numpy.float32)
    nrows = numpy.array(ndes, dtype = numpy.float32)
    rowsize = len(hrows[0])

    
# kNN training - learn mapping from hrow to hkeypoints index
samples = hrows
responses = numpy.arange(len(hkp), dtype = numpy.float32)

#print len(samples), len(responses)
knn = cv2.KNearest()
knn.train(samples,responses)

# retrieve index and value through enumeration
for i, des in enumerate(nrows):
    des = numpy.array(des, dtype = numpy.float32).reshape((1, rowsize))

    #print i, descriptor.shape, samples[0].shape
    retval, results, neigh_resp, dist = knn.find_nearest(des, 1)
    res, dist =  int(results[0][0]), dist[0][0]

    #print res, dist
    if dist < 0.2746517:
        
        # draw matched keypoints in red color
        color = (0, 0, 255)
        
    else:
        # draw unmatched in blue color
        color = (255, 0, 0)
        
    # draw matched key points on query image
    x,y = hkp[res].pt
    center = (int(x),int(y))
    cv2.circle(img2,center,7,color,-1)
    
    # draw matched key points on train image
    x,y = nkp[i].pt
    center = (int(x),int(y))
    cv2.circle(img1,center,7,color,-1)

print("---%s seconds---"% (time.time() - start_time))

tkp = cv2.drawKeypoints (img1, nkp, None, (255,0,0),4)
qkp = cv2.drawKeypoints (img2, hkp, None, (255,0,0),4)
cv2.imshow ("Img1 with kp", tkp)
cv2.imshow ("Img2 with kp", qkp) 
cv2.imshow('Img1',img1)
cv2.imshow('Img2',img2)
cv2.waitKey(1000)



