'''This program detects keypoints and computes the descriptors
of both a train image and a query image and matches these KPs
with K nearest neighbors'''

import cv2
import numpy

train = cv2.imread('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Test Images\qcc.jpg')
query = cv2.imread('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Test Images\up.jpg')

#convert color images to gray
ngrey = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
hgrey = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

# build feature detector and descriptor extractor
hessian_threshold = 15000
surf = cv2.SURF(hessian_threshold)
hkp, hdes = surf.detectAndCompute(hgrey, None)
nkp, ndes = surf.detectAndCompute(ngrey, None)

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
    if dist < 0.01:
        
        # draw matched keypoints in red color
        color = (0, 0, 255)
        
    else:
        # draw unmatched in blue color
        color = (255, 0, 0)
        
    # draw matched key points on query image
    x,y = hkp[res].pt
    center = (int(x),int(y))
    cv2.circle(query,center,7,color,-1)
    
    # draw matched key points on train image
    x,y = nkp[i].pt
    center = (int(x),int(y))
    cv2.circle(train,center,7,color,-1)

tkp = cv2.drawKeypoints (train, nkp, None, (255,0,0),4)
qkp = cv2.drawKeypoints (query, hkp, None, (255,0,0),4)
cv2.imshow ("Train with kp", tkp)
cv2.imshow ("Query with kp", qkp) 
cv2.imshow('query',query)
cv2.imshow('train',train)
cv2.waitKey(0)
cv2.destroyAllWindows()


