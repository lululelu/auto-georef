import cv2
import scipy as sp
import numpy as np
import time

start_time = time.time()
print 'SURF + FlannMatcher + Homography'


MIN_MATCH_COUNT = 10 #set a condition that atleast 10 matches are to be found in the object
print 'Minimum match count:', MIN_MATCH_COUNT

qcc = cv2.imread('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Test Images\qcc.jpg') # queryImage
up = cv2.imread('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Test Images\up.jpg') # trainImage

#convert color images to gray
img1 = cv2.cvtColor(qcc, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
 
# Initiate SURF detector
surf = cv2.SURF()

# detect keypoints and compute descriptors using SURF
k1, des1 = surf.detectAndCompute(img1,None)
k2, des2 = surf.detectAndCompute(img2,None)

print 'Keypoints in image1: %d, image2: %d' % (len(k1), len(k2))

#Fast Library for Approximate Nearest Neighbors
FLANN_INDEX_KDTREE = 0

#pass two (2) dictionaries specifying algorithm to be used
#or algorithms like SIFT, SURF etc. you can pass following:
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

#specifies the number of times the trees in the index should be recursively traversed. 
#Higher values gives better precision,but also takes more time. 
#If you want to change the value, pass search_params = dict(checks=100)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)
print 'Matches:', len(matches	)
# store all the good matches as per Lowe's ratio test.  
good = []

for m,n in matches:
    if m.distance < 0.2377*n.distance:
        good.append(m)
		
if len(good)>MIN_MATCH_COUNT:    #if enough matches are found, extract locations of matched keypoints
    src_pts = np.float32([ k1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ k2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()



		
		
		
	
print 'Good matches:', len(good)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
view[:h1, :w1, 0] = img1
view[:h2, w1:, 0] = img2
view[:, :, 1] = view[:, :, 0]
view[:, :, 2] = view[:, :, 0]

for m in good:
    # draw the keypoints
    # print m.queryIdx, m.trainIdx, m.distance
    color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
    cv2.line(view, (int(k1[m.queryIdx].pt[0]), int(k1[m.queryIdx].pt[1])) , (int(k2[m.trainIdx].pt[0] + w1), int(k2[m.trainIdx].pt[1])), color)
print("---%s seconds---"% (time.time() - start_time))







    

