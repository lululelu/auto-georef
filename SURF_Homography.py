#solves for SURF det-des, uses FLANN Matcher, solves for Homography
import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()



MIN_MATCH_COUNT = 10 #set a condition that atleast 10 matches are to be found in the object


first = cv2.imread('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Programming\Working Codes\ht01.jpg') # trainImage
second = cv2.imread('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Programming\Working Codes\ht02.jpg') # queryImage

#convert color images to gray
img1 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
 
# Initiate SURF detector
surf = cv2.SURF()

# detect keypoints and compute descriptors using SURF
k1, des1 = surf.detectAndCompute(img1,None)
k2, des2 = surf.detectAndCompute(img2,None)

#Fast Library for Approximate Nearest Neighbors
FLANN_INDEX_KDTREE = 0

#pass two (2) dictionaries specifying algorithm to be used
#or algorithms like mSIFT, SURF etc. you can pass following:
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

#specifies the number of times the trees in the index should be recursively traversed. 
#Higher values gives better precision,but also takes more time. 
#If you want to change the value, pass search_params = dict(checks=100)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.  
good = []

for m,n in matches:
    if m.distance < 0.2377*n.distance:
        good.append(m)
		
if len(good)>MIN_MATCH_COUNT:    #if enough matches are found, extract locations of matched keypoints
    src_pts = np.float32([ k1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ k2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist() 


else:
	print "not enough matches are found - %d%d" %(len(good), MIN_MATCH_COUNT)
	matchesMask = None
	
rows, cols = img1.shape
pts2 = M
pts1 = np.float32([ [0,0], [0, rows-1], [cols-1, rows-1], [cols-1,0] ]).reshape(-1,1,2) #where the points will be warped
d = cv2.perspectiveTransform(pts1,pts2)

out = cv2.polylines(img2, [np.int32(d)], True,255,3)
warp = cv2.warpPerspective(img2,M,(1200,900))	#warp Image 2 to Image 1 coordinates

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

print 'SURF + FlannMatcher + Homography'
print 'Minimum match count:', MIN_MATCH_COUNT
print 'Keypoints in image1: %d, image2: %d' % (len(k1), len(k2))	
print 'Matches:', len(matches)
print 'Good matches:', len(good)
print("---%s seconds---"% (time.time() - start_time))


cv2.imshow("Original", img1)
cv2.imshow("Img2 warped", warp)
cv2.waitKey(1000)
#cv2.imshow("out", out)
#cv2.imshow("Homography", view)
#cv2.waitKey(1000)
#plt.imshow(view)
#plt.show (0)





    

