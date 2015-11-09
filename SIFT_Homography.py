#solves for SIFT det-des, uses FLANN Matcher, solves for Homography
import cv2
import scipy as sp
import numpy as np
import time

start_time = time.time()

print 'SIFT + FlannMatcher + Homography'

img1_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Programming\Test Images\sample_images\im.esri17.jpg'
img2_path = 'C:\Users\EnviSAGE ResLab\Desktop\Accuracy Tests Journ\Img2 warped.jpg'

img1_c = cv2.imread(img1_path)
img2_c = cv2.imread(img2_path)

img1 = cv2.cvtColor(img1_c, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_c, cv2.COLOR_BGR2GRAY)


sift1 = cv2.SIFT
sift2 = cv2.SIFT

# detect keypoints and compute descriptors using SURF
k1, d1  = sift1.detectAndCompute(img1, None)
k2, d2  = sift2.detectAndCompute(img2, None)


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

matches = flann.knnMatch(d1,d2,k=2)

# store all the good matches as per Lowe's ratio test.  
good = []

for m,n in matches:
    if m.distance < 0.1*n.distance:
        good.append(m)
Lratio = m.distance/n.distance
#print "Lowe's Ratio - %d" %(Lratio)
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
#out = cv2.polylines(img2, [np.int32(d)], True,255,3)
warp = cv2.warpPerspective(img2,M,(cols,rows))  #warp Image 2 to Image 1 coordinates


#visualizations
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
view[:h1, :w1, 0] = img1
view[:h2, w1:, 0] = img2
view[:, :, 1] = view[:, :, 0]
view[:, :, 2] = view[:, :, 0]

cols1 = img1.shape[1]
cols2 = img2.shape[1]
 
for mat in good:
# Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = k1[img1_idx].pt
        (x2,y2) = k2[img2_idx].pt
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1s
        cv2.circle(view, (int(x1),int(y1)), 1, (255, 0, 0), 2)   
        cv2.circle(view, (int(x2)+cols1,int(y2)), 1, (255, 0, 0), 2)
        #draw line connecting the matching keypoints
        #cv2.line(view, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    
    #annotate points for source image
N = len(good)
s = np.array(src_pts)
d = np.array(dst_pts)
d.shape = (N,2)
s.shape = (N,2)
labels = ['{0}'.format(i) for i in range(N)]
    
    #showing image using plt:
        #plot image 1 with labels
plt.subplots_adjust(bottom = 0)
for label, x, y in zip(labels, s[:, 0], s[:, 1]):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5))#,
            #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        #plot image 2 with labels
plt.subplots_adjust(bottom = 0)
for label, x, y in zip(labels, (d[:, 0] +cols1), d[:, 1]):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5))#,
            #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))  
        
    
#Accuracy drafts: RMSE = (((xr-xi)^2 + (yr-yi)^2))^0.5

dx = s[:,0] - d[:,0] #X Residual
dy = s[:,1] - d[:,1] #Y Residual

sqx = np.square([dx])
sqy = np.square([dy])

sqs = sqx + sqy

sqr = np.sqrt([sqs])

print len(dx)
print len(dy)
print sqx
print sqy
print sqr

print 'SURF + FlannMatcher + Homography'
print 'Minimum match count:', MIN_MATCH_COUNT
print 'Keypoints in image1: %d, image2: %d' % (len(k1), len(k2))    
print 'Matches:', len(matches)
print 'Good matches:', len(good)
print("---%s seconds---"% (time.time() - start_time))

#cv2.imshow("Matches", view)
#cv2.imshow("Img2 warped", warp)
#cv2.imshow("out", out)
#cv2.imshow("Homography", view)
#cv2.waitKey(1000)
#plt.imshow(view)
#plt.show (10)

