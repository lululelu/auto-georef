# matching features of two images using a combination of surf detector - descriptor and brute force matcher
import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

img1_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Programming\Test Images\TestImages\up.google.jpg'
img2_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Programming\Test Images\TestImages\up.here.jpg'

img1_c = cv2.imread(img1_path)
img2_c = cv2.imread(img2_path)

img1 = cv2.cvtColor(img1_path, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_path, cv2.COLOR_BGR2GRAY)

# Initiate SURF detector
surf = cv2.SURF (10000)   #keypoint detector and descriptor
bf = cv2.BFMatcher()   # keypoint matcher

# detect keypoints and compute descriptors using SURF
k1, d1  = surf.detectAndCompute(img1, None)
k2, d2  = surf.detectAndCompute(img2, None)

# match the keypoints
matches = bf.match(d1, d2)

# visualize the matches
dist = [m.distance for m in matches]

# threshold (variable)
thres_dist =(sum(dist) / len(dist)) 

# keep only the reasonable matches
good = [m for m in matches if m.distance < thres_dist]

#if enough matches are found, extract locations of matched keypoints
src_pts = np.float32([ k1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #src plane : img1 
dst_pts = np.float32([ k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)  #dst plane : img2

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist() 

rows = img1.shape[0]
cols = img1.shape[1]
pts2 = M
pts1 = np.float32([ [0,0], [0, rows-1], [cols-1, rows-1], [cols-1,0] ]).reshape(-1,1,2) #where the points will be warped
d = cv2.perspectiveTransform(pts1,pts2)
out = cv2.polylines(img2, [np.int32(d)], True,255,3)
warp = cv2.warpPerspective(img2,M,(cols,rows))	#warp Image 2 to Image 1 coordinates

#visualizations
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8) #sp.zeros((heigth, width,3),3)
view[:h1, :w1] = img1
view[:h2, w1:] = img2
view[:, :, 1] = view[:, :, 0]
view[:, :, 2] = view[:, :, 0]

cols1 = img1.shape[1]
cols2 = img2.shape[1]

# For each pair of points we have between both images
    # draw circles, then connect a line between them
for mat in good:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = k1[img1_idx].pt
        (x2,y2) = k2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 2
        # colour blue
        # thickness = 1
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
			#bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
		#plot image 2 with labels
	plt.subplots_adjust(bottom = 0)
	for label, x, y in zip(labels, (d[:, 0] +cols1), d[:, 1]):
		plt.annotate(
			label, 
			xy = (x, y), xytext = (-10, 10),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			#bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))	
    
    


print 'SURF + Brute Force Matcher'
print 'Keypoints in image1: %d, image2: %d' % (len(k1), len(k2))	
print 'Matches:', len(matches)
print 'Good matches:', len(good)
print("---%s seconds---"% (time.time() - start_time))	


#cv2.imshow("Original", img1)
#cv2.imshow("Img2 warped", warp)
#cv2.waitKey(1000)
#cv2.imshow("out", out) 
#cv2.imshow("Homography", view)
#cv2.waitKey(1000)
plt.imshow(view)
plt.show (1000)
