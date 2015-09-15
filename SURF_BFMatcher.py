# matching features of two images using a combination of surf detector - descriptor and brute force matcher
import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

img1_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Working Codes\incheon.rising.310.jpg'
img2_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Working Codes\incheon.here.crop.jpg'

img1 = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

surf = cv2.SURF (400)   #keypoint detector and descriptor
bf = cv2.BFMatcher()   # keypoint matcher

# detect keypoints
k1, d1  = surf.detectAndCompute(img1, None)
k2, d2  = surf.detectAndCompute(img2, None)

print '#keypoints in image1: %d, image2: %d' % (len(k1), len(k2))
print '#keypoints in image1: %d, image2: %d' % (len(d1), len(d2))


# match the keypoints
matches = bf.match(d1, d2)

# visualize the matches
dist = [m.distance for m in matches]

# threshold: half the mean
thres_dist =(sum(dist) / len(dist)) * 0.61

# keep only the reasonable matches
sel_matches = [m for m in matches if m.distance < thres_dist]
   #if enough matches are found, extract locations of matched keypoints
src_pts = np.float32([ k1[m.queryIdx].pt for m in sel_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ k2[m.trainIdx].pt for m in sel_matches]).reshape(-1,1,2)


print '#matches:', len(matches)
print 'distance: min: %.3f' % min(dist)
print 'distance: mean: %.3f' % (sum(dist) / len(dist))
print 'distance: max: %.3f' % max(dist)
print '#selected matches:', len(sel_matches)

######################################
# visualization
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
view[:h1, :w1, 0] = img1
view[:h2, w1:, 0] = img2
view[:, :, 1] = view[:, :, 0]
view[:, :, 2] = view[:, :, 0]

#

cols1 = img1.shape[1]
cols2 = img2.shape[1]

# For each pair of points we have between both images
    # draw circles, then connect a line between them
for mat in sel_matches:

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
		#allow values of src_pts and dst_pts to take values from 1-->10 for annotation
		#s = np.range((int(k1[m.queryIdx].pt[0]), int(k1[m.queryIdx].pt[1]))
        #cv2.putText(view, '1', src_pts[0,:], cv2.FONT_HERSHEY_SIMPLEX,1,2)	
	#annotate points for source image
	N = len(sel_matches)
	s = np.array(src_pts)
	d = np.array(dst_pts)
	d.shape = (10,2)
	s.shape = (10,2)
	labels = ['{0}'.format(i) for i in range(N)]
	for label, x, y in zip(labels, s[:, 0], s[:, 1]):
		cv2.putText(view,label, (x,y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (1,0,0)) #Draw the text
	for label, x, y in zip(labels, (d[:, 0] +cols1), d[:, 1]):
		cv2.putText(view,label, (x,y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (1,0,0)) #Draw the text
		
	'''
	plt.subplots_adjust(bottom = 0)
	for label, x, y in zip(labels, s[:, 0], s[:, 1]):
		plt.annotate(
			label, 
			xy = (x, y), xytext = (-10, 10),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			#bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	plt.subplots_adjust(bottom = 0)
	for label, x, y in zip(labels, (d[:, 0] +cols1), d[:, 1]):
		plt.annotate(
			label, 
			xy = (x, y), xytext = (-10, 10),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			#bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))	
              #cv2.line(view, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    '''
	
cv2.imshow("Matched", view)
#plt.imshow(view)
#plt.show (0)
cv2.waitKey(1000)
