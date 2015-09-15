# matching features of two images using a combination of surf detector-descriptor, and brute force matcher
import cv2
import scipy as sp
import numpy as np

img1_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Test Images\qcc.jpg'
img2_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Test Images\up.jpg'

img1 = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

surf = cv2.SURF ()   #keypoint detector and descriptor
bf = cv2.BFMatcher()   # keypoint matcher

# detect keypoints
k1, d1  = surf.detectAndCompute(img1, None)
k2, d2  = surf.detectAndCompute(img2, None)

print '#keypoints in image1: %d, image2: %d' % (len(k1), len(k2))
print '#keypoints in image1: %d, image2: %d' % (len(d1), len(d2))


# match the keypoints
matches = bf.match(d1, d2)

# visualize the matches
print '#matches:', len(matches)
dist = [m.distance for m in matches]

print 'distance: min: %.3f' % min(dist)
print 'distance: mean: %.3f' % (sum(dist) / len(dist))
print 'distance: max: %.3f' % max(dist)

# threshold: half the mean
thres_dist =(sum(dist) / len(dist)) * 0.15

# keep only the reasonable matches
sel_matches = [m for m in matches if m.distance < thres_dist]

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

for m in sel_matches:
    # draw the keypoints
    # print m.queryIdx, m.trainIdx, m.distance
    color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
    cv2.line(view, (int(k1[m.queryIdx].pt[0]), int(k1[m.queryIdx].pt[1])) , (int(k2[m.trainIdx].pt[0] + w1), int(k2[m.trainIdx].pt[1])), color)

cv2.imshow("SURF", view)
cv2.waitKey(0)
cv2.destroyAllWindows ()
