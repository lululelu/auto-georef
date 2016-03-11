
###############################################################################
# Project:  Automatic Georeferencing of DWIATA Imagery
# Purpose:  Script to automatically extract and match feautres between images, 
#           read coordinate system and geotransformation matrix of the master image
#           and report latitude/longitude coordinates of the keypoints, 
#           update the geotransformation of slave images
# Author:   Jerine A. Amado
#           PHL-Microsat DPAD
# References: 
# [1] "Lowe, David G.","Distinctive Image Features from Scale-Invariant Keypoints",
#     "International Journal of Computer Vision", 60(2) 91-110.
# [2] "Bay, Herbert,  Ess, Andreasa, Tuytelaars, Tinne, and Van Gool, Luc"","Speede-Up Robust
#      Features, Computer Vision and Image Understanding"", 110 (2008) 346–359
# [3] "Garrard, Chris", "Manning Early Access Program Geoprocessing with Python 
#      Copyright 2015 Manning Publications", (2015).
# [4] ÖpenCV-Python Tutorials", http://opencv-python-tutroals.readthedocs.org
# 
#
###############################################################################



# matching features of two images using a combination of surf detector - descriptor and brute force matcher
import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
from PIL import Image
from tabulate import tabulate as tab
import time

start_time = time.time()

# keypoint extraction stage
img1_path = 'H:\_uSAT\Qgis Exercise\Day 2 Exercise 1 Data (copy)\Landsat8_Butuan.jpg'
img2_path = 'H:\_uSAT\Fully Georef (GCP)\Butuan.Slave.jpg'

img1 = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

surf = cv2.SURF (900)   #keypoint detector and descriptor
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
thres_dist =(sum(dist) / len(dist)) * 0.63

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
	source = np.array(src_pts)
	dest = np.array(dst_pts)
	source.shape = (N,2)
	dest.shape = (N,2)
	
	labels = ['{0}'.format(i) for i in range(N)]
	'''
	for label, x, y in zip(labels, s[:, 0], s[:, 1]):
		cv2.putText(view,label, (x,y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (1,0,0)) #Draw the text
	for label, x, y in zip(labels, (d[:, 0] +cols1), d[:, 1]):
		cv2.putText(view,label, (x,y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (1,0,0)) #Draw the text
	'''	
	

	#showing image using plt:
	plt.subplots_adjust(bottom = 0)
	for label, x, y in zip(labels, source[:, 0], source[:, 1]):
		plt.annotate( 
			label, 
			xy = (x, y), xytext = (-10, 10),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	plt.subplots_adjust(bottom = 0)
	for label, x, y in zip(labels, (dest[:, 0] +cols1), dest[:, 1]):
		plt.annotate(
			label, 
			xy = (x, y), xytext = (-10, 10),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))	
              #cv2.line(view, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


#plt.imshow(view)
#plt.show (10000)


# this part is still being improved
# coordinate transformation stage
# open master (georeferenced image)

master_georef = gdal.Open (img1_path)

# unravel GDAL affine transformation parameters
# from the master image

c1, a1 , b1, f1,d1 ,e1 = master_georef.GetGeoTransform()

x_value = source[:,0]
y_value = source[:,1]

# Returns global coordintes of master image
xp = a1*x_value + b1*y_value + a1*0.5 + b1*0.5 + c1
yp = d1*x_value + e1*y_value + d1*0.5 + e1*0.5 + f1

# Reshape to (N,1)
xp.shape =(N,1)
yp.shape = (N,1)

# Global coordinate of src_pts of master image
final_coord = np.hstack((xp,yp))

headers = ["Lat", "Lon"]
headers2 = ["Master", "Slave"]
headers3 = ["X", "Y"]
headers4 =  ["RMSE per GCP"]


# print global coordinates of the source points (src_pts)
print tab(final_coord, headers,numalign="right")

# Converting slave image in jpg to tif
img = Image.open('H:\_uSAT\Fully Georef (GCP)\Butuan.Slave.jpg')  # input
img.save('H:\Butuan.Slave.tiff')         # output

# slave image is not yet georeferenced
# its raw coordinates maybe checked using the GDAL affine coefficients

img2_tif = r'H:\Butuan.Slave.tiff'
slave_tif = gdal.Open (img2_tif)
c2, a2 , b2, f2,d2 ,e2 = slave_tif.GetGeoTransform()

print  c2, a2 , b2, f2,d2 ,e2 
# will show 0.0 1.0 0.0 0.0 0.0 1.0, still in pixel coord (offset)

# update the source raw tif global coordinates
slave_update = gdal.Open (img2_tif, gdal.GA_Update)
sr = osr.SpatialReference()     
sr.SetWellKnownGeogCS('WGS84')   # setting the coordinate system (WSG84 with no projection)

# add gcps using the  georeferenced master image - global coord paired to source points
# needs automation of GCP values input
gcps = [
gdal.GCP(125.535 ,  8.95834, 0 ,175.19830322 , 112.04946899),
gdal.GCP(125.589 ,8.96914, 0, 333.76828003,   80.90518951 ), 
gdal.GCP(125.499 , 8.96441, 0,71.12488556  , 93.5379715  ), 
gdal.GCP( 125.527 , 8.93583, 0, 149.82575989 , 176.47032166 ), 
gdal.GCP( 125.534 , 8.96407, 0,171.52664185 ,  95.27961731 ),
gdal.GCP( 125.501 , 8.94016, 0,73.97042847 , 163.30134583 ),
gdal.GCP( 125.538 , 8.9185, 0,180.14434814 , 228.19236755 ),
gdal.GCP( 125.579 , 8.94783, 0,300.4359436 ,  142.44551086 ),
gdal.GCP( 125.537 ,8.96804, 0, 181.77171326 ,  82.96598053 ),
gdal.GCP( 125.54 , 8.95218, 0, 188.70367432 , 130.03863525),
gdal.GCP( 125.546 , 8.96161, 0,206.5252533 ,  102.8332901 ),
gdal.GCP( 125.5 , 8.94195, 0,69.46138763 , 158.69232178 ) ]


slave_update.SetGCPs(gcps, sr.ExportToWkt())
#slave_update = None #close dataset to flush it to disk

print gcps

print slave_update
# source_raw is already updated, and so are the geotransform coefficients
# unravel GDAL affine transform parameters
slave_georef = gdal.Open(r'H:\Butuan.Slave.tiff')
c3, a3 , b3, f3,d3 ,e3 = slave_georef.GetGeoTransform()

# must be updated!
print c3, a3 , b3, f3,d3 ,e3

# from slave image
x2_value = dest[:,0]
y2_value = dest[:,1]

# Returns global coordintes of slave image
xpd = a1*x2_value  + b1*y2_value + a1*0.5  +b1 *0.5 + c1
ypd = d1*x2_value + e1*y2_value + d1*0.5 +e1 *0.5 + f1

# Reshape to (N,1)
xpd.shape =(N,1)
ypd.shape = (N,1)

# Global coordinate of dst_pts of master image
final_coord_d = np.hstack((xpd,ypd))

# solve for the root-mean square error
rmse_per_gcp = np.sqrt(np.square(final_coord_d[:,0] - final_coord[:,0])) + np.square(final_coord_d[:,1] - final_coord[:,1])
rpg = np.array (rmse_per_gcp, np.float32)
rpg.shape = (N,1)

print tab(final_coord_d, headers,numalign="right")
print tab(rpg)

gcp = np.arange(0, N, 1)

plt.plot(gcp,rmse_per_gcp, 'g--')
plt.axis([0, 13, 0.0, 0.015])
plt.ylabel ("RMSE")
plt.xlabel ("GCP No.")
plt.title("RMSE per GCP (deg)")
plt.grid(True)
plt.plot(figsize=(5,5))
plt.show (10000)
