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
img1_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Xy to Latlong\Landsat8_Butuan.jpg'
img2_path = 'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Xy to Latlong\Butuan.Slave.02.jpg'

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
	s = np.array(src_pts) #pixel coordinates of master keypoints
	d = np.array(dst_pts) #pixel cordinates of slave keypoints
	s.shape = (N,2)
	d.shape = (N,2)
	
	labels = ['{0}'.format(i) for i in range(N)]
	'''
	for label, x, y in zip(labels, s[:, 0], s[:, 1]):
		cv2.putText(view,label, (x,y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (1,0,0)) #Draw the text
	for label, x, y in zip(labels, (d[:, 0] +cols1), d[:, 1]):
		cv2.putText(view,label, (x,y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (1,0,0)) #Draw the text
	'''	
	

	#showing image using plt:
	plt.subplots_adjust(bottom = 0)
	for label, x, y in zip(labels, s[:, 0], s[:, 1]):
		plt.annotate( 
			label, 
			xy = (x, y), xytext = (-10, 10),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	plt.subplots_adjust(bottom = 0)
	for label, x, y in zip(labels, (d[:, 0] +cols1), d[:, 1]):
		plt.annotate(
			label, 
			xy = (x, y), xytext = (-10, 10),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			bbox = dict(boxstyle = 'round4,pad=0.5', fc = 'cyan', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))	
              #cv2.line(view, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
	

# coordinate transformation stage
# open georeferenced image 
ds = gdal.Open(img1_path)

#unravel GDAL affine transform parameters
c, a , b, f,D ,e = ds.GetGeoTransform()

x_value = s[:,0]   # x-coordinate of s, master keypoints
y_value = s[:,1]   # y-coordinate of s, master keypoints

# Returns global coordintes of master image
xp = a*x_value + b*y_value + a*0.5 + b *0.5 + c   # compute for global cooridnate (longitude)
yp = D*x_value + e*y_value + D*0.5 + e *0.5 + f   # compute for global coordinate (latitude)

# Reshape to (N,1)
xp.shape =(N,1)
yp.shape = (N,1)

# Global coordinate of src_pts of master image
final_coord = np.hstack((xp,yp))    # will give (long,lat) values


# Converting Slave image in jpg to tif
img = Image.open('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Xy to Latlong\Butuan.Slave.02.jpg')
img.save('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Xy to Latlong\Butuan.Slave.02.tiff')


# Transform slave image coordinates to world coordinates
fn = r'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Xy to Latlong\Butuan.Slave.02.tiff'    # open slave image
ds2 = gdal.Open (fn, gdal.GA_Update)                                                # update slave image
sr = osr.SpatialReference()
sr.SetWellKnownGeogCS('EPSG') 

# Access global coordinates of src_pts and assign to dst_pts, # no. of GCPS must be 3 or above
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
gdal.GCP( 125.5 , 8.94195, 0,69.46138763 , 158.69232178 ) 
]


#ds2.SetGCPs(gcps, sr.ExportToWkt())
#ds2= None

#ds2.SetProjection(sr.ExportToWkt()) 
#ds2.SetGeoTransform(gdal.GCPsToGeoTransform(gcps)) 
#get global coordinates of slave image

#unravel GDAL affine transform parameters
img3_path = "C:\Users\EnviSAGE ResLab\Desktop\DPAD\Xy to Latlong\Butuan.Slave.02.tiff"
ds3 = gdal.Open(img3_path)
c,a , b,f,D ,e = ds3.GetGeoTransform()

#def pixel2coord(col,row):
x2_value = d[:,0]
y2_value = d[:,1]


xpd = a*x2_value  + b*y2_value + a*0.5  +b *0.5 + c
ypd = D*x2_value + e*y2_value + D*0.5 +e *0.5 + f

xpd.shape =(N,1)
ypd.shape = (N,1)

final_coord_d = np.hstack((xpd,ypd))

rmse_per_gcp = np.sqrt(np.square(final_coord_d[:,0] - final_coord[:,0])) + np.square(final_coord_d[:,1] - final_coord[:,1])
rpg = np.array (rmse_per_gcp, np.float32)
rpg.shape = (N,1)
#rmse_per_gcp_x.shape = (N,1)
#rmse_per_gcp_y.shape = (N,1)

#total_rmse_per_gcp = np.hstack((rmse_per_gcp_x, rmse_per_gcp_y))
gcp = np.arange(0, N, 1)

table = [[xp,yp]]
headers = ["Lat", "Lon"]
headers2 = ["Master", "Slave"]
headers3 = ["X", "Y"]
headers4 =  ["RMSE per GCP"]
table2 = [gcp[:], xpd[:],ypd[:]]
plt.plot(gcp,rmse_per_gcp, 'go')
plt.axis([0, 13, 0.0, 0.05])
plt.ylabel ("RMSE")
plt.xlabel ("GCP No.")
plt.title("RMSE per GCP (deg)")
plt.grid(True)
plt.plot(figsize=(5,5))

#table = [[rmse_per_gcp_x]]
#print tab(table)
#print img1.shape
#print img1.size
#print img2.size
#print "col:", col
#print "row:", row
#print "Lat:", xp
#print "Lon:", yp

#print "Master Global Coord:", final_coord
#print "Slave Global Coord:", final_coord_d
#print "Master KPs Coord", pixel_coord
#print "Slave KPs Coord:", dst_pts
#print "Image 1 Size:", img1.shape
#print "Image 2 Size:", img2.shape
#print "RMSE per GCP Northing:", rmse_per_gcp_x
#print "RMSE per GCP Easting:", rmse_per_gcp_y
#print "RMSE per GCP:", rmse_per_gcp
#cv2.imshow("Matched", view)
#plt.imshow(view) 
#print tab(final_coord, headers,numalign="right")
print tab(d)
#print tab(final_coord_d, headers,numalign="right")
#print tab(rpg)
print c, a,b,f, D,e
#print C, A,B,F, D2,E
#plt.plot (gcp, rmse_per_gcp_x, "bs")
#plt.plot (gcp, rmse_per_gcp_y, "ro")
#cv2.imshow("Matched", view)
#cv2.waitKey(1000)
#plt.imshow(view)
plt.show (10000)
