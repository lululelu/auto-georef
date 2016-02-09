# matching features of two images using a combination of surf detector - descriptor and brute force matcher
import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
from PIL import Image
#from tabulate import tablate as tab
import time

start_time = time.time()

# keypoint extraction stage
img1_path = 'H:\_uSAT\Qgis Exercise\Day 2 Exercise 1 Data (copy)\Landsat8_Butuan.jpg'
img2_path = 'H:\Fully Georef (GCP)\Butuan.Slave.jpg'

img1 = cv2.imread(img1_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

surf = cv2.SURF (1000)   #keypoint detector and descriptor
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
thres_dist =(sum(dist) / len(dist)) * 0.65

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
	s = np.array(src_pts)
	d = np.array(dst_pts)
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
ds = gdal.Open(img1_path)

#unravel GDAL affine transform parameters
c, a , b, f,d ,e = ds.GetGeoTransform()

#def pixel2coord(col,row):
x_value = s[:,1]
y_value = s[:,0]

"""Returns global coordintes to pixel center using base-0 raster index"""
xp = a*x_value + b*y_value + a*0.5 + b *0.5 + c
yp = d*x_value + e*y_value + d*0.5 + e *0.5 + f


xp.shape =(N,1)
yp.shape = (N,1)

final_coord = np.hstack((xp,yp))

pixel_coord = np.hstack((src_pts,dst_pts))

#converting jpg to tif
#img = Image.open('H:\Fully Georef (GCP)\Butuan.Slave.jpg')
#img.save('H:\Fully Georef (GCP)\Butuan.Slave.tiff')

#transform slave image coordinates to world coordinates
fn = r'H:\Fully Georef (GCP)\Butuan.Slave.tiff'
ds2 = gdal.Open (fn, gdal.GA_Update)
sr = osr.SpatialReference()
sr.SetWellKnownGeogCS('EPSG') 
gcps = [gdal.GCP(125.53572083, 8.95880413, 0, 175.20127869,112.04125214), 
gdal.GCP(125.52501678, 8.90466595, 0, 333,80), 
gdal.GCP( 125.52970886, 8.99489594, 0, 71,93 ), 
gdal.GCP( 125.55802917, 8.96727371, 0, 149,176),
gdal.GCP( 125.55802917, 8.96727371, 0, 171,95),
gdal.GCP( 125.55802917, 8.96727371, 0, 118,81),
gdal.GCP( 125.55802917, 8.96727371, 0, 180,228),
gdal.GCP( 125.55802917, 8.96727371, 0, 181,82),
gdal.GCP( 125.55802917, 8.96727371, 0, 188,130)]
#ds2.SetGCPs(gcps, sr.ExportToWkt())
#ds2= None
#ds2.SetProjection(sr.ExportToWkt()) 
#ds2.SetGeoTransform(gdal.GCPsToGeoTransform(gcps)) 

#get global coordinates of slave image

#unravel GDAL affine transform parameters
img3_path = "H:\Fully Georef (GCP)\Butuan.Slave01.tiff"
ds3 = gdal.Open(img3_path)
c, a , b, f,d ,e = ds3.GetGeoTransform()

#def pixel2coord(col,row):
x2_value = dst_pts[0,:]
y2_value = dst_pts[1,:]

"""Returns global coordintes to pixel center using base-0 raster index"""
xpd = a*x_value + b*y_value + a*0.5 + b *0.5 + c
ypd = d*x_value + e*y_value + d*0.5 + e *0.5 + f

xpd.shape =(N,1)
ypd.shape = (N,1)

final_coord_d = np.hstack((xpd,ypd))

rmse_per_gcp = np.sqrt(np.square(final_coord_d[:,0] - final_coord[:,0])) + np.square(final_coord_d[:,1] - final_coord[:,1])


#rmse_per_gcp_x.shape = (N,1)
#rmse_per_gcp_y.shape = (N,1)

#total_rmse_per_gcp = np.hstack((rmse_per_gcp_x, rmse_per_gcp_y))
gcp = np.arange(0, N, 1)


plt.plot(gcp,rmse_per_gcp, 'go')
plt.axis([-0.5, 13, -0.01, 0.05])
plt.ylabel ("RMSE")
plt.xlabel ("GCP No.")
plt.title("RMSE per GCP")
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

print "Master Global Coord:", final_coord
print "Slave Global Coord:", final_coord_d
print "Master KPs Coord", pixel_coord
print "Slave KPs Coord:", dst_pts
#print "Image 1 Size:", img1.shape
#print "Image 2 Size:", img2.shape
#print "RMSE per GCP Northing:", rmse_per_gcp_x
#print "RMSE per GCP Easting:", rmse_per_gcp_y
print "RMSE per GCP:", rmse_per_gcp
#cv2.imshow("Matched", view)
#plt.imshow(view) 

#plt.plot (gcp, rmse_per_gcp_x, "bs")
#plt.plot (gcp, rmse_per_gcp_y, "ro")
plt.show (10000)
#cv2.waitKey(1000)


