import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
from PIL import Image
from tabulate import tabulate as tab
import time


fn = r'C:\Users\EnviSAGE ResLab\Desktop\DPAD\Xy to Latlong\Butuan.Slave.02.tiff'    # open slave image
ds2 = gdal.Open (fn, gdal.GA_Update)                                                # update slave image
sr = osr.SpatialReference()
sr.SetWellKnownGeogCS('EPSG') 
# Access global coordinates of src_pts and assign to dst_pts, # no. of GCPS must be 3 or above
# add gcps using the  georeferenced master image - global coord paired to source points
# This is the same as attaching the GCPs to the raster image
# needs automation of GCP values input
gcps = [
gdal.GCP(125.53526306, 8.95834255, 0 ,146.19915771, 116.02167511),
gdal.GCP(125.58892822, 8.96913815, 0, 304.72198486, 84.87580109), 
gdal.GCP(125.49948883, 8.96440601, 0,42.23839188, 97.51493835 ), 
gdal.GCP( 125.52687073, 8.93583393, 0, 120.81288147, 180.48278809 ), 
gdal.GCP( 125.53765869, 8.91849518, 0,151.1084137, 232.2142334  ),
gdal.GCP( 125.53739166, 8.96804428, 0,152.79367065, 86.9480896),
gdal.GCP( 125.54602814, 8.96160507, 0, 177.540802, 106.82711029),
gdal.GCP(  125.49951935, 8.94194889, 0, 40.47888184, 162.66494751 ),
gdal.GCP( 125.54803467, 8.93322849, 0, 183.19909668,190.85552979)
]

ds2.SetGCPs(gcps, sr.ExportToWkt())
#ds2= None
ds2.SetProjection(sr.ExportToWkt()) 
ds2.SetGeoTransform(gdal.GCPsToGeoTransform(gcps)) 

#get global coordinates of slave image

#unravel GDAL affine transform parameters
img3_path = "C:\Users\EnviSAGE ResLab\Desktop\DPAD\Xy to Latlong\Butuan.Slave.02.tiff"
ds2 = gdal.Open(img3_path)
C,A , B,F,D2 ,E = ds2.GetGeoTransform()

print C,A,B,F,D2,E

