#This code detects and computes descriptors in an image using the SURF algorithm

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Working Codes\incheon.rising.310.jpg', 0)
surf = cv2.SURF(400)
kp, des = surf.detectAndCompute (img, None)
img2 = cv2.drawKeypoints (img, kp, None, (255,0,0),4)
print len(kp)
cv2.imshow("Surf", img2)
cv2.waitKey(1000)
                           