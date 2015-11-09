#detects and computes descriptors in an image using the SURF algorithm
import cv2
import numpy as np
from matplotlib import pyplot as plt



img_path = 'C:\Users\EnviSAGE ResLab\Desktop\Accuracy Tests Journ\Rising.Warped.jpg'  #Warped Slave
img_c = cv2.imread(img_path)
img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

surf = cv2.SURF(500)
kp, des = surf.detectAndCompute (img, None)
img2 = cv2.drawKeypoints (img, kp, None, (255,0,0),4)
'''
CPs = []
coords = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
'''
print len(kp)
cv2.imshow("Surf", img2)
cv2.waitKey(1000)
                           