#detects and computes descriptors in an image using the SIFT algorithm
import cv2
import numpy as np

qcc =cv2.imread('C:\Users\EnviSAGE ResLab\Desktop\DPAD\Test Images\qcc.jpg')

#convert color images to gray
img1 = cv2.cvtColor(qcc, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp, des = sift.detectAndCompute(img1, None)
img2 = cv2.drawKeypoints (img1,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print len(kp)


cv2.imshow("SIFT Keypoints", img2)
cv2.waitKey(0)
cv2.destroyAllWindows ()