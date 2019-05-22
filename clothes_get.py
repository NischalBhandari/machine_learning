import cv2
import numpy as np 
import matplotlib.pyplot as plt
img1=cv2.imread('tshirt.jpg')

img2gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
print(img2gray.shape)
plt.imshow(img2gray)
plt.show()
ret,mask=cv2.threshold(img2gray,50,255,cv2.THRESH_BINARY)
th2=cv2.adaptiveThreshold(img2gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('mean',th2)
cv2.imshow('masked',mask)
mask_inv= cv2.bitwise_not(mask)
#img1_bg=cv2.bitwise_not(img1)
img1_bg=cv2.bitwise_and(img1,img1,mask=mask_inv)

img1_bg=cv2.cvtColor(img1_bg,cv2.COLOR_BGR2GRAY)
print(img1_bg.shape)
cv2.imshow('res',img1_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()