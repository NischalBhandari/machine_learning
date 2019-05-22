import cv2
import numpy as np
import matplotlib.pyplot as plt

img1=cv2.imread('face.jpg')
img2=cv2.imread('logo.png')

rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]


img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
print("return value is ",ret)
print("mask value is ",mask)
mask_inv = cv2.bitwise_not(mask)
print(mask_inv)

#now black out the area of logo in roi
#img1_bg=cv2.bitwise_and(input,output,mask)
img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
plt.imshow(mask)
plt.show()

#take out only region of logo from logo image
img2_fg = cv2.bitwise_and(img2,img2,mask=mask)

dst=cv2.add(img1_bg,img2_fg)
img1[0:rows,0:cols]=dst
cv2.imshow('img1bg',img1_bg)
cv2.imshow('img2bg',img2_fg)
cv2.imshow('dst',dst)

cv2.imshow('done',img2gray)
cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()