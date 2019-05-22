import cv2
import numpy

img1=cv2.imread('face.jpg')
img2=cv2.imread('balls.jpg')
print(img2.shape)
height=img2.shape[0]
width = img2.shape[1]
img1=cv2.resize(img1,(int(width),int(height)))
print(img1.shape)
imgadded=cv2.add(img1,img2)
dst=cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow('added',imgadded)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
