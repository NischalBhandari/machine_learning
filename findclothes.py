
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import cv2

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names=['T-shirt/top','Trouser','Pullover', 'Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

#normalize the data
train_images = train_images /255.0
test_images = test_images /255.0

#import the classifier model
new_model=keras.models.load_model('clothes_classifier')

predictions = new_model.predict(test_images)

print(predictions[1])

print(np.argmax(predictions[1]))

print(test_labels[1])

def plot_image(i, predictions_array,true_label,img):
	predictions_array,true_label,img=predictions_array[i],true_label[i],img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap=plt.cm.binary)
	predicted_label=np.argmax(predictions_array)
	if predicted_label==true_label:
		color='blue'
	else:
		color='red'
	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label]),color=color)


def plot_value_array(i,predictions_array,true_label):
	predictions_array,true_label=predictions_array[i],true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot=plt.bar(range(10),predictions_array,color="#777777")
	plt.ylim([0,1])
	predicted_label=np.argmax(predictions_array)
	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

# Plot the first X testpul images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions, test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions, test_labels)
# plt.show()


img=test_images[0]
print(type(test_images))
img = (np.expand_dims(img,0))
print(img.shape)
print(img)
predictions_single=new_model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
np.argmax(predictions_single[0])

plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
plt.show()


#convert image to numpy array 

# try:
# 	filename='clothes.jpg'
# 	img=Image.open(filename)
# 	width,height = img.size
# 	print(width,height)
# 	img=img.resize((28,28),Image.ANTIALIAS)
	

# 	plt.imshow(img)
# 	plt.show()
# 	img=img.convert('1')
# 	img.save('resized_image.jpg')
# except IOError:
# 	print("image not found ")


# img=Image.open('resized_image.jpg')

img1=cv2.imread('clothes.jpg')
img1=cv2.resize(img1,(28,28))


img2gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
print(img2gray.shape)
plt.imshow(img2gray)
plt.show()
ret,mask=cv2.threshold(img2gray,100,255,cv2.THRESH_BINARY)
mask_inv= cv2.bitwise_not(mask)
img1_bg=cv2.bitwise_not(img1)

# img1_bg=cv2.bitwise_and(img1,img1,mask=mask_inv)

img1_bg=cv2.cvtColor(img1_bg,cv2.COLOR_BGR2GRAY)
print(img1_bg.shape)
cv2.imshow('res',img1_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()


np_im=np.array(img1_bg)
print (np_im.shape)
np_im = np_im/255.0

plt.figure()
plt.imshow(np_im)
plt.colorbar()
plt.grid(False)
plt.show()

np_im = (np.expand_dims(np_im,0))
print(np_im.shape)
print(np_im)
predictions_single=new_model.predict(np_im)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(predictions_single[0])
print(np.argmax(predictions_single))
