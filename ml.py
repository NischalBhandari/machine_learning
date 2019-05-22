from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names=['T-shirt/top','Trouser','Pullover', 'Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

#normalize the data
train_images = train_images /255.0
test_images = test_images /255.0

#show the figures 
# plt.figure(figsize=(10,10))
# for i in range(25):
# 	plt.subplot(5,5,i+1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	plt.imshow(train_images[i], cmap=plt.cm.binary)
# 	plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10,activation=tf.nn.softmax)

	])

#compiles the models with the related parameters
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

def testepochs(num):
	text_file=open("Output.txt","a")
	saver="clothers_classifier"+str(num)
#starts training the data 
	model.fit(train_images,train_labels,epochs=num)

#give the loss and accuracy of the model
	test_loss,test_acc=model.evaluate(test_images,test_labels)
	text_file.write("models num is :%s \n" %saver)
	text_file.write("Test Accuracy: %.4f \n" %test_acc)

	text_file.write("Test Accuracy: %.17f \n" %test_loss)
	text_file.close()
	print('Test Accuracy:', test_acc)
	print('Test loss:', test_loss)



	model.save(saver)
for i in range(1,10):
	testepochs(i)	
