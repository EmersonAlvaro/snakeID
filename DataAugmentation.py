import tensorflow as tf
import numpy as np
import pathlib
import os

from tensorflow import keras
import cv2 as cv
from numpy import expand_dims
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


import matplotlib.pyplot as plt

img = cv.imread('foto.jpg')

data = np.asarray(img)

samples = expand_dims(data, 0)

datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
# prepare iterator
it = datagen.flow(data, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	plt.imshow(image)
# show the figure
plt.show()

