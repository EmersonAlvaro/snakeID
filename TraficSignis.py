import tensorflow as tf
import numpy as np
import pathlib
import os

from tensorflow import keras
import cv2 as cv

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.applications.vgg16 import VGG16

import matplotlib.pyplot as plt


ROOT_PATH = pathlib.Path('TraficSigns')
TRAIN_PATH =pathlib.Path('TraficSigns/Training')
TEST_PATH =pathlib.Path('TraficSigns/Testing')
IMAGE_SIZE = 124

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            img = cv.imread(f)
            img = cv.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
            cv.imwrite(f,img)
            images.append(img)
            labels.append(int(d))
    return images, labels

images, labels = load_data(str(TRAIN_PATH))

# # print(images[8])
# print(len(labels))
# print(len(images))

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
print(base_model.summary())

imgs = np.asarray(images)
labs = np.array(labels)
labs = np_utils.to_categorical(labs,62)

print(imgs.shape)
print(labs.shape)
