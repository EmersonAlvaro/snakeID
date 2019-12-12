from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pathlib
import os

from tensorflow import keras
from snake.config import *
import cv2 as cv
# import keras
from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D, Embedding, LSTM
#
import matplotlib.pyplot as plt

ROOT_PATH = pathlib.Path('TraficSigns')
TRAIN_PATH =pathlib.Path('TraficSigns/Training')
TEST_PATH =pathlib.Path('TraficSigns/Testing')

def load_data(data_directory):
    drid = 0;
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".jpg")]
        for f in file_names:
            img = cv.imread(f)
            # img = cv.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
            # cv.imwrite(f,img)
            images.append(img)
            labels.append(int(drid))

        drid = drid +1;
    return images, labels

images, labels = load_data(str(train_data))


imgs = np.asarray(images)
labs = np.array(labels)
labs = np_utils.to_categorical(labs,13)

# print(imgs.shape)
# print(labs.shape)

images, labels = load_data(str(test_data))
imgs_test = np.asarray(images)
labs_test = np.array(labels)
labs_test = np_utils.to_categorical(labs_test,13)

# print('-------------------------------------')
# print(imgs_test.shape)
# print(labs_test.shape)


# First Model Build

input_shape =(IMAGE_SIZE,IMAGE_SIZE, 3)

model = keras.Sequential ([
    keras.layers.Conv2D(32, (3, 3), input_shape=input_shape, activation=tf.nn.relu),
    # keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    # keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # keras.layers.Dropout(0.25),
    # keras.layers.Flatten(),
    # keras.layers.Dense(64,  activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Flatten(),
    # keras.layers.Dropout(0.25),
    keras.layers.Dense(13, activation=tf.nn.softmax)

])
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# print(model.summary())

model.fit(imgs, labs,
          batch_size=124, epochs=10,
          validation_data=(imgs_test, labs_test),
          verbose=1)

# model.save('my_model.h5')

# input_layer = keras.layers.Dense (64,input_shape=input_shape,activation='relu')

# model.add(LSTM(32, input_shape=input_shape))
# # model.add(input_layer)
#
# # model.add(Conv2D(48, (3, 3), activation="relu"))
# # model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dense(64, activation='relu'))
#
# # model.add(Dropout(0.5))
# model.add(Dense(64,activation='relu'))
# model.add(Flatten())
# model.add(Dense(13, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# # print(model.summary())
#
# model.fit(imgs, labs,
#           batch_size=124, epochs=5, verbose=1)