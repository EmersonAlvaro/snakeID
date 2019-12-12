# from google.colab import drive
# drive.mount('/content/drive/')

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pathlib
import os
import cv2 as cv

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.utils import np_utils
from sklearn import metrics
from tensorflow.python.keras.utils.vis_utils import plot_model

num_class = 5
IMAGE_SIZE = 299
train_data = pathlib.Path('/content/drive/My Drive/MozSnake/Training/')
Channel_axis = 3;


def conv2d(x, filter,  kernel_size,strides=(1, 1), padding='same'):

    x= layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization(axis=Channel_axis)(x)
    x = tf.nn.relu(x)

    return x

def inception_A(x):

    a1 = conv2d(x, 64, kernel_size=(1, 1), padding='same')
    a1 = conv2d(a1, 96, kernel_size=(3, 3), padding='same')
    a1 = conv2d(a1,  96, kernel_size=(3, 3), padding='same')

    a2 = conv2d(x, 64, kernel_size=(1, 1), padding='same')
    a2 = conv2d(a2,96, kernel_size=(3, 3), padding='same')

    a3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    a3 = conv2d(a3, 96, kernel_size=(1, 1), padding='same')

    a4 = conv2d(x, 96, kernel_size=(1, 1), padding='same')

    x = layers.concatenate([a1, a2, a3, a4], axis=-1)

    return x

def inception_B(x):

    b1 = conv2d(x,  192, kernel_size=(1, 1), padding='same')
    b1 = conv2d(b1, 192, kernel_size=(7, 1), padding='same')
    b1 = conv2d(b1, 224, kernel_size=(1, 7), padding='same')
    b1 = conv2d(b1, 224, kernel_size=(7, 1), padding='same')
    b1 = conv2d(b1, 256, kernel_size=(1, 7), padding='same')

    b2 = conv2d(x, 192, kernel_size=(1, 1), padding='same')
    b2 = conv2d(b2, 224, kernel_size=(1, 7), padding='same')
    b2 = conv2d(b2, 256, kernel_size=(7, 1), padding='same')

    b3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    b3 = conv2d(b3, 128, kernel_size=(1, 1), padding='same')

    b4 = conv2d(x, 384, kernel_size=(1, 1), padding='same')

    x = layers.concatenate([b1, b2, b3, b4], axis=-1)

    return x

def inception_C(x):
    c1 = conv2d(x, 384, kernel_size=(1, 1), padding='same')
    c1 = conv2d(c1, 446, kernel_size=(3, 1), padding='same')
    c1 = conv2d(c1, 512, kernel_size=(1, 3), padding='same')
    c1_1 = conv2d(c1, 256, kernel_size=(1, 3), padding='same')
    c1_2 = conv2d(c1, 256, kernel_size=(3, 1), padding='same')

    c2 = conv2d(x, 384,  kernel_size=(1, 1), padding='same')
    c2_1 = conv2d(c2, 256, kernel_size=(1, 3), padding='same')
    c2_2 = conv2d(c2, 256, kernel_size=(3, 1), padding='same')

    c3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    c3 = conv2d(c3, 256, kernel_size=(1, 1), padding='same')

    c4 = conv2d(x, 256, kernel_size=(1, 1), padding='same')

    x = layers.concatenate([c1_1, c1_2, c2_1,c2_2, c3, c4], axis=-1)

    return x


def stem(img_imput):

    x = conv2d(img_imput, 32, kernel_size=(3, 3),strides=(2, 2), padding='same')
    x = conv2d(x, 32, kernel_size=(3, 3), padding='same')
    x = conv2d(x, 64, kernel_size=(3, 3), padding='same')

    x_1 = conv2d(x, 96, kernel_size=(3, 3), strides=(2, 2), padding='same')
    x_2 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = layers.concatenate([x_1, x_2], axis=-1)

    x1 = conv2d(x, 64, kernel_size=(3, 3), padding='same')
    x1 = conv2d(x1, 96, kernel_size=(3, 3), padding='same')

    x2 = conv2d(x, 64, kernel_size=(3, 3), padding='same')
    x2 = conv2d(x2, 64, kernel_size=(3, 3), padding='same')
    x2 = conv2d(x2, 64, kernel_size=(3, 3), padding='same')
    x2 = conv2d(x2, 96, kernel_size=(3, 3), padding='same')

    x = layers.concatenate([x1, x2], axis=-1)

    x_1 = conv2d(x, 192, kernel_size=(3, 3), strides=(2, 2),  padding='same')
    x_2 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = layers.concatenate([x_1, x_2], axis=-1)

    return x

def reduction_A(x):

    x1 = conv2d(x, 192, kernel_size=(1, 1), padding='same')
    x1 = conv2d(x1, 224, kernel_size=(3, 3), padding='same')
    x1 = conv2d(x1, 256, kernel_size=(3, 3), strides=(2, 2), padding='same')


    x2 = conv2d(x, 32, kernel_size=(3, 3),strides=(2, 2), padding='same')

    x3 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = layers.concatenate([x1, x2, x3], axis=-1)

    return x

def reduction_B(x):

    x1 = conv2d(x, 256, kernel_size=(1, 1), padding='same')
    x1 = conv2d(x1, 256, kernel_size=(1, 7), padding='same')
    x1 = conv2d(x1, 320, kernel_size=(7, 1), padding='same')
    x1 = conv2d(x1, 320, kernel_size=(3, 3), strides=(2, 2), padding='same')

    x3 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x2 = conv2d(x, 192, kernel_size=(1, 1), padding='same')
    x2 = conv2d(x2, 192, kernel_size=(3, 3),strides=(2, 2),  padding='same')

    x = layers.concatenate([x1, x2, x3], axis=-1)

    return x

def inceptionV4( num_class):

    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = stem(img_input)

    for i in range(4):
        x= inception_A(x)

    x = reduction_A(x)

    for i in range(7):
        x = inception_B(x)

    x = reduction_B(x)

    for i in range(3):
        x = inception_C(x)

    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(num_class, activation=tf.nn.softmax)(x)

    model = models.Model(inputs=img_input, outputs=output, name='InceptionV4')

    return model

Inception = inceptionV4(num_class)

print(Inception.summary())
plot_model(Inception, to_file='model.png', show_shapes=True, show_layer_names=True)



