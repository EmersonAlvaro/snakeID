from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pathlib
import os

import cv2 as cv

from tensorflow.keras import layers
from tensorflow.keras import models
# from tensorflow.keras.utils import np_utils
from sklearn import metrics
from tensorflow.keras.utils import plot_model


num_class = 5
IMAGE_SIZE = 224
train_data = pathlib.Path('/content/drive/My Drive/MozSnake/Training/')

def VGG19(num_class) :
    stride =(2, 2);
    kernel_size=(3, 3)
    pool_size = (2, 2)

    Channel_axis = 3;

    def comom_bloxk(x):
        # x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = tf.nn.relu(x)

        return x

    def desnblock (x):
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(img_input)
        x = comom_bloxk(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = comom_bloxk(x)

        return x;

    def transtionblock(x):
        x = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(img_input)
        x = comom_bloxk(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = comom_bloxk(x)

        return x;

    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # stage1
    x = layers.Conv2D(filters=64, kernel_size=kernel_size, padding='same')(img_input)
    x = comom_bloxk(x)
    x = layers.Conv2D(filters=64, kernel_size=kernel_size,  padding='same')(x)
    x = comom_bloxk(x)
    x = layers.MaxPool2D(pool_size=pool_size, strides=stride)(x)

    # stage2
    x = layers.Conv2D(filters=128, kernel_size=kernel_size,  padding='same')(x)
    x = comom_bloxk(x)
    x = layers.Conv2D(filters=128, kernel_size=kernel_size,  padding='same')(x)
    x = comom_bloxk(x)
    x = layers.MaxPool2D(pool_size=pool_size, strides=stride)(x)

    # stage3
    x = layers.Conv2D(filters=256, kernel_size=kernel_size, padding='same')(x)
    x = comom_bloxk(x)
    x = layers.Conv2D(filters=256, kernel_size=kernel_size,  padding='same')(x)
    x = comom_bloxk(x)
    x = layers.Conv2D(filters=256, kernel_size=kernel_size,  padding='same')(x)
    x = comom_bloxk(x)
    x = layers.MaxPool2D(pool_size=pool_size, strides=stride)(x)

    # stage4
    x = layers.Conv2D(filters=512, kernel_size=kernel_size, padding='same')(x)
    x = comom_bloxk(x)
    x = layers.Conv2D(filters=521, kernel_size=kernel_size, padding='same')(x)
    x = comom_bloxk(x)
    x = layers.Conv2D(filters=512, kernel_size=kernel_size, padding='same')(x)
    x = comom_bloxk(x)
    x = layers.MaxPool2D(pool_size=pool_size, strides=stride)(x)

    # stage5
    x = layers.Conv2D(filters=512, kernel_size=(3, 3),  padding='same')(x)
    x = comom_bloxk(x)
    x = layers.Conv2D(filters=521, kernel_size=(3, 3),  padding='same')(x)
    x = comom_bloxk(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3),  padding='same')(x)
    x = comom_bloxk(x)
    x = layers.MaxPool2D(pool_size=pool_size, strides=stride)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation=tf.nn.softmax)(x)
    x = layers.Dense(4096, activation=tf.nn.softmax)(x)

    output = layers.Dense(num_class, activation=tf.nn.softmax)(x)

    model = models.Model(inputs=img_input, outputs=output, name='VGG19')

    return model

VGG19 = VGG19(num_class)

print(VGG19.summary())


