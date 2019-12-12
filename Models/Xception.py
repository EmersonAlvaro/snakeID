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
IMAGE_SIZE = 224
train_data = pathlib.Path('/content/drive/My Drive/MozSnake/Training/')

def Xception(num_class) :
    stride =(2, 2);
    kernel_size=(3, 3)
    pool_size = (2, 2)
    Channel_axis = 3;

    def middle_flow(x) :

        x_temp = x

        for i in range (8):

            x = tf.nn.relu(x)
            x = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = tf.nn.relu(x)
            x = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = tf.nn.relu(x)
            x = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)



            x = layers.add([x, x_temp])
            x_temp = x

        return x

    def exit_flow(x):

        x_temp = x

        x = tf.nn.relu(x)
        x = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)

        x = tf.nn.relu(x)
        x = layers.SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)

        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x_shortcut = layers.Conv2D(1024, kernel_size=(1, 1), strides=(2, 2),   padding='same')(x_temp)
        x = layers.add([x, x_shortcut])


        x = layers.SeparableConv2D(filters=1536, kernel_size=(3, 3), padding='same')(x)
        x = tf.nn.relu(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)

        x = layers.SeparableConv2D(filters=2048, kernel_size=(3, 3), padding='same')(x)
        x = tf.nn.relu(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)

        x = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(num_class, activation=tf.nn.softmax)(x)

        return x

    def entry_flow(img_imput):

        x = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(img_input)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = tf.nn.relu(x)

        x = layers.Conv2D(64, kernel_size=(3, 3),  padding='same')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = tf.nn.relu(x)

        x_temp = x

        for filter in [128, 256, 728]:

            # if filter!=128:
            x = tf.nn.relu(x)
            x = layers.SeparableConv2D(filter, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = tf.nn.relu(x)
            x = layers.SeparableConv2D(filter, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

            x_shortcut = layers.Conv2D(filter, kernel_size=(1, 1), strides=(2, 2), padding='same')(x_temp)

            x = layers.add([x, x_shortcut])
            x_temp = x

        return x

    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = entry_flow(img_input)
    x=middle_flow(x)
    output = exit_flow(x)

    model = models.Model(inputs=img_input, outputs=output, name='Xception')

    return model

Xception = Xception(num_class)

print(Xception.summary())
plot_model(Xception, to_file='model.png', show_shapes=True, show_layer_names=True)



