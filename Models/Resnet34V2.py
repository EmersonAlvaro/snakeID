# from google.colab import drive
# drive.mount('/content/drive/')

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pathlib
import os

import cv2 as cv
import pydotplus

from tensorflow.python.keras import layers
from tensorflow.keras import models
# from tensorflow.keras.utils import np_utils
from sklearn import metrics
from tensorflow.keras.utils import plot_model
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


num_class = 5
IMAGE_SIZE = 124
train_data = pathlib.Path('/content/drive/My Drive/MozSnake/Training/')


def resnt34( num_class):

    stride =1;
    Channel_axis=3;

    def comom_bloxk(x):
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = tf.nn.relu(x)
        return x

    def residual_block(x_in, filters):

        x_tmp = x_in

        x_tmp = comom_bloxk(x_tmp)
        x_tmp = layers.Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')(x_tmp)

        x_tmp = comom_bloxk(x_tmp)
        x_tmp = layers.Conv2D(filters, kernel_size=(3, 3), strides=stride,  padding='same')(x_tmp)

        x_shortcut = layers.Conv2D(filters, kernel_size=(2, 2), strides=stride,   padding='same')(x_in)

        x = layers.add([x_tmp, x_shortcut])
        x = comom_bloxk(x)

        return x

    #     Conv 1
    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(img_input)
    x = comom_bloxk(x)

    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Conv2
    for i in range(3):
        x = residual_block(x, 64)

    # Conv2
    for i in range(4):
        x = residual_block(x, 128)

    # Conv3
    for i in range(6):
        x = residual_block(x, 256)


    # Conv4
    for i in range(3):
        x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)

    # x = layers.Dropout(0.25)(x)
    x= layers.Flatten()(x)
    output = layers.Dense(num_class, activation=tf.nn.softmax)(x)
    model = models.Model(inputs=img_input, outputs=output, name='ResNet34')

    return model

model = resnt34(num_class)
print(model.summary())

# plot_model(model, show_shapes=True, to_file='resnet34.png')

# def load_data(data_directory):
#
#     directories = [d for d in os.listdir(data_directory)
#                    if os.path.isdir(os.path.join(data_directory, d))]
#     labels = []
#     images = []
#     for d in directories:
#         label_directory = os.path.join(data_directory, d)
#         file_names = [os.path.join(label_directory, f)
#                       for f in os.listdir(label_directory)]
#         for f in file_names:
#             img = cv.imread(f)
#             img = cv.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
#             # cv.imwrite(f,img)
#             images.append(img)
#             labels.append(int(d))
#     return images, labels
#
# imagens , labels = load_data(train_data)
#
#
#
#
# kf = StratifiedKFold(n_splits=10)
#
# for train, test in kf.split(imagens, labels):
#     imgs_train = []
#     labs_train = []
#     imgs_test = []
#     labs_test = []
#     # print("Train::", train, "Test::", test)
#
#     for i in train:
#         imgs_train.append(imagens[i])
#         labs_train.append(labels[i])
#
#     for i in test:
#         imgs_test.append(imagens[i])
#         labs_test.append(labels[i])
#
#
#     imgs_train = np.asarray(imgs_train)
#     imgs_test = np.asarray(imgs_test)
#
# #     imgs_train = imgs_train.astype('float32')/np.float32(255)
# #     imgs_test = imgs_test.astype('float32')/np.float32(255)
#
#
#     labs_train, labs_test = np.array(labs_train), np.array(labs_test)
#     # labs_train = labs_train.astype(np.int32)
#     # labs_test = labs_test.astype(np.int32)
#
#     labs_train = tf.keras.utils.to_categorical(labs_train, num_class)
#
#     print(imgs_train.shape)
#     print(labs_train.shape)
#
#     model = resnt34(num_class)
#     model.compile(loss=tf.keras.losses.categorical_crossentropy,
#                   optimizer=tf.keras.optimizers.Adam(),
#                   metrics=['accuracy'])
#
#     model.fit(imgs_train, labs_train, verbose=2, epochs=70)
#     # model.load_weights('my_modelRN.h5')
#
#     x_predict = model.predict(imgs_test)
#     x_predict = np.argmax(x_predict, axis=1)
#
#     accuracy = metrics.accuracy_score(labs_test, x_predict)
#     f1 = metrics.f1_score(labs_test, x_predict, average="micro")
#     precision = metrics.precision_score(labs_test, x_predict, average="micro")
#     recall = metrics.recall_score(labs_test, x_predict, average="micro")
#
#     print('Acurracy: %f' % accuracy)
#     print('F1: %f' % f1)
#     print('Precision: %f' % precision)
#     print('Recall: %f' % recall)
