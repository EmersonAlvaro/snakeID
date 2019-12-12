# from google.colab import drive
# drive.mount('/content/drive/')

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
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

num_class = 5
IMAGE_SIZE = 299
train_data = pathlib.Path('/content/drive/My Drive/MozSnake/Training/')
Channel_axis = 3;


def conv2d(x, filter,  kernel_size,strides=(1, 1), padding='same'):

    x= layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    # x = layers.BatchNormalization(axis=Channel_axis)(x)
    x = tf.nn.relu(x)

    return x

def inception_A(x):
    input = x

    a1 = conv2d(x, 32, kernel_size=(1, 1), padding='same')
    a1 = conv2d(a1, 48, kernel_size=(3, 3), padding='same')
    a1 = conv2d(a1,  64, kernel_size=(3, 3), padding='same')

    a2 = conv2d(x, 32, kernel_size=(1, 1), padding='same')
    a2 = conv2d(a2,32, kernel_size=(3, 3), padding='same')

    a3 = conv2d(x, 32, kernel_size=(1, 1), padding='same')

    a4 = layers.concatenate([a1, a2, a3,], axis=-1)

    a4 = conv2d(a4, 384, kernel_size=(1, 1), padding='same')

    x = layers.Lambda(lambda x: x * 0.1)(a4)

    x = layers.concatenate([input, x])
    # x = layers.BatchNormalization(axis=Channel_axis)(x)
    x = tf.nn.relu(x)

    return x

def inception_B(x):

    input= x
    b1 = conv2d(x,  128, kernel_size=(1, 1), padding='same')
    b1 = conv2d(b1, 160, kernel_size=(1, 7), padding='same')
    b1 = conv2d(b1, 192, kernel_size=(7, 1), padding='same')

    b2 = conv2d(x, 192, kernel_size=(1, 1), padding='same')

    b2 = layers.concatenate([b1, b2,], axis=-1)
    b2 = conv2d(b2, 1154, kernel_size=(1, 1), padding='same')

    x = layers.Lambda(lambda x: x * 0.1)(b2)

    x = layers.concatenate([input, x])
    # x = layers.BatchNormalization(axis=Channel_axis)(x)
    x = tf.nn.relu(x)

    return x

def inception_C(x):

    input = x

    c1 = conv2d(x, 192, kernel_size=(1, 1), padding='same')
    c1 = conv2d(c1, 224, kernel_size=(1, 3), padding='same')
    c1 = conv2d(c1, 256, kernel_size=(3, 1), padding='same')

    c2 = conv2d(x, 192,  kernel_size=(1, 1), padding='same')

    c4 = layers.concatenate([c1, c2], axis=-1)

    c4 = conv2d(c4, 2048, kernel_size=(1, 1), padding='same')

    x = layers.Lambda(lambda x: x * 0.1)(c4)

    x = layers.concatenate([input, x])
    # x = layers.BatchNormalization(axis=Channel_axis)(x)
    x = tf.nn.relu(x)

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
    x = tf.nn.relu(x)

    return x

def reduction_A(x, k, l, m, n ):

    x1 = conv2d(x, k, kernel_size=(1, 1), padding='same')
    x1 = conv2d(x1, l, kernel_size=(3, 3), padding='same')
    x1 = conv2d(x1, m, kernel_size=(3, 3), strides=(2, 2), padding='same')

    x2 = conv2d(x,  n, kernel_size=(3, 3),strides=(2, 2), padding='same')

    x3 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = layers.concatenate([x1, x2, x3], axis=-1)
    x = tf.nn.relu(x)

    return x

def reduction_B(x):

    x1 = conv2d(x, 256, kernel_size=(1, 1), padding='same')
    x1 = conv2d(x1, 288, kernel_size=(3, 3), padding='same')
    x1 = conv2d(x1, 320, kernel_size=(3, 3), strides=(2, 2), padding='same')

    x2 = conv2d(x, 256, kernel_size=(1, 1), padding='same')
    x2 = conv2d(x2, 288, kernel_size=(3, 3),strides=(2, 2),  padding='same')

    x3 = conv2d(x, 256, kernel_size=(1, 1), padding='same')
    x3 = conv2d(x3, 384, kernel_size=(3, 3), strides=(2, 2), padding='same')

    x4 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)
    x = tf.nn.relu(x)

    return x

def inceptionV4( num_class):

    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = stem(img_input)

    for i in range(5):
        x= inception_A(x)

    x = reduction_A(x, k=256, l=256, m=384, n=384)

    for i in range(10):
        x = inception_B(x)

    x = reduction_B(x)

    for i in range(5):
        x = inception_C(x)

    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(num_class, activation=tf.nn.softmax)(x)

    model = models.Model(inputs=img_input, outputs=output, name='InceptionResNetV2')

    return model

Inception = inceptionV4(num_class)

print(Inception.summary())
# plot_model(Inception, to_file='model.png', show_shapes=True, show_layer_names=True)

def load_data(data_directory):

    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)]
        for f in file_names:
            img = cv.imread(f)
            img = cv.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
            # cv.imwrite(f,img)
            images.append(img)
            labels.append(int(d))
    return images, labels

imagens , labels = load_data(train_data)


kf = StratifiedKFold(n_splits=10)

for train, test in kf.split(imagens, labels):
    imgs_train = []
    labs_train = []
    imgs_test = []
    labs_test = []
    # print("Train::", train, "Test::", test)

    for i in train:
        imgs_train.append(imagens[i])
        labs_train.append(labels[i])

    for i in test:
        imgs_test.append(imagens[i])
        labs_test.append(labels[i])


    imgs_train = np.asarray(imgs_train)
    imgs_test = np.asarray(imgs_test)

#     imgs_train = imgs_train.astype('float32')/np.float32(255)
#     imgs_test = imgs_test.astype('float32')/np.float32(255)


    labs_train, labs_test = np.array(labs_train), np.array(labs_test)
    # labs_train = labs_train.astype(np.int32)
    # labs_test = labs_test.astype(np.int32)

    labs_train = tf.keras.utils.to_categorical(labs_train, num_class)

    print(imgs_train.shape)
    print(labs_train.shape)

    model = inceptionV4(num_class)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(imgs_train, labs_train, verbose=2, epochs=70)
    # model.load_weights('my_modelRN.h5')

    x_predict = model.predict(imgs_test)
    x_predict = np.argmax(x_predict, axis=1)

    accuracy = metrics.accuracy_score(labs_test, x_predict)
    f1 = metrics.f1_score(labs_test, x_predict, average="micro")
    precision = metrics.precision_score(labs_test, x_predict, average="micro")
    recall = metrics.recall_score(labs_test, x_predict, average="micro")

    print('Acurracy: %f' % accuracy)
    print('F1: %f' % f1)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)


