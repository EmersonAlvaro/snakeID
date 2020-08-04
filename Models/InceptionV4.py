from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pathlib
import os
import gc
import cv2 as cv
import random
import pandas as pd
import time

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
# from tensorflow.keras.utils import np_utils
from sklearn import metrics
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

num_class = 4
IMAGE_SIZE = 256
epoch = 70
loss_function = sparse_categorical_crossentropy
optmizer = Adam()
num_folds = 10
batchsize = 32

# container for metrics

acc_folds = []
f1_folds = []
prec_folds = []
recall_folds = []

# train_data = pathlib.Path('/content/drive/My Drive/MozSnake/train/')
# test_data = pathlib.Path('/content/drive/My Drive/MozSnake/test/')
# pesos = pathlib.Path('/content/drive/My Drive/MozSnake/pesos/')

# Kagle Notebook Emerson Cardoso
train_data = pathlib.Path('/kaggle/input/snakemoz/MozSnake/train/')
test_data = pathlib.Path('/kaggle/input/snakemoz/MozSnake/test/')
pesos = pathlib.Path('/kaggle/working/')

Channel_axis = 3;


def inceptionv4(num_class):
    def conv2d(x, filter, kernel_size, strides=(1, 1), padding='same'):
        x = layers.Conv2D(filters=filter, kernel_size=kernel_size,
                          use_bias=False, kernel_initializer='glorot_uniform',
                          strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = layers.Activation('relu')(x)

        return x

    def stem(img_imput):

        x = conv2d(img_imput, 32, kernel_size=(3, 3), strides=(2, 2), padding='same')
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
        x = layers.BatchNormalization(axis=Channel_axis)(x)

        x_1 = conv2d(x, 192, kernel_size=(3, 3), strides=(2, 2), padding='same')
        x_2 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = layers.concatenate([x_1, x_2], axis=-1)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = tf.nn.relu(x)

        return x

    def inception_a(x):
        a1 = conv2d(x, 64, kernel_size=(1, 1), padding='same')
        a1 = conv2d(a1, 96, kernel_size=(3, 3), padding='same')
        a1 = conv2d(a1, 96, kernel_size=(3, 3), padding='same')

        a2 = conv2d(x, 64, kernel_size=(1, 1), padding='same')
        a2 = conv2d(a2, 96, kernel_size=(3, 3), padding='same')

        a3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        a3 = conv2d(a3, 96, kernel_size=(1, 1), padding='same')

        a4 = conv2d(x, 96, kernel_size=(1, 1), padding='same')

        x = layers.concatenate([a1, a2, a3, a4], axis=-1)
        x = tf.nn.relu(x)

        return x

    def reduction_a(x):
        x1 = conv2d(x, 192, kernel_size=(1, 1), padding='same')
        x1 = conv2d(x1, 224, kernel_size=(3, 3), padding='same')
        x1 = conv2d(x1, 256, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x2 = conv2d(x, 32, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x3 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = layers.concatenate([x1, x2, x3], axis=-1)

        x = tf.nn.relu(x)

        return x

    def inception_b(x):
        b1 = conv2d(x, 192, kernel_size=(1, 1), padding='same')
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
        x = tf.nn.relu(x)

        return x

    def reduction_b(x):
        x1 = conv2d(x, 256, kernel_size=(1, 1), padding='same')
        x1 = conv2d(x1, 256, kernel_size=(1, 7), padding='same')
        x1 = conv2d(x1, 320, kernel_size=(7, 1), padding='same')
        x1 = conv2d(x1, 320, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x3 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x2 = conv2d(x, 192, kernel_size=(1, 1), padding='same')
        x2 = conv2d(x2, 192, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x = layers.concatenate([x1, x2, x3], axis=-1)

        x = tf.nn.relu(x)

        return x

    def inception_c(x):
        c1 = conv2d(x, 384, kernel_size=(1, 1), padding='same')
        c1 = conv2d(c1, 446, kernel_size=(3, 1), padding='same')
        c1 = conv2d(c1, 512, kernel_size=(1, 3), padding='same')
        c1_1 = conv2d(c1, 256, kernel_size=(1, 3), padding='same')
        c1_2 = conv2d(c1, 256, kernel_size=(3, 1), padding='same')

        c2 = conv2d(x, 384, kernel_size=(1, 1), padding='same')
        c2_1 = conv2d(c2, 256, kernel_size=(1, 3), padding='same')
        c2_2 = conv2d(c2, 256, kernel_size=(3, 1), padding='same')

        c3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        c3 = conv2d(c3, 256, kernel_size=(1, 1), padding='same')

        c4 = conv2d(x, 256, kernel_size=(1, 1), padding='same')

        x = layers.concatenate([c1_1, c1_2, c2_1, c2_2, c3, c4], axis=-1)

        x = tf.nn.relu(x)

        return x

    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = stem(img_input)

    for i in range(4):
        x = inception_a(x)

    x = reduction_a(x)

    for i in range(7):
        x = inception_b(x)

    x = reduction_b(x)

    for i in range(3):
        x = inception_c(x)

    x = layers.GlobalAveragePooling2D()(x)

    output = layers.Dense(num_class, activation='softmax', dtype=tf.float32)(x)

    model = models.Model(inputs=img_input, outputs=output, name='InceptionV4')

    return model


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    file_names = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names += [os.path.join(label_directory, f)
                       for f in os.listdir(label_directory)]

    random.shuffle(file_names)
    random.shuffle(file_names)

    for f in file_names:
        # print(f)
        img = cv.imread(f)
        img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv.INTER_AREA)
        dirname = os.path.split(os.path.dirname(f))[1]
        # print(dirname)

        images.append(img)
        labels.append(int(dirname))

    images, labels = np.asarray(images), np.asarray(labels)
    images = images.astype('float32')
    # labels = labels.astype('int32')
    images /= 255

    return images, labels


# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

imagens, labels = load_data(train_data)

kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=5)
TrainAgain = False

indice = 0

for train, test in kf.split(imagens, labels):
    imgs_train, imgs_val = imagens[train], imagens[test]
    labs_train, labs_val = labels[train], labels[test]

    print(imgs_train.shape)
    print(labs_train.shape)
    print(indice)

    ## instantiating the model in the strategy scope creates the model on the TPU
    with tpu_strategy.scope():
        model = inceptionv4(num_class)

        checkpoint = ModelCheckpoint(str(pesos) + "/Inceptionv4" + ".h5",
                                     monitor='accuracy', verbose=1,
                                     save_best_only=True, mode='auto')
        model.compile(loss=loss_function,
                      optimizer=optmizer,
                      metrics=['accuracy'])

        # if indice > 0:
        #     model.load_weights(str(pesos) + "/Inceptionv4" + ".h5")
        #     print("Load Model to new model ")

    incept = model.fit(imgs_train, labs_train,
                       batch_size=batchsize,
                       validation_data=(imgs_val, labs_val), verbose=1,
                       epochs=epoch
                       #                        , callbacks=[checkpoint]
                       )
    imgs_train = None
    labs_train = None
    imgs_val = None
    labs_val = None

    indice += 1

    hist_df = pd.DataFrame(incept.history)
    hist_csv_file = str(pesos) + '/InceptionhistoryFold' + str(indice) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    #     imgs_test, labs_test = load_data(test_data)

    #     #Predict from the best model
    #     model.load_weights(str(pesos) + "/Inceptionv4" + ".h5")

    #     labs_predict = model.predict(imgs_test)
    #     labs_predict = np.argmax(labs_predict, axis=1)

    #     accuracy = metrics.accuracy_score(labs_test, labs_predict)
    #     acc_folds.append(accuracy)

    #     f1 = metrics.f1_score(labs_test, labs_predict, average="macro")
    #     f1_folds.append(f1)

    #     precision = metrics.precision_score(labs_test, labs_predict, average="macro")
    #     prec_folds.append(precision)

    #     recall = metrics.recall_score(labs_test, labs_predict, average="macro")
    #     recall_folds.append(recall)

    #     print('Acurracy: %f' % accuracy)
    #     print('F1: %f' % f1)
    #     print('Precision: %f' % precision)
    #     print('Recall: %f' % recall)

    plt.figure(0)
    plt.plot(incept.history['accuracy'], 'r', label="train_acc")
    plt.plot(incept.history['val_accuracy'], 'g', label="val_acc")
    # plt.xticks(np.arange(0, 16.00, 2.0))
    # plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accurac")
    plt.title("Inception - Training and Validation Accuracy")
    plt.legend(loc='upper right')
    plt.grid(True, color='w', linestyle='-')
    plt.gca().patch.set_facecolor('lightgrey')
    plt.legend()
    #     plt.savefig(str(pesos) + '/InceptionAcc.png')
    plt.show()
    plt.close()

    plt.figure(1)
    plt.plot(incept.history['loss'], 'b', label="train_loss")
    plt.plot(incept.history['val_loss'], 'm', label="val_loss")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.title("Inception - Training and Validation Loss")
    plt.legend(loc='upper right')
    plt.grid(True, color='w', linestyle='-')
    plt.gca().patch.set_facecolor('lightgrey')
    plt.legend()
    #     plt.savefig(str(pesos) + '/InceptionLoss.png')
    plt.show()

    model = None
    imgs_test = None
    labs_test = None

    plt.show()
    del incept
    gc.collect()
    del model
    gc.collect()
    del imgs_train
    gc.collect()
    del imgs_test
    gc.collect()
    del labs_test
    gc.collect()
    del labs_train
    gc.collect()
    del imgs_val
    gc.collect()
    del labs_val
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    # time.sleep(60)

print("===================================================================")
print(f'> Global accuracy: {np.mean(acc_folds)} {np.std(acc_folds)}')
print(f'> Global F1_Score:  {np.mean(f1_folds)} {np.std(f1_folds)}')
print(f'> Global Precision: {np.mean(prec_folds)} {np.std(prec_folds)}')
print(f'> Global Recall:  {np.mean(recall_folds)} {np.std(recall_folds)}')
print("===================================================================")

print("Acuracia", acc_folds)
print("===================================================================")
print("F1 Score", f1_folds)
print("===================================================================")
print("Precision", prec_folds)
print("===================================================================")
print("Recall", recall_folds)