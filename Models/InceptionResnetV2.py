from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pathlib
import os
import gc
import cv2 as cv
import random
import time
import pandas as pd

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

# # Colab Path
# train_data = pathlib.Path('/content/drive/My Drive/MozSnake/train/')
# test_data = pathlib.Path('/content/drive/My Drive/MozSnake/test/')
# pesos = pathlib.Path('/content/drive/My Drive/MozSnake/pesos/')
#
# from google.colab import drive
#
# drive.mount('/content/drive/')

# Kagle Notebook
train_data = pathlib.Path('/kaggle/input/snakemoz/MozSnake/train/')
test_data = pathlib.Path('/kaggle/input/snakemoz/MozSnake/test/')
pesos = pathlib.Path('/kaggle/working/')


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
        img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        dirname = os.path.split(os.path.dirname(f))[1]

        # print(dirname)

        images.append(img)
        labels.append(int(dirname))

    images, labels = np.asarray(images), np.asarray(labels)
    images = images.astype('float32') / 255.0

    return images, labels


imagens, labels = load_data(train_data)
print(imagens.shape)
print(labels.shape)


def inception_resnet_v2(num_class):

    def conv2d(x, filter, kernel_size, strides=(1, 1), padding='same'):
        Channel_axis = 3;
        x = layers.Conv2D(filters=filter, kernel_size=kernel_size,
                          use_bias=False, kernel_initializer='glorot_uniform',
                          strides=strides, padding=padding)(x)

        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = LeakyReLU(alpha=0.1)(x)

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

        x_1 = conv2d(x, 192, kernel_size=(3, 3), strides=(2, 2), padding='same')
        x_2 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = layers.concatenate([x_1, x_2], axis=-1)
        x = tf.nn.relu(x)

        return x

    def inception_resnet_A(x):

        a1 = conv2d(x, 32, kernel_size=(1, 1), padding='same')
        a1 = conv2d(a1, 48, kernel_size=(3, 3), padding='same')
        a1 = conv2d(a1, 64, kernel_size=(3, 3), padding='same')

        a2 = conv2d(x, 32, kernel_size=(1, 1), padding='same')
        a2 = conv2d(a2, 32, kernel_size=(3, 3), padding='same')

        a3 = conv2d(x, 32, kernel_size=(1, 1), padding='same')

        a4 = layers.concatenate([a1, a2, a3, ], axis=-1)

        a4 = layers.Conv2D(384, kernel_size=1, activation='linear', padding='same')(a4)

        x = layers.concatenate([x, a4])
        x = tf.nn.relu(x)

        return x

    def reduction_A(x, k, l, m, n):
        x1 = conv2d(x, k, kernel_size=(1, 1), padding='same')
        x1 = conv2d(x1, l, kernel_size=(3, 3), padding='same')
        x1 = conv2d(x1, m, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x2 = conv2d(x, n, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x3 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = layers.concatenate([x1, x2, x3], axis=-1)
        x = tf.nn.relu(x)

        return x

    def inception_resnet_B(x):

        b1 = conv2d(x, 128, kernel_size=(1, 1), padding='same')
        b1 = conv2d(b1, 160, kernel_size=(1, 7), padding='same')
        b1 = conv2d(b1, 192, kernel_size=(7, 1), padding='same')

        b2 = conv2d(x, 192, kernel_size=(1, 1), padding='same')

        b2 = layers.concatenate([b1, b2, ], axis=-1)
        b2 = layers.Conv2D(1154, kernel_size=(1, 1), activation='linear', padding='same')(b2)

        x = layers.concatenate([x, b2])
        x = tf.nn.relu(x)

        return x

    def reduction_B(x):
        x1 = conv2d(x, 256, kernel_size=(1, 1), padding='same')
        x1 = conv2d(x1, 288, kernel_size=(3, 3), padding='same')
        x1 = conv2d(x1, 320, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x2 = conv2d(x, 256, kernel_size=(1, 1), padding='same')
        x2 = conv2d(x2, 288, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x3 = conv2d(x, 256, kernel_size=(1, 1), padding='same')
        x3 = conv2d(x3, 384, kernel_size=(3, 3), strides=(2, 2), padding='same')

        x4 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = layers.concatenate([x1, x2, x3, x4], axis=-1)
        x = tf.nn.relu(x)

        return x

    def inception_resnet_C(x):

        c1 = conv2d(x, 192, kernel_size=(1, 1), padding='same')
        c1 = conv2d(c1, 224, kernel_size=(1, 3), padding='same')
        c1 = conv2d(c1, 256, kernel_size=(3, 1), padding='same')

        c2 = conv2d(x, 192, kernel_size=(1, 1), padding='same')

        c4 = layers.concatenate([c1, c2], axis=-1)

        c4 = layers.Conv2D(2048, kernel_size=(1, 1), activation='linear', padding='same')(c4)

        x = layers.concatenate([x, c4])
        x = tf.nn.relu(x)

        return x

    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = stem(img_input)

    for i in range(5):
        x = inception_resnet_A(x)

    x = reduction_A(x, k=256, l=256, m=384, n=384)

    for i in range(10):
        x = inception_resnet_B(x)

    x = reduction_B(x)

    for i in range(5):
        x = inception_resnet_C(x)

    x = layers.GlobalAveragePooling2D()(x)

    output = layers.Dense(num_class, activation=tf.nn.softmax)(x)

    model = models.Model(inputs=img_input, outputs=output, name='InceptionResNetV2')

    return model


# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)

kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=29)

indice = 0

for train, test in kf.split(imagens, labels):

    #reinitialize the TPU
    tf.tpu.experimental.initialize_tpu_system(tpu)
    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    imgs_train, imgs_val = imagens[train], imagens[test]
    labs_train, labs_val = labels[train], labels[test]

    # labs_train = tf.keras.utils.to_categorical(labs_train, num_class)
    # labs_test = tf.keras.utils.to_categorical(labs_test, num_class)

    print(imgs_train.shape)
    print(labs_train.shape)

    # instantiating the model in the strategy scope creates the model on the TPU
    with tpu_strategy.scope():

        model = inception_resnet_v2(num_class)

        # checkpoint = ModelCheckpoint(str(pesos) + "/InceptionResNetV2" + ".h5",
        #                              monitor='accuracy', verbose=1,
        #                              save_best_only=True, mode='auto')

        model.compile(loss=loss_function,
                      optimizer=optmizer,
                      metrics=['accuracy'])

    inceptresnt = model.fit(imgs_train, labs_train,
                            batch_size=batchsize,
                            validation_data=(imgs_val, labs_val), verbose=1,
                            epochs=epoch
                            # , callbacks=[checkpoint]
                            )
    imgs_train = None
    labs_train = None
    imgs_val = None
    labs_val = None

    # imgs_test, labs_test = load_data(test_data)
    #
    # labs_predict = model.predict(imgs_test)
    # labs_predict = np.argmax(labs_predict, axis=1)
    # #
    #
    # accuracy = metrics.accuracy_score(labs_test, labs_predict)
    # acc_folds.append(accuracy)
    #
    # f1 = metrics.f1_score(labs_test, labs_predict, average="macro")
    # f1_folds.append(f1)
    #
    # precision = metrics.precision_score(labs_test, labs_predict, average="macro")
    # prec_folds.append(precision)
    #
    # recall = metrics.recall_score(labs_test, labs_predict, average="macro")
    # recall_folds.append(recall)
    #
    # print('Acurracy: %f' % accuracy)
    # print('F1: %f' % f1)
    # print('Precision: %f' % precision)
    # print('Recall: %f' % recall)

    indice += 1
    hist_df = pd.DataFrame(inceptresnt.history)
    hist_csv_file = str(pesos) + '/InCeptionResNethistoryFold' + str(indice) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    plt.figure(0)
    plt.plot(inceptresnt.history['accuracy'], 'r', label="train_acc")
    plt.plot(inceptresnt.history['val_accuracy'], 'g', label="val_acc") 
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("InceptionResNet - Training and Validation Accuracy")
    plt.legend(loc='upper right')
    plt.grid(True, color='w', linestyle='-', linewidth=2)
    plt.gca().patch.set_facecolor('lightgrey')
    plt.legend()
    # plt.savefig(str(pesos) + '/InceptionResNetAcc.png')
    plt.show()    
    plt.close()
    
    plt.figure(1)
    plt.plot(inceptresnt.history['loss'], 'b', label="train_loss")
    plt.plot(inceptresnt.history['val_loss'], 'm', label="val_loss")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("InceptionResNet - Training and Validation Loss")
    plt.legend(loc='upper right')
    plt.grid(True, color='w', linestyle='-', linewidth=2)
    plt.gca().patch.set_facecolor('lightgrey')
    plt.legend()
    # plt.savefig(str(pesos) + '/InceptionResNetLoss.png')
    plt.show()  

    model = None
    plt.show()
    del inceptresnt
    del model
    del imgs_train
    # del imgs_test
    # del labs_test
    del labs_train
    del imgs_val
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