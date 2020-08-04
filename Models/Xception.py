from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pathlib
import os
import gc
import cv2 as cv
import random
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
from tensorflow.keras.callbacks import EarlyStopping
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

    #     random.shuffle(file_names)
    #     random.shuffle(file_names)

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


def xception(num_class):
    stride = (2, 2);
    kernel_size = (3, 3)
    pool_size = (2, 2)
    Channel_axis = 3;

    def entry_flow(img_input):

        x = layers.Conv2D(32, kernel_size=kernel_size,
                          use_bias=False, kernel_initializer='glorot_uniform',
                          strides=stride, padding='SAME')(img_input)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(64, kernel_size=kernel_size,
                          use_bias=False, kernel_initializer='glorot_uniform',
                          padding='SAME')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x_temp = x

        for filter in [128, 256, 728]:
            if filter != 128:
                x = LeakyReLU(alpha=0.1)(x)
            x = layers.SeparableConv2D(filters=filter, kernel_size=kernel_size,
                                       use_bias=False, kernel_initializer='glorot_uniform',
                                       padding='SAME')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = LeakyReLU(alpha=0.1)(x)
            x = layers.SeparableConv2D(filters=filter, kernel_size=kernel_size,
                                       use_bias=False, kernel_initializer='glorot_uniform',
                                       padding='SAME')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

            x_shortcut = layers.Conv2D(filters=filter, kernel_size=(1, 1),
                                       use_bias=False, kernel_initializer='glorot_uniform',
                                       strides=stride, padding='SAME')(x_temp)
            x_shortcut = layers.BatchNormalization(axis=Channel_axis)(x_shortcut)

            x = layers.add([x, x_shortcut])
            x_temp = x

        return x

    def middle_flow(x):

        x_temp = x

        for i in range(8):
            x = LeakyReLU(alpha=0.1)(x)
            x = layers.SeparableConv2D(filters=728, kernel_size=kernel_size,
                                       use_bias=False, kernel_initializer='glorot_uniform',
                                       padding='SAME')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = LeakyReLU(alpha=0.1)(x)
            x = layers.SeparableConv2D(filters=728, kernel_size=kernel_size,
                                       use_bias=False, kernel_initializer='glorot_uniform',
                                       padding='SAME')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = LeakyReLU(alpha=0.1)(x)
            x = layers.SeparableConv2D(filters=728, kernel_size=kernel_size,
                                       use_bias=False, kernel_initializer='glorot_uniform',
                                       padding='SAME')(x)
            x = layers.BatchNormalization(axis=Channel_axis)(x)

            x = layers.add([x, x_temp])
            x_temp = x

        return x

    def exit_flow(x):

        x_temp = x

        x = LeakyReLU(alpha=0.1)(x)
        x = layers.SeparableConv2D(filters=728, kernel_size=kernel_size,
                                   use_bias=False, kernel_initializer='glorot_uniform',
                                   padding='SAME')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)

        x = LeakyReLU(alpha=0.1)(x)
        x = layers.SeparableConv2D(filters=1024, kernel_size=kernel_size,
                                   use_bias=False, kernel_initializer='glorot_uniform',
                                   padding='SAME')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)

        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x_shortcut = layers.Conv2D(filters=1024, kernel_size=kernel_size,
                                   use_bias=False, kernel_initializer='glorot_uniform',
                                   strides=stride, padding='SAME')(x_temp)
        x_shortcut = layers.BatchNormalization(axis=Channel_axis)(x_shortcut)

        x = layers.add([x, x_shortcut])

        x = layers.SeparableConv2D(filters=1536, kernel_size=kernel_size,
                                   use_bias=False, kernel_initializer='glorot_uniform',
                                   padding='SAME')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.SeparableConv2D(filters=2048, kernel_size=kernel_size,
                                   use_bias=False, kernel_initializer='glorot_uniform',
                                   padding='SAME')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.GlobalAveragePooling2D()(x)

        #         x = layers.Dropout(0.8)(x)

        output = layers.Dense(num_class, activation=tf.nn.softmax, dtype=tf.float32)(x)

        return output

    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = entry_flow(img_input)
    x = middle_flow(x)
    output = exit_flow(x)

    model = models.Model(inputs=img_input, outputs=output, name='Xception')

    return model


# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=29)

indice = 0

for train, test in kf.split(imagens, labels):
    imgs_train, imgs_val = imagens[train], imagens[test]
    labs_train, labs_val = labels[train], labels[test]

    # labs_train = tf.keras.utils.to_categorical(labs_train, num_class)
    # labs_test = tf.keras.utils.to_categorical(labs_test, num_class)

    print(imgs_train.shape)
    print(labs_train.shape)

    ## instantiating the model in the strategy scope creates the model on the TPU
    with tpu_strategy.scope():
        model = None

        model = xception(num_class)

        checkpoint = ModelCheckpoint(str(pesos) + "/Xception" + ".h5",
                                     monitor='accuracy', verbose=1,
                                     save_best_only=True, mode='auto')

        # earlstop = EarlyStopping(monitor='val_loss', patience=12,  verbose=0)

        # if indice > 0:
        #     model.load_weights(str(pesos) + "/Xception" + ".h5")
        #     print("Load Model")

        model.compile(loss=loss_function,
                      optimizer=optmizer,
                      metrics=['accuracy'])

    xcept = model.fit(imgs_train, labs_train,
                      batch_size=batchsize,
                      validation_data=(imgs_val, labs_val), verbose=1,
                      epochs=epoch, callbacks=[checkpoint])

    imgs_test, labs_test = load_data(test_data)

    # Load the best model to make predict
    model.load_weights(str(pesos) + "/Xception" + ".h5")
    model.compile(loss=loss_function, optimizer=optmizer, metrics=['accuracy'])

    print(imgs_test.shape)
    print(labs_test.shape)

    labs_predict = model.predict(imgs_test)
    labs_predict = np.argmax(labs_predict, axis=1)

    accuracy = metrics.accuracy_score(labs_test, labs_predict)
    acc_folds.append(accuracy)

    f1 = metrics.f1_score(labs_test, labs_predict, average="macro")
    f1_folds.append(f1)

    precision = metrics.precision_score(labs_test, labs_predict, average="macro")
    prec_folds.append(precision)

    recall = metrics.recall_score(labs_test, labs_predict, average="macro")
    recall_folds.append(recall)

    print('Acurracy: %f' % accuracy)
    print('F1: %f' % f1)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)

    indice += 1

    hist_df = pd.DataFrame(xcept.history)
    hist_csv_file = str(pesos) + '/Xceptionhistory_' + str(indice) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    plt.figure(0)
    plt.plot(xcept.history['accuracy'], 'r', label="train_acc")
    plt.plot(xcept.history['val_accuracy'], 'g', label="val_acc")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.yscale('linear')
    plt.xscale('linear')
    plt.title("Xception Fold "+ str(indice) +" \n \n"
              "Training and Validation Accuracy")
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.grid(True, color='w', linestyle='-', linewidth=2)
    plt.gca().patch.set_facecolor('lightgrey')
    plt.legend()
    plt.savefig(str(pesos) + '/XceptionAcc_' + str(indice) + '.png')
    plt.show()
    plt.close()

    plt.figure(1)
    plt.plot(xcept.history['loss'], 'b', label="train_loss")
    plt.plot(xcept.history['val_loss'], 'm', label="val_loss")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.yscale('linear')
    plt.xscale('linear')
    plt.title("Xception Fold "+ str(indice) +" \n \n"
              "Training and Validation Loss")
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.grid(True, color='w', linestyle='-', linewidth=2)
    plt.gca().patch.set_facecolor('lightgrey')
    plt.legend()
    plt.savefig(str(pesos) + '/XceptionLoss_' + str(indice) + '.png')
    plt.show()

    model = None
    imgs_test = None
    labs_test = None
    imgs_train = None
    labs_train = None
    imgs_val = None
    labs_val = None

    del xcept
    del model
    del imgs_train
    del imgs_test
    del labs_test
    del labs_train
    del imgs_val
    del labs_val
    gc.collect()

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
