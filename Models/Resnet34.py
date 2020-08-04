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
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import SGD

num_class = 4
IMAGE_SIZE = 256
epoch = 70
loss_function = sparse_categorical_crossentropy
optmizer = SGD()
num_folds = 10
batchsize = 32

# container for metrics
acc_folds = []
f1_folds = []
prec_folds = []
recall_folds = []

# #Colab Path
# train_data = pathlib.Path('/content/drive/My Drive/MozSnake/train/')
# test_data = pathlib.Path('/content/drive/My Drive/MozSnake/test/')
# pesos = pathlib.Path('/content/drive/My Drive/MozSnake/pesos/')
#
# #Mount drive
# from google.colab import drive
# drive.mount('/content/drive/')


# Kagle Notebook Emerson Cardoso
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
        img = cv.imread(f)
        img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        # cv.imwrite(f,img)
        dirname = os.path.split(os.path.dirname(f))[1]

        images.append(img)
        labels.append(int(dirname))

    images, labels = np.asarray(images), np.asarray(labels)
    images, labels = np.asarray(images), np.asarray(labels)
    images = images.astype('float32')
    images /= 255

    return images, labels


imagens, labels = load_data(train_data)
print(imagens.shape)
print(labels.shape)


def resnt34(num_class):
    stride = 1;
    Channel_axis = 3;

    def residual_block(x, filter):

        residual = layers.Conv2D(filters=filter, kernel_size=(2, 2), use_bias=False,
                                 kernel_initializer='glorot_uniform',
                                 padding='SAME')(x)
        residual = layers.BatchNormalization(axis=Channel_axis)(residual)

        x = layers.Conv2D(filters=filter, kernel_size=(3, 3), use_bias=False,
                          # kernel_regularizer=regularizers.l2(0.0001),
                          kernel_initializer='glorot_uniform',
                          padding='SAME')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(filters=filter, kernel_size=(3, 3), use_bias=False,
                          # kernel_regularizer=regularizers.l2(0.0001),
                          kernel_initializer='glorot_uniform', padding='SAME')(x)
        x = layers.BatchNormalization(axis=Channel_axis)(x)

        x = layers.add([x, residual])
        x = LeakyReLU(alpha=0.1)(x)

        return x

    #     Conv 1
    img_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), use_bias=False,
                      kernel_initializer='glorot_uniform',
                      padding='SAME')(img_input)
    x = layers.BatchNormalization(axis=Channel_axis)(x)
    x = tf.nn.relu(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME')(x)

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

    # x = layers.Dropout(0.5)(x)

    output = layers.Dense(num_class, activation=tf.nn.softmax, dtype=tf.float32)(x)

    model = models.Model(inputs=img_input, outputs=output, name='ResNet34')

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
    # labs_val = tf.keras.utils.to_categorical(labs_val, num_class)

    print(imgs_train.shape)
    print(labs_train.shape)

    # instantiating the model in the strategy scope creates the model on the TPU
    with tpu_strategy.scope():
        model = None

        model = resnt34(num_class)

        checkpoint = ModelCheckpoint(str(pesos) + "/ResNet34" + ".h5",
                                     monitor='accuracy', verbose=1,
                                     save_best_only=True, mode='auto')
        # if indice > 0:
        #     model.load_weights(str(pesos) + "/ResNetV1" + ".h5")
        #     print("Load Model")

        model.compile(loss=loss_function,
                      optimizer=optmizer,
                      metrics=['accuracy'])

    resnet = model.fit(imgs_train, labs_train,
                       batch_size=batchsize,
                       validation_data=(imgs_val, labs_val), verbose=1,
                       epochs=epoch, callbacks=[checkpoint])

    imgs_test, labs_test = load_data(test_data)

    #Load the best model to make predict
    model.load_weights(str(pesos) + "/ResNet34" + ".h5")
    model.compile(loss=loss_function,optimizer=optmizer,metrics=['accuracy'])

    labs_predict = model.predict(imgs_test)
    labs_predict = np.argmax(labs_predict, axis=1)
    #
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
    hist_df = pd.DataFrame(resnet.history)
    hist_csv_file = str(pesos) + '/InCeptionResNethistoryFold' + str(indice) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    plt.figure()

    plt.plot(resnet.history['accuracy'], 'r', label="train_acc")
    plt.plot(resnet.history['val_accuracy'], 'g', label="val_acc")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.yscale('linear')
    plt.xscale('linear')
    plt.title("ResNet 34" + str(indice) +"\n \n"
              "Training and Validation Accuracy")
    plt.tight_layout()
    plt.grid(True, color='w', linestyle='-', linewidth=2)
    plt.gca().patch.set_facecolor('lightgrey')
    plt.legend()
    plt.savefig(str(pesos) + '/ResNetACC_' + str(indice) + '.png')
    plt.show()

    plt.plot(resnet.history['loss'], 'b', label="train_loss")
    plt.plot(resnet.history['val_loss'], 'm', label="val_loss")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.yscale('linear')
    plt.xscale('linear')
    plt.title("ResNet 34" + str(indice) +"\n \n"
              "Training and Validation Loss")
    plt.tight_layout()
    plt.grid(True, color='w', linestyle='--')
    plt.gca().patch.set_facecolor('lightgrey')
    plt.legend()
    plt.savefig(str(pesos) + '/ResNetLoss_' + str(indice) + '.png')
    plt.show()

    model = None
    plt.show()
    del resnet
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