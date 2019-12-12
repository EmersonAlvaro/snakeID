# from __future__ import absolute_import, division, print_function, unicode_literals
# import tensorflow as tf
# import numpy as np
# import pathlib
import os
#
# from snake.config import *
# import cv2 as cv
#
# from tensorflow.python.keras import layers
# from tensorflow.python.keras import models
# from tensorflow.python.keras.utils import np_utils
import random
from sklearn.model_selection import KFold

DataSets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
DataSet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


# DataSetsCopy = DataSets.copy()
kf = KFold(n_splits=4, shuffle=True)
for train, test in kf.split(DataSets):
    img_test=[]
    img_train =[]
    lab_test=[]
    lab_train = []
    print("%s %s" % (train, test))

    for i in train:
        # img_train, lab_train = DataSets[i], DataSet[i]
        img_train.append(DataSets[i])
        lab_train.append(DataSet[i])

    for i in test:
        # img_test, lab_test = DataSet[i], DataSet[i]
        img_test.append(DataSets[i])
        lab_test.append(DataSet[i])

    print('Train::', img_train,'Test::', img_test)
    print('Train::', lab_train, 'Test::', lab_test)
    print('=====================================================================')

#
# for i in range(5):
#     print("######################## Interacao" , i, "######################")
#
#     for i in range(len(DataSets)):
#
#         C = DataSets.__getitem__(i)
#         DataSetsCopy.remove(C)
#
#         print('==========Train==========')
#
#         for D in DataSetsCopy:
#             print('Train', D)
#         print('Test', C)
#
#         DataSetsCopy = DataSets.copy();

