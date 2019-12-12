import cv2 as cv
import glob
import os
import random
from snake.config import *


def datasplit () :
    for dr in all_data.iterdir():
        trainper =0
        testper=0
        validationper=0
        #
        # print('---------------------------------------')
        files = os.listdir(dr)
        # print(files)
        random.shuffle(files)
        # print(files)

        for i in range(len(files)):
            photoOrigPath = all_data.joinpath(str(dr.name)).joinpath(files[i])
            # if(os.path.getsize(photoOrigPath)>0):
            print(photoOrigPath)
            img = cv.imread(str(photoOrigPath), 1)              #Resize da imagens para 200 *200
            orig = cv.resize(img,(IMAGE_SIZE,IMAGE_SIZE))

            trainper = round(len(files) * TrainPer)             #Cotacao de dados para treinamento
            testper = round(len(files) * TestPer)               #Cotacao de dados para teste
            validationper = round(len(files) * ValidationPer)   #Cotacao de dados para validacao
            total = testper+trainper+validationper;

            if total < len(files):
                trainper=trainper+1
            elif total > len(files):
                trainper = trainper -1

            total = testper + trainper + validationper;

            if i < trainper:                                #Datasplit acording with Train, Test and Validation Quot
                # print('train', files[i])
                photoNewPath=train_data.joinpath(str(dr.name))
                # print(photoNewPath)
                cv.imwrite(str(photoNewPath) + "/" + str(dr.name) + "Original-" + str(i) + ".jpg", orig)
            elif i >= trainper and i < (testper + trainper) :
                # print('test', files[i])
                photoNewPath=test_data.joinpath(str(dr.name))
                # print(photoNewPath)
                cv.imwrite(str(photoNewPath) + "/"+str(dr.name) +"Original-" + str(i) + ".jpg", orig)
            elif i >= (testper + trainper) :
                # print('validation', files[i])
                photoNewPath=validation_data.joinpath(str(dr.name))
                # print(photoNewPath)
                cv.imwrite(str(photoNewPath) + "/"+str(dr.name) +"Original-" + str(i) + ".jpg", orig)

datasplit()


# def imagens():
#     for dr in all_data.iterdir():
#         for file in dr.iterdir():
#             print(str(file))
#             # img = cv.imread(str(file), 1)
#             # orig = cv.resize(img,(heights,width))
#             # cv.imshow("imagen", orig)
#             # cv.waitKey(0);
#             # cv.destroyAllWindows()
#             # print(file.name)
#             # # photoNewPath=test_data.joinpath(str(dr.name)).joinpath(str(file.name))
#             # # print(photoNewPath)
#             # # cv.imwrite(str(photoNewPath),orig)
#
# # imagens()