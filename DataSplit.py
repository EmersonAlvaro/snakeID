import pathlib
import os
import random
import cv2 as cv
import shutil

data_root = pathlib.Path('dataroot')
all_data = pathlib.Path('dataroot/alldata')
train_data = pathlib.Path('dataroot/traindata')
test_data = pathlib.Path('dataroot/testdata')
validation_data =pathlib.Path('dataroot/validation_data')
TRAIN_K_DATA = pathlib.Path('dataroot/TRAIN_K_DATA')

IMAGE_SIZE = 224



def creat_fold(k):

    if TRAIN_K_DATA.exists():
        shutil.rmtree(TRAIN_K_DATA)

    if not TRAIN_K_DATA.exists():
        TRAIN_K_DATA.mkdir()


    pathFolders = []

    for i in range(k):
        fold = 'Fold' + str(i)
        pathfold = TRAIN_K_DATA.joinpath(fold)
        pathlib.Path(pathfold).mkdir()

        pathFolders.append(pathfold)

        for dr in all_data.glob('*/'):
            pathSpecies =pathfold.joinpath(dr.name)
            pathlib.Path(pathSpecies).mkdir()
            files = os.listdir(dr)

    return pathFolders
# creat_fold(5)

def split_data (k):

    pathFolders = creat_fold(k)

    for dr in all_data.iterdir():

        files = files = os.listdir(dr)

        random.shuffle(files)

        QtdOriginal =round( len(files)/ k)
        fold = 0

        Qtd = round(QtdOriginal);

        directories = [os.path.join(TRAIN_K_DATA,d)
                       for d in os.listdir(TRAIN_K_DATA)
                       if os.path.isdir(os.path.join(TRAIN_K_DATA, d))]

        # print('====================')
        # print(len(files))
        # print(round(QtdOriginal))


        for i in range(len(files)):

            photoOrigPath = all_data.joinpath(str(dr.name)).joinpath(files[i])

            if i ==Qtd and i <= len(files):
                if fold == k-1:
                    fold = fold
                else:
                    fold=fold +1
                Qtd = Qtd+QtdOriginal
            # print('fold', fold, 'index====', i)

            # print(os.path.join(directories[fold], files[i]))
            # f = os.path.join()
            # if os.path.getsize(photoOrigPath) != 0:
            # print(photoOrigPath)
            f=os.path.join(directories[fold], dr.name, files[i])

            img = cv.imread(str(photoOrigPath), 1)
            img = cv.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
            cv.imwrite(f,img)

        fold = 0

    return pathFolders

# split_data(5)
