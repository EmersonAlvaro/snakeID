import pathlib
import os
import random
import cv2
import shutil
import imutils
import numpy as np

train_data = pathlib.Path('Dataset/MozSnake/train/001')
test_data = pathlib.Path('Dataset/MozSnake/test/001')

# train_dataD = pathlib.Path('Dataset/Mzsnake/train')
# train_dataOr = pathlib.Path('dataset/MozSnakeCopy/train')

IMAGE_SIZE = 256

Class = ['PuffAdder', 'BlackMamba', 'TwigSnake', 'Boomslang','SpitingCobra']

def creat_fold():

    for dr in train_data.glob('*/'):
        # pathSpecies =test_data.joinpath(dr.name)
        # pathlib.Path(pathSpecies).mkdir()
        files = os.listdir(dr)

        random.shuffle(files)

        qtd = round(len(files) * 0.2)

        for i in range(qtd):
            src = train_data.joinpath(dr.name).joinpath(files[i])
            dst = test_data.joinpath(dr.name).joinpath(files[i])
            os.rename(src, dst)

    return 1

creat_fold()
