import pathlib
import os
import random
import cv2 as cv
import shutil
import numpy as np

train_data = pathlib.Path('Dataset/MozSnake/train')
test_data = pathlib.Path('Dataset/MozSnake/test')
IMAGE_SIZE = 224


splitperc = 0.15


def creat_fold():

    for dr in train_data.glob('*/'):
        pathSpecies =test_data.joinpath(dr.name)
        pathlib.Path(pathSpecies).mkdir()
        files = os.listdir(dr)

        random.shuffle(files)

        qtd = round(len(files) * splitperc)

        for i in range(qtd):
            src = train_data.joinpath(dr.name).joinpath(files[i])
            dst = test_data.joinpath(dr.name).joinpath(files[i])
            os.rename(src, dst)

    return 1

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
            img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            # cv.imwrite(f,img)
            images.append(img)
            labels.append(int(d))

    images, labels = np.asarray(images), np.asarray(labels)

    return images, labels

imagens, labels = load_data(str(train_data))
print("Helllo There")

# creat_fold();
