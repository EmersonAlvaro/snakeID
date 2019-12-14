import pathlib
import os
import cv2 as cv



data_root = pathlib.Path('dataset/MozSnake')

num_class = 5
IMAGE_SIZE = 299
train_data = pathlib.Path('/content/drive/My Drive/MozSnake/Training/')  #Only for Google drive

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
