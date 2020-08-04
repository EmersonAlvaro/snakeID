import pathlib
import os
import random
import cv2 as cv
import shutil
import numpy as np
from PIL import Image
import imutils

from skimage import exposure


train_data = pathlib.Path('dataset/MozSnake/train')
# test_data = pathlib.Path('Dataset/MozSnake/test')


IMAGE_SIZE = 256

Nomes = ['PuffAdder', 'BlackMamba', 'SpitingCobra', 'Boomslang']
i0 = 0
i1 = 0
i2 = 0
i3 = 0
i4 = 0

def data_augmentation(data_directory):

    global i0

    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    file_names = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names += [os.path.join(label_directory, f)
                       for f in os.listdir(label_directory)]

    for f in file_names:
        print(f)
        img = cv.imread(f)
        img = preprocess(img, IMAGE_SIZE, IMAGE_SIZE)

        # cv.imwrite(f, img)

        img1 = contrast_stretching(img)
        img_fv = flip(img1, vflip=True, hflip=False) #flip vertical
        save_image(img_fv, f, 'flipvert')

        img2 = HE(img)
        img2 *= 255
        save_image(img2, f, 'he')

        img_fh = flip(img, vflip=False, hflip=True) #flip vertical
        save_image(img_fh, f, 'fliphori')

        img3 = AHE(img)
        img3 *=255
        save_image(img3, f, 'ahe')

        print("Done it")


def save_image(img, pathfoto, type):

    global  i0, i1, i2, i3, i4

    dirname = os.path.split(os.path.dirname(pathfoto))[1]

    if int(dirname) == 0:
        i0 += 1
        cv.imwrite(os.path.dirname(pathfoto) + "/" + Nomes[int(dirname)]+str(i0) +type + ".jpg", img)
    if int(dirname) == 1:
        i1 += 1
        cv.imwrite(os.path.dirname(pathfoto) + "/" + Nomes[int(dirname)]+str(i1) +type + ".jpg", img)
    if int(dirname) == 2:
        i2 += 1
        cv.imwrite(os.path.dirname(pathfoto) + "/" + Nomes[int(dirname)]+str(i2) +type + ".jpg", img)
    if int(dirname) == 3:
        i3 += 1
        cv.imwrite(os.path.dirname(pathfoto) + "/" + Nomes[int(dirname)]+str(i3) +type + ".jpg", img)

def preprocess(image, width, height, inter=cv.INTER_AREA):
    width = width
    height = height
    inter = inter

    (h, w) = image.shape[:2]
    dW = 0
    dH = 0

    if w < h:
        image = imutils.resize(image, width=width, inter=inter)
        dH = int((image.shape[0] - height) / 2.0)
    else:
        image = imutils.resize(image, height=height,inter=inter)
        dW = int((image.shape[1] - width) / 2.0)

    (h, w) = image.shape[:2]
    image = image[dH:h - dH, dW:w - dW]

    return cv.resize(image, (width, height),interpolation=inter)

def rotate(image, angle):

    rows, cols, c = image.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv.warpAffine(image, M, (cols, rows))

    return image


def flip (image, vflip=False, hflip=False):

    if hflip or vflip:
        if hflip and vflip:
            c = -1
        else:
            c = 0 if vflip else 1
        image = cv.flip(image, flipCode=c)
    return image

def contrast (image,ksize):

    image = cv.Sobel(image,cv.CV_64F,1,0,ksize=ksize)

    return image

def averageing_blur(image,shift):
    image=cv.blur(image,(shift,shift))
    return image

def erosion_image(image,shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv.erode(image,kernel,iterations = 1)
    return image

def dilation_image(image,shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv.dilate(image,kernel,iterations = 1)
    return image

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv.LUT(image, table)


# Contrast stretching
def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

# Histogram equalization
def HE(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq

# Adaptive histogram equalization
def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq

# data_augmentation(train_data)
# print("Helllo There")
#

