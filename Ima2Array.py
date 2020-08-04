# import the necessary packages
import imutils

import cv2



def preprocess(image, width, height, inter=cv2.INTER_AREA):
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

    return cv2.resize(image, (width, height),interpolation=inter)

img = cv2.imread('Teste.jpg')
img = preprocess(img, 256, 256)
cv2.imshow("Rotated", img)
cv2.waitKey(0)
cv2.imwrite("ddd.jpg", img)
