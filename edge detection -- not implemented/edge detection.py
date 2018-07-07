import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from skimage import transform
from skimage.color import rgb2gray
import cv2 as cv

def define_Picture(img):
    string = 0
    maxValue = 34
    image3 = 0

    for i in range (maxValue):
        string = str(i)
        #filename = "C:/Users/Peter/Downloads/Thumbs/Training/png/" + string + ".jpg"
       # img = cv.imread(filename)


        mask = cv.imread("C:/Users/Peter/Downloads/Thumbs/Training/png/download.jpg", 0)
        lower = np.array([0, 30, 40], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")

        image3 = cv.resize(img,(512, 512))
         #plt.show()

        converted = (cv.cvtColor(image3, cv.COLOR_BGR2HSV))
        skinmask = cv.inRange(converted, lower, upper)
        res = cv.bitwise_and(image3, image3, mask = skinmask)
        plt.imshow(res)
        plt.show()

        ret,thresh  = cv.threshold(res,250,255,cv.THRESH_BINARY,0)

        plt.imshow(thresh)

        #plt.show()

        res = cv.Canny(res, 300, 100, 0)
        test = np.array(res)
        flatt = test.flatten()
        print("The flattened value, {}" .format(flatt))
        print(test)
        plt.imshow(res)
        plt.show()


