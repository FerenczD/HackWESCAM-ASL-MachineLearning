import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import cv2 as cv



def define_Picture(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    plt.imshow(gray)
    plt.show()


