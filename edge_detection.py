import sys
import cv2 as cv
import numpy as np
import glob



def gaussian_blur(img, kernel_size):
    return cv.gaussian_blur(img,(kernel_size, kernel_size),0)


def grayscale(img):
    #returning the saturation part only (could by L)
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    return s

def threshold(img):
    cann_thresh = (0,1)
    binary = np.zeros_like(img)
    binary[(img > cann_thresh[0]) & (img < cann_thresh[1])] = 1

    return binary


def canny(img):
    img = grayscale(img)
    img = cv.Canny(img,150,200, L2gradient = True) 
    return img


