import sys
import cv2 as cv
import numpy as np
import glob



def gaussian_blur(img, kernel_size):
    return cv.gaussian_blur(img,(kernel_size, kernel_size),0)

def r_threshold(img, th1, th2):
    mini = np.min(img)
    maxi = np.max(img)

    low = mini + (maxi- mini)*th1
    high = mini + (maxi - mini)*th2
    return np.uint8((img >= low) & (img <= high))*255

def l_threshold(img, th1, th2):
    return np.uint8((img >= th1) & (img <= th2))*255

def grayscale(img):
    #returning the saturation part only (could by L)
    #gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    v = hsv[:,:,2]

    """  r_lane = r_threshold(l, 0.8, 1.0)
    r_lane[:,750] = 0

    l_lane = l_threshold(s, 20,30)
    l_lane &= r_threshold(v, 0.7,1.0)

    output = l_lane | r_lane """
    r_lane = r_threshold(l, 0.75, 2)
    r_lane[:,:700] = 0
    
    l_lane = l_threshold(h, 30,100)
    l_lane &= r_threshold(s, 0.1,1)
   
    l_lane[:, 700:] = 0
    l_lane[:, :200] = 0
    return ( l_lane | r_lane)

def threshold(img):
    cann_thresh = (0,1)
    binary = np.zeros_like(img)
    binary[(img > cann_thresh[0]) & (img < cann_thresh[1])] = 1

    return binary


def canny(img):
    img = grayscale(img)

    #img = cv.Canny(img,100,255, L2gradient = True) 

    
    #img = threshold(img)
    return img


def lane_line_markings(img):
    src = np.float32(
        [[595, 452],
          [685, 452],
          [1110, img.shape[0]],
          [220, img.shape[0]]])

    detected_plot = cv.polylines(np.copy(img), np.int32([
        src]), True, (147, 20, 255), 3)

    return detected_plot
