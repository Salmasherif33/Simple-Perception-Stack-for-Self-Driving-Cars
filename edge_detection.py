import sys
import cv2 as cv
import numpy as np
import glob



def gaussian_blur(img, kernel_size):
    return cv.gaussian_blur(img,(kernel_size, kernel_size),0)

#define low and high threshold percentage for the image to make the lanes clearer
def r_threshold(img, th1, th2):
    mini = np.min(img)  #get the minimum pixel value in the image 5
    maxi = np.max(img)  #get the maximum pixel value in the image 200

    low = mini + (maxi- mini)*th1   #the actual threshold value (the lowest)
    high = mini + (maxi - mini)*th2 #the highest

    #convert it to binary image 
    return np.uint8((img >= low) & (img <= high))*255

def l_threshold(img, th1, th2):
    #use the actual threshold value directly
    return np.uint8((img >= th1) & (img <= th2))*255

def grayscale(img):
   
    #convert img to HLS and HLV scale
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    v = hsv[:,:,2]

    #use Lightening(L) channel : highlight the right lane more(white lane)
    r_lane = r_threshold(l, 0.75, 2)    
    r_lane[:,:700] = 0  #make the left half of the image == 0
    
    #use H & s channel to helight the left lane(yellow one)
    l_lane = l_threshold(h, 30,100) 
    l_lane &= r_threshold(s, 0.1,1) 
   
   #make the right half of the image == 0 , and the most left part in the imag
    l_lane[:, 700:] = 0
    l_lane[:, :200] = 0
    return ( l_lane | r_lane)   #combine them

def threshold(img):
    cann_thresh = (0,1)
    binary = np.zeros_like(img)
    binary[(img > cann_thresh[0]) & (img < cann_thresh[1])] = 1

    return binary


def canny(img):
    img = grayscale(img)
    return img


