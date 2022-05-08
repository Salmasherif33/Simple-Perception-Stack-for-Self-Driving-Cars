import numpy as np
import cv2 as cv

def sliding_windows(img):
    xspan = img.shape[1] - 0
    yspan = img.shape[0] -0
    window_list = []
    for i in range (0,xspan , 64):
        for j in range (64*5,yspan , 64):
            window_list.append(((i, j), (i+64, j+64)))
    return window_list
    
def vis_windows(img, window_list):
    for window in window_list:
        img = cv.rectangle(img,(window[0][0],window[0][1]),(window[1][0],window[1][1]),(0,255,0), 2) 
    return img


