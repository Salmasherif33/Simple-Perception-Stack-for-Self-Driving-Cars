import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    line_dst_offset = 200   #to bring left/right lines closer to each other (curves)
    width = img.shape[1]
    height = img.shape[0]
    padding = int(0.25 * width)
    src = np.float32(
        [
            [600, 470],  # Top-left corner
            [300, 660],  # Bottom-left corner
            [1050, 684],  # Bottom-right corner
            [725, 470]  # Top-right corner
        ])

    padding = int(0.20 * width)

    dst = np.float32(
        [
            [padding+30,100],                # Top-left corner
            [padding+30,height -10],           # Bottom-left corner
            [width - (padding+100), height],   # Bottom-right corner#
            [width - (padding+100),50]        # Top-right corner
        ])


    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    
    binary_warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_LINEAR)
    ## HACKY way to make a black left vertical rectangle to eliminate the bad lines ##
    """ h = binary_warped.shape[0]
    w = binary_warped.shape[1]
    (cX, cY) = (w // 5, h )
    #binary_warped[0:cY , 0:cX] = 0 ##hard-setting all values at this rectangle to be zeroes 
    binary_warped[0:h , w-cX-70:w] = 0 ##hard-setting all values at this rectangle to be zeroes  """


    return binary_warped, Minv

def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    return histogram



def bird_view_markings(img):
    width = img.shape[1]
    height = img.shape[0]
    padding = int(0.20 * width)

    dst = np.float32(
        [
            [padding+30,50],                # Top-left corner
            [padding+30,height ],           # Bottom-left corner
            [width - (padding+100), height],   # Bottom-right corner#
            [width - (padding+120),50]        # Top-right corner
        ])


    detected_plot = cv.polylines(np.copy(img), np.int32([
        dst]), True, (147, 20, 255), 3)

    return detected_plot


def final_bird(img):
    warped , Minv = warp(img)
    histogram = get_histogram(warped)
    img = bird_view_markings(warped)
    ret,thresh1 = cv.threshold(img,0,255,cv.THRESH_BINARY)
    #plt.plot(histogram)
    #plt.show()
    
    return thresh1 , histogram , Minv

