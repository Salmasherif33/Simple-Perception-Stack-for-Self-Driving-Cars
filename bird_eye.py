import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    line_dst_offset = 200   #to bring left/right lines closer to each other (curves)
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[595, 452],
          [685, 452],
          [1110, img.shape[0]],
          [220, img.shape[0]]])

    dst = np.float32(
        [[src[3][0]+line_dst_offset, 0],
          [src[2][0] - line_dst_offset, 0],
          [src[2][0] - line_dst_offset, src[2][1]],
          [src[3][0]+ line_dst_offset, src[3][1] ]])


    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    
    binary_warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_LINEAR)
    ## HACKY way to make a black left vertical rectangle to eliminate the bad lines ##
    h = binary_warped.shape[0]
    w = binary_warped.shape[1]
    (cX, cY) = (w // 10 , h )
    binary_warped[0:cY , 0:cX] = 0 ##hard-setting all values at this rectangle to be zeroes 



    return binary_warped, Minv

def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    return histogram
def add_threshold(img):
    exampleImg_LThresh = hls_lthresh(exampleImg_unwarp, (min_thresh, max_thresh))



def bird_view_markings(img):
    width = img.shape[1]
    height = img.shape[0]
    padding = int(0.25 * width)

    dst = np.float32(
        [
            [padding,0],                # Top-left corner
            [padding,height],           # Bottom-left corner
            [width - padding, height],   # Bottom-right corner#
            [width - padding, 0]        # Top-right corner
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

