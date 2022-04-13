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

    return binary_warped, Minv

def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    return histogram

def final_bird(img):
    warped , Minv = warp(img)
    histogram = get_histogram(warped)
    #plt.plot(histogram)
    #plt.show()
    
    return warped , histogram

