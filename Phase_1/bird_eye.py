import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from edge_detection import *


def warp(img):
    img_size = (img.shape[1], img.shape[0])  # width, height
    width = img.shape[1]
    height = img.shape[0]
    padding = int(0.25 * width)
    src = np.float32(
        [
            (572, 454),  # Top-left corner
            (250, 660),  # Bottom-left corner
            (1100, 684),  # Bottom-right corner
            (734, 454)  # Top-right corner
        ])

    dst = np.float32(
        [
            [padding,0],                # Top-left corner
            [padding,height],           # Bottom-left corner
            [width - padding, height],   # Bottom-right corner#
            [width - padding, 0]        # Top-right corner
        ])

    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)

    binary_warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_LINEAR)

    return  binary_warped, Minv


def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]//2):, :], axis=0)
    return histogram


def final_bird(img):

    warped, Minv = warp(img)
   
    histogram = get_histogram(warped)
    

    return warped, histogram

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

