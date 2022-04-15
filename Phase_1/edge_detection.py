import sys
import cv2 as cv
import numpy as np
import glob

def gaussian_blur(img, kernel_size):
    return cv.gaussian_blur(img, (kernel_size, kernel_size), 0)


def grayscale(img):
    # returning the saturation part only (could by L)
    #gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    return s


def threshold(img):
    cann_thresh = (0, 1)
    binary = np.zeros_like(img)
    binary[(img > cann_thresh[0]) & (img < cann_thresh[1])] = 1

    return binary


def canny(img):
    img = grayscale(img)
    #img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.Canny(img, 150, 200,  L2gradient = True)
    return img


def lane_line_markings(img):
    src = np.float32(
        [
            (572, 454),  # Top-left corner
            (250, 660),  # Bottom-left corner
            (1100, 684),  # Bottom-right corner
            (734, 454)  # Top-right corner
        ])

    detected_plot = cv.polylines(np.copy(img), np.int32([
        src]), True, (147, 20, 255), 3)

    return detected_plot
