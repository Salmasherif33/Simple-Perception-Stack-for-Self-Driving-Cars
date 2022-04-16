#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt


def center(img , left_fit,right_fit,left_lane_inds, right_lane_inds):
  
  
    ## vehicle
    offset = "Vehicle is "
    if right_fit is not None and left_fit is not None:
        h = img.shape[0]
        w = img.shape[1]
        car_midpoint = w/2    #dividing width by 2 to get the center of img

        
        left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_midpoint = (right_fit_x_int + left_fit_x_int) /2
        center_offset = (car_midpoint - lane_midpoint) * (3.7/378) 
        center_offset = round(center_offset, 4)
        offset += str(center_offset) + "m from the center"
    
    lane_distance = left_fit - right_fit
    return offset

def calculate_curvature(ploty,leftx,lefty,rightx,righty):
    # Set the y-value where we want to calculate the road curvature.
    # Select the maximum y-value, which is the bottom of the frame.
    y_eval = np.max(ploty)
    # Pixel parameters for x and y dimensions
    YM_PER_PIX = 10.0 / 1000 # meters per pixel in y dimension
    XM_PER_PIX = 3.7 / 781 # meters per pixel in x dimension

    # Fit polynomial curves to the real world environment
    left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * (XM_PER_PIX), 2)
    right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * (XM_PER_PIX), 2)

    # Calculate the radii of curvature
    left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[
                    1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvem = ((1 + (2*right_fit_cr[
                    0]*y_eval*YM_PER_PIX + right_fit_cr[
                    1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curvem, right_curvem