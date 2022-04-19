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

    y_eval = np.max(ploty)
    ym = 10.0 / 600 # assumption --> each 60 pixel = 1 m 
    xm = 3.7 / 781 

    # Fit polynomial curves to the real world environment
    left_fit_cr = np.polyfit(lefty * ym, leftx * (xm), 2)
    right_fit_cr = np.polyfit(righty * ym, rightx * (xm), 2)

    # Calculate the radii of curvature
    left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*ym + left_fit_cr[
                    1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvem = ((1 + (2*right_fit_cr[
                    0]*y_eval*ym + right_fit_cr[
                    1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curvem, right_curvem
