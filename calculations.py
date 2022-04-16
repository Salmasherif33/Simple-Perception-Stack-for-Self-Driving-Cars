#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt


def radius_and_center(img , left_fit,right_fit,left_lane_inds, right_lane_inds):
    ## radius calculation
    radius = "Radius Of Curvature = "



    radius += "15"
    radius += "m"

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
    return radius, offset