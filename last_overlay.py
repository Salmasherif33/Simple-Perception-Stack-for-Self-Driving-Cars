#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

def rescale(img,scale):
    w = int (img.shape[1]*scale)
    h = int(img.shape[0]*scale)
    dimensions = (w,h)

    return cv.resize(img,dimensions, interpolation=cv.INTER_AREA)

def last_overlay(left_curvem,right_curvem, text2 , img ):
    origin1 = (50,50)
    origin2 = (50,100)
    font = cv.FONT_HERSHEY_SIMPLEX
    fonts = 1
    thick = 3
    color = (255,255,255)
    text1 = 'Radius of Curvature = '
    text1 += str((left_curvem+right_curvem)/2)[:7]+' m'

    img = cv.putText(img , text1 , origin1, font , fonts , color , thick)
    img = cv.putText(img , text2 , origin2, font , fonts , color , thick)
    
    return img


def debug_overlay(img ,bird_draw,slid_out,warped,lane_markings):
    bird_draw = rescale(bird_draw, 0.2)
    x_offset= int(img.shape[0]) + 40
    y_offset=5
    img[y_offset:y_offset+bird_draw.shape[0], x_offset:x_offset+bird_draw.shape[1]] = bird_draw



    slid_out = rescale(slid_out, 0.2)
    x_offset= int(img.shape[0]) + 300
    y_offset=5
    img[y_offset:y_offset+slid_out.shape[0], x_offset:x_offset+slid_out.shape[1]] = slid_out

    warped =  cv.cvtColor(warped,cv.COLOR_GRAY2BGR)
    warped = rescale(warped, 0.2)
    x_offset= int(img.shape[0]) + 300
    y_offset=150
    img[y_offset:y_offset+warped.shape[0], x_offset:x_offset+warped.shape[1]] = warped


  
    lane_markings = rescale(lane_markings, 0.2)
    x_offset= int(img.shape[0]) + 40
    y_offset=150
    img[y_offset:y_offset+lane_markings.shape[0], x_offset:x_offset+lane_markings.shape[1]] = lane_markings


    return img