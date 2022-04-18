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


def overlay(left_curvem,right_curvem, text2 , img ,bird_draw,slid_out):
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
    
    bird_draw = rescale(bird_draw, 0.2)
    x_offset= int(img.shape[0]) + 20
    y_offset=20
    img[y_offset:y_offset+bird_draw.shape[0], x_offset:x_offset+bird_draw.shape[1]] = bird_draw



    slid_out = rescale(slid_out, 0.2)
    x_offset= int(img.shape[0]) + 300
    y_offset=20
    img[y_offset:y_offset+slid_out.shape[0], x_offset:x_offset+slid_out.shape[1]] = slid_out
    return img