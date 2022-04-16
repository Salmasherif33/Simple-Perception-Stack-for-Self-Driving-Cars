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

def overlay(text1,text2 , img ,slid_out):
    origin1 = (50,50)
    origin2 = (50,100)
    font = cv.FONT_HERSHEY_SIMPLEX
    fonts = 1
    thick = 3
    color = (255,255,255)
    img = cv.putText(img , text1 , origin1, font , fonts , color , thick)
    img = cv.putText(img , text2 , origin2, font , fonts , color , thick)
    
    slid_out = rescale(slid_out, 0.2)
    #small_w = int(slid_out.shape[1])
    #small_h=  int(slid_out.shape[0])
    x_offset= int(img.shape[0]) + 300
    y_offset=20
    img[y_offset:y_offset+slid_out.shape[0], x_offset:x_offset+slid_out.shape[1]] = slid_out
    return img