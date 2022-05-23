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

def last_overlay(left_curvem,right_curvem, text2 , img , bbox , labels ,colors):
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
    for i, label in enumerate(labels):
        color = colors[i]
        cv.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)
        cv.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def debug_overlay(img , left_curvem,right_curvem,text2,bird_draw,slid_out,warped,lane_markings,bbox,labels,colors):

    ## darkening background 
    mask = cv.rectangle(np.copy(img), (0, 0), (img.shape[1], 220), (0, 0, 0), thickness=cv.FILLED)
    img = cv.addWeighted(src1=mask, alpha=0.3, src2=img, beta=0.8, gamma=0)


    origin1 = (20,190)
    origin2 = (20,210)
    font = cv.FONT_HERSHEY_SIMPLEX
    fonts = 0.5
    thick = 1
    color = (255,255,255)
    text1 = 'Radius of Curvature = '
    text1 += str((left_curvem+right_curvem)/2)[:7]+' m'

    img = cv.putText(img , text1 , origin1, font , fonts , color , thick)
    img = cv.putText(img , text2 , origin2, font , fonts , color , thick)
    bird_draw = rescale(bird_draw, 0.2)
    x_offset= int(img.shape[0]) + 40
    y_offset=5
    img [5:5+50 , 200:200+20 ]
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


    for i, label in enumerate(labels):
        color = colors[i]
        thumbnail = cv.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)
        cv.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    thumb_w=100 
    thumb_h=80
    off_x=10
    off_y=30
    ##([x, y, w, h])
    #int(x), int(y), int(x+w), int(y+h)
    #     0,0  0,1     1,0  1,1
    ## [ [ x-w,y-h]  [ x+w,y+h] ] 
    cv.putText(img, 'Detected Vehicles', (20,37), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv.LINE_AA)
    if bbox:
        for i, bbox in enumerate(bbox):
            thumbnail = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] 
            if thumbnail.any():
                vehicle_thumb = cv.resize(thumbnail, dsize=(thumb_w, thumb_h))
                start_x = 50 + (i+1) * off_x + i * thumb_w
                img[off_y + 30:off_y + thumb_h + 30, start_x:start_x + thumb_w, :] = vehicle_thumb

    return img