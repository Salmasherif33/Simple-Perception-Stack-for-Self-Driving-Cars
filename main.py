#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from edge_detection import canny
from bird_eye import *
from window import *
from lane_detection import *
from last_overlay import overlay
from calculations import *
def main():
    #usage: type(vid/img) PATH(relative or absolute)
    try:
        type_ = sys.argv[1]
        path = sys.argv[2]
        stages = int(sys.argv[3])
    except:
        if(len(sys.argv) < 4):
            print("USAGE:python3 main.py type(vid/img) PATH <stageNO(1..5)>")
            return 1


    if(type_ == "vid"):
        cv.waitKey(0)
        capture = cv.VideoCapture(path)



        istrue, frame = capture.read()      #istrue = true if there is a frame
        
        while istrue:
            if (stages == 1):
                output_img = canny(frame)
                
            elif(stages==2):
                output_img = canny(frame)
                warped,histogram,Minv = final_bird(output_img)
                left_fit,right_fit,left_lane_ends, right_lane_ends, visualization_data, slid_out, ploty,leftx,lefty,rightx,righty =sliding_window_polyfit(frame,warped)
                bird_draw = bird_draw_lane(warped,left_fit,right_fit)
                result = draw_lane(frame,output_img,left_fit,right_fit,Minv)   
                offset = center(warped , left_fit,right_fit,left_lane_ends,right_lane_ends)
                
                left_curvem,right_curvem = calculate_curvature(ploty,leftx,lefty,rightx,righty)
              
                result = overlay(left_curvem,right_curvem, offset,result,bird_draw,slid_out)
                result = bird_view_markings(warped)
            #cv.imshow('input_Video',slid_out)        
            cv.imshow('Output_Video',result)
            istrue, frame = capture.read()      #istrue = true if there is a frame
            if  cv.waitKey(20) & 0xFF == ord('e'):    #exit = e
                break


       


    elif (type_ == "img"):
        img = cv.imread(path)
        #img = canny(img)
        #output_img = lane_line_markings(img)
        #output_img,histogram = final_bird(img)
        output_img = canny(img)
        warped,histogram,Minv = final_bird(output_img)
        left_fit,right_fit,left_lane_inds, right_lane_inds, visualization_data, slid_out =sliding_window_polyfit(img,warped)
        result = draw_lane(img,output_img,left_fit,right_fit,Minv)        
        """ cv.imshow('warped',warped)   
        cv.imshow('sliding window',result)
        #cv.imshow('Output Image',result)
        cv.waitKey(0) """
        plt.imshow( warped)
        plt.show()




if __name__== "__main__":
    main()