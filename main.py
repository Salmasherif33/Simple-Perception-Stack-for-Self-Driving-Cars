#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from edge_detection import canny
from bird_eye import final_bird
from window import *
from lane_detection import *

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
                
                """
                # Getting the left lane from the 1st sliding window fn and the right one from the second
                # bad bad idea but works :)
                """

                ploty,left_fitx,s1_right_fitx, s1_left_fit, s1_right_fit,out_img = slide_window(warped, histogram)
                s2_left_fit, s2_right_fit, s2_left_lane_inds, s2_right_lane_inds, s2_visualization_data, slid_out =sliding_window_polyfit(frame,warped)
                
                result = draw_lane(frame,output_img,s1_left_fit,s2_right_fit,Minv)     
            
            cv.imshow('input_Video',out_img)        
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
        ploty,left_fitx,s1_right_fitx, s1_left_fit, s1_right_fit,out_img = slide_window(warped, histogram)
        s2_left_fit, s2_right_fit, s2_left_lane_inds, s2_right_lane_inds, s2_visualization_data, slid_out =sliding_window_polyfit(img,warped)
        result = draw_lane(img,output_img,s1_left_fit,s2_right_fit,Minv)        
        cv.imshow('warped',warped)   
        cv.imshow('sliding window',result)
        #cv.imshow('Output Image',result)
        cv.waitKey(0)




if __name__== "__main__":
    main()