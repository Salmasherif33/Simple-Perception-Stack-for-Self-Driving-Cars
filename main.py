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
from last_overlay import *
from calculations import *
from phase2.load import *
from phase2.features import *
def main():
    #usage: type(vid/img) PATH(relative or absolute)
    try:
        type_ = sys.argv[1]
        path = sys.argv[2]
        destination = sys.argv[3]
        train_path = sys.argv[4]
        mode_reduntant = sys.argv[5]
        mode = int(sys.argv[6])
    except:
        if(len(sys.argv) < 6):
            print("USAGE:python3 main.py type(vid/img) PATH outputPATH train_path mode 0/1")
            return 1


    if(type_ == "vid"):
        capture = cv.VideoCapture(path)
        i = 0
        istrue, frame = capture.read()      #istrue = true if there is a frame
        height, width, layers = frame.shape
        size = (width,height)
        out = cv.VideoWriter(destination,cv.VideoWriter_fourcc(*"mp4v"), 25, size)  
        img_array = []
        while istrue:
            output_img = canny(frame)
            warped,histogram,Minv = final_bird(output_img)
            left_fit,right_fit,left_lane_ends, right_lane_ends, visualization_data, slid_out, ploty,leftx,lefty,rightx,righty =sliding_window_polyfit(frame,warped)
            bird_draw = bird_draw_lane(warped,left_fit,right_fit)
            result = draw_lane(frame,output_img,left_fit,right_fit,Minv)   
            offset = center(warped , left_fit,right_fit,left_lane_ends,right_lane_ends)
            lane_markings = lane_line_markings(frame)
            left_curvem,right_curvem = calculate_curvature(ploty,leftx,lefty,rightx,righty)

            if(mode == 1):
                result = debug_overlay(result,bird_draw,slid_out,warped,lane_markings)
            elif (mode ==0):
                result = last_overlay(left_curvem,right_curvem, offset,result)
            else:
                print("ERROR:No valid debug_mode was given")
                break

            out.write(result)

            #cv.imshow('input_Video',slid_out)        
            #cv.imshow('Output_Video',result)
            istrue, frame = capture.read()      #istrue = true if there is a frame
            #if  cv.waitKey(20) & 0xFF == ord('e'):    #exit = e
            #    break
        out.release()



    elif (type_ == "img"):
        img = cv.imread(path)
        #img = canny(img)
        #output_img = lane_line_markings(img)
        #output_img,histogram = final_bird(img)
        output_img = canny(img)
        warped,histogram,Minv = final_bird(output_img)
        left_fit,right_fit,left_lane_ends, right_lane_ends, visualization_data, slid_out, ploty,leftx,lefty,rightx,righty =sliding_window_polyfit(img,warped)
        bird_draw = bird_draw_lane(warped,left_fit,right_fit)
        result = draw_lane(img,output_img,left_fit,right_fit,Minv)     
        offset = center(warped , left_fit,right_fit,left_lane_ends,right_lane_ends)
        lane_markings = lane_line_markings(img)
        left_curvem,right_curvem = calculate_curvature(ploty,leftx,lefty,rightx,righty)
        if(mode == 1):
            result = debug_overlay(result,bird_draw,slid_out,warped,lane_markings)
        elif (mode ==0):
            result = last_overlay(left_curvem,right_curvem, offset,result)
        else:
            print("ERROR:No valid debug_mode was given")



        ## PHASE II ##
        cars,not_cars = load(train_path)
        cv2.imshow("not car",not_cars[833])




    


        #cv.imshow('Output Image',result)
        cv.waitKey(0)
    




if __name__== "__main__":
    main()