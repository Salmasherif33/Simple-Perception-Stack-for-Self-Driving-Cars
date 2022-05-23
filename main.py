#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob
from pickle import load
import matplotlib.pyplot as plt
import time
from edge_detection import canny
from bird_eye import final_bird
from window import *
from lane_detection import *
from last_overlay import *
from calculations import *
from phase2.load import *
from phase2.features import *
from phase2.windows import *
from phase2.svm import *
from YOLO.detect import *

def main():
    #usage: type(vid/img) PATH(relative or absolute)
    try:
        type_ = sys.argv[1]
        path = sys.argv[2]
        destination = sys.argv[3]
        mode_reduntant = sys.argv[4]
        mode = int(sys.argv[5])
    except:
        if(len(sys.argv) < 5):
            print("USAGE:python3 main.py type(vid/img) PATH outputPATH train_path mode 0/1")
            return 1


    if(type_ == "vid"):
        capture = cv.VideoCapture(path)
        i = 0
        istrue, frame = capture.read()      #istrue = true if there is a frame
        height, width, layers = frame.shape
        size = (width,height)
        out = cv.VideoWriter(destination,cv.VideoWriter_fourcc(*"mp4v"),22, size)  
        img_array = []
        load_path = ""
        net ,classes ,output_layers,colors = load_yolo(load_path, load_path ,load_path)
        c = 0
        start_time = time.time()

        while istrue:
            output_img = canny(frame)
            warped,histogram,Minv = final_bird(output_img)
            left_fit,right_fit,left_lane_ends, right_lane_ends, visualization_data, slid_out, ploty,leftx,lefty,rightx,righty =sliding_window_polyfit(frame,warped)
            bird_draw = bird_draw_lane(warped,left_fit,right_fit)
            result = draw_lane(frame,output_img,left_fit,right_fit,Minv)   
            offset = center(warped , left_fit,right_fit,left_lane_ends,right_lane_ends)
            lane_markings = lane_line_markings(frame)
            left_curvem,right_curvem = calculate_curvature(ploty,leftx,lefty,rightx,righty)
            c+=1
            # YOLO
            class_ids , boxes , confidences  = detect(frame,net,output_layers)
            bbox,labels  =  vis(frame,class_ids , boxes, confidences,classes,colors)
            if(mode == 1):
                result = debug_overlay(result,left_curvem,right_curvem,offset,bird_draw,slid_out,warped,lane_markings,bbox,labels,colors)
            elif (mode ==0):
                result = last_overlay(left_curvem,right_curvem,offset,result,bbox,labels,colors)
            else:
                print("ERROR:No valid debug_mode was given")
                break
            out.write(result)
            #cv.imshow('Output_Video',result)
            '''

            
            hot_windows = search_windows(img, windows_list, svc, X_scaler)
            result = vis_windows(frame,hot_windows)
            out.write(result)
            cv.imshow('Output_Video',result)
            
            '''
            istrue, frame = capture.read()      #istrue = true if there is a frame
            #if  cv.waitKey(20) & 0xFF == ord('e'):    #exit = e
            #    break
        out.release()
        print("Video total time =  %s seconds " % (time.time() - start_time))




    elif (type_ == "img"):
        start_time = time.time()
        img =  cv.imread(path)
        output_img = canny(img)
        warped,histogram,Minv = final_bird(output_img)
        left_fit,right_fit,left_lane_ends, right_lane_ends, visualization_data, slid_out, ploty,leftx,lefty,rightx,righty =sliding_window_polyfit(img,warped)
        
        bird_draw = bird_draw_lane(warped,left_fit,right_fit)
        result = draw_lane(img,output_img,left_fit,right_fit,Minv)     
        offset = center(warped , left_fit,right_fit,left_lane_ends,right_lane_ends)
        lane_markings = lane_line_markings(img)
        left_curvem,right_curvem = calculate_curvature(ploty,leftx,lefty,rightx,righty)
        ##YOLO
        
        load_path = ""
        net ,classes ,output_layers,colors = load_yolo(load_path, load_path ,load_path)
        class_ids , boxes , confidences  = detect(img,net,output_layers)
        bbox,labels  =  vis(img,class_ids , boxes, confidences,classes,colors)
        
        if(mode == 1):
            result = debug_overlay(result,left_curvem,right_curvem,offset,bird_draw,slid_out,warped,lane_markings,bbox,labels,colors)
        elif (mode ==0):
            result = last_overlay(left_curvem,right_curvem, offset,result,bbox,labels,colors)
        else:
            print("ERROR:No valid debug_mode was given")
        
        print("Image total time =  %s seconds " % (time.time() - start_time))
        cv.imwrite(destination,result)
        cv.imshow('Output Image',result)
        cv.waitKey(0)
        


if __name__== "__main__":
    main()