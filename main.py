#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob
from pickle import load
import matplotlib.pyplot as plt
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
        load_path = "/home/anto/Downloads/"
        net ,classes ,output_layers,colors = load_yolo(load_path, load_path ,load_path)
        '''
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0
        windows_list = sliding_windows(img)
        '''
        c = 0
        
        while istrue:
            if c % 5 == 0:
                c+=1
                istrue, frame = capture.read()
                continue
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
                result = debug_overlay(result,bird_draw,slid_out,warped,lane_markings,bbox,labels,colors)
            elif (mode ==0):
                result = last_overlay(left_curvem,right_curvem, offset,result,bbox,labels,colors)
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



    elif (type_ == "img"):
        img =  cv.imread(path)
        
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
        ##YOLO
        
        load_path = "/home/anto/Downloads/"
        net ,classes ,output_layers,colors = load_yolo(load_path, load_path ,load_path)
        class_ids , boxes , confidences  = detect(img,net,output_layers)
        bbox,labels  =  vis(img,class_ids , boxes, confidences,classes,colors)
        
        if(mode == 1):
            result = debug_overlay(result,bird_draw,slid_out,warped,lane_markings,bbox,labels,colors)
        elif (mode ==0):
            result = last_overlay(left_curvem,right_curvem, offset,result,bbox,labels,colors)
        else:
            print("ERROR:No valid debug_mode was given")
        

        ## PHASE II ##
        '''
        img = img.astype(np.float32)/255.0
        
        
        ## TRAIN SVM MODEL ##
        cars,not_cars = load_imgs(train_path)
        hogged_car, car_features = extract_features(cars[0:4000],'ALL')
        hogged_not_car, not_car_features = extract_features(not_cars[0:4000],'ALL' )
        svc , X_scaler = train(car_features,not_car_features)        
        ##END OF TRAINING
        
        ## LOAD SVM 
        svc = load(open('model.pkl', 'rb'))
        X_scaler = load(open('scaler.pkl', 'rb'))
        
        

        
        windows_list = sliding_windows(img)
        hot_windows = search_windows(img, windows_list, svc, X_scaler)
        result = vis_windows(cv.cvtColor(img, cv.COLOR_RGB2BGR),hot_windows)
        '''
        cv.imshow('Output Image',result)
        cv.waitKey(0)
        


if __name__== "__main__":
    main()