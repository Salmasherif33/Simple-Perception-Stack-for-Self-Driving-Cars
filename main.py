#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from edge_detection import canny
from bird_eye import final_bird
from window import slide_window


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
                warped,histogram = final_bird(output_img)
                ploty,left_fitx,right_fitx, left_fit, right_fit,out_img = slide_window(warped, histogram)
                out_img = warped     #only for temp view
            elif (stages==3):
                ##TODO:: lane detection on the straight lane
            elif(stages==4):
                ##TODO:: show lane detection + vehicle position + curvature radius 
            

            cv.imshow('Output_Video',warped)
            istrue, frame = capture.read()      #istrue = true if there is a frame
            if  cv.waitKey(20) & 0xFF == ord('e'):    #exit = e
                break


       


    elif (type_ == "img"):
        img = cv.imread(path)
        img = canny(img)
        output_img,histogram = final_bird(img)
        cv.imshow('Output Image',output_img)
        cv.waitKey(0)




if __name__== "__main__":
    main()