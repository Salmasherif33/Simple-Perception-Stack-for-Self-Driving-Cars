import os
import glob
import cv2

def load(path):
    cars = []
    not_cars =[]

    for img in glob.glob(path + "/vehicles/GTI_Far/*.png"):
        GTI= cv2.imread(img)
        cars.append(GTI)
    for img in glob.glob(path + "/vehicles/GTI_Left/*.png"):
        GTI= cv2.imread(img)
        cars.append(GTI)
    for img in glob.glob(path + "/vehicles/GTI_MiddleClose/*.png"):
        GTI= cv2.imread(img)
        cars.append(GTI)
    for img in glob.glob(path + "/vehicles/GTI_Right/*.png"):
        GTI= cv2.imread(img)
        cars.append(GTI) 
    for img in glob.glob(path + "/vehicles/KITTI_extracted/*.png"):
        GTI= cv2.imread(img)
        cars.append(GTI) 

    ## NOT_CARS ARRAY ##
    for img in glob.glob(path + "/non-vehicles/GTI/*.png"):
        GTI= cv2.imread(img)
        not_cars.append(GTI)
    for img in glob.glob(path + "/non-vehicles/Extras/*.png"):
        Extra= cv2.imread(img)
        not_cars.append(Extra)


    return cars ,not_cars