import os
import glob
import cv2

def load(path):
    cars = []
    not_cars =[]
    basedir= path + "/vehicles/"
    image_types = os.listdir(basedir)
    for imtype in image_types:
        cars.extend(glob.glob(basedir+imtype+'/*'))
    '''
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
    '''
    ## NOT_CARS ARRAY ##
    basedir=path + "/non-vehicles/"
    image_types = os.listdir(basedir)
    for imtype in image_types:
        not_cars.extend(glob.glob(basedir+imtype+'/*'))
    '''
    for img in glob.glob(path + "/non-vehicles/Extras/*.png"):
        Extra= cv2.imread(img)
        not_cars.append(Extra)
    '''
    
    return cars ,not_cars


def save(cars,not_cars):
    with open("cars.txt", 'w') as f:
        for car in cars:
            f.write(str(car)+'\n')

    with open("notcars.txt", 'w') as f:
        for notcar in not_cars:
            f.write(str(notcar)+'\n')


def load_features():
    cars = []
    not_cars = []
    i = 0
    j=0
    with open("cars.txt") as fp:
        for line in fp:
            list_1 = line.split()
            cars.append(list_1)
            
    with open("notcars.txt") as f:
        for line in f:
            list_1 = line.split()
            cars.append(list_1)
             
    return cars,not_cars