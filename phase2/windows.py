import numpy as np
import cv2 as cv
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from features import *
def sliding_windows(img,overlap=(0.5, 0.5)):
    window_list = []     
    xspan = img.shape[1] -0
    yspan = 660 - 360

    ##compute number of windows in x&y directions
    x_windows = np.int(xspan/np.int((64,64)[0]*(0.5))) - 1
    y_windows = np.int(yspan/np.int((64,64)[1]*(0.5))) - 1

    for ys in range(y_windows):
        for xs in range(x_windows):
            # Calculate x&y of top left corner of the rectangle
            i = xs*np.int((64,64)[0]*(0.5))
            j = ys*np.int((64,64)[1]*(0.5)) + 360
    
            window_list.append(((i, j), (i+ (64,64)[0],(j + (64,64)[1]))))
    return window_list
    
def vis_windows(img, window_list):
    for window in window_list:
        img = cv.rectangle(img,(window[0][0],window[0][1]),(window[1][0],window[1][1]),(0,255,0), 2) 
    return img




def search_windows(img, windows, classifier, scaler):
    hot_windows = []

    for window in windows:
        ## get the smaller window img
        test_img = cv.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64))  
        features = extract_features_single(test_img,0)
        '''
        We have to change the features list to be arr in order to reshape it
        '''
        features_arr = np.array(features)
        test_features = scaler.transform(features_arr.reshape(1, -1))
        ## make a prediction 
        prediction = classifier.predict(test_features)
        ## append positive predictions to hot_windows
        if prediction == 1:
            hot_windows.append(window)

    return hot_windows

        