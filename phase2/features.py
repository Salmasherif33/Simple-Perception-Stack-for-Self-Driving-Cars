

from cv2 import resize
from load import *
from skimage.feature import hog
import numpy as np
import matplotlib.image as mpimg


def feature(imgs):
    ##TODO  loop on each image in the list
    ##TODO  append to a feature vector ([spatial features] , [Histogram features])
    ##TODO  call get_hog_features function for each img -> append it's return vector to the previously mentioned vector .
    ##TODO  append that vector to a list of feature vectors
    ##TODO  return the mentioned list 

    ##NB : cspace is always RGB ....... vis = False ..... SIZE IS ALWAYS (64, 64, 3)

    return


def resizing_image(img, size):
    res_img = resize(img, size)

def get_hog_features(img, orientations):
    features, hog_image = hog(img, orientations, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=False, visualize=True,feature_vector=True,channel_axis=-1)

    return features, hog_image


# The Numpy histogram function is similar to the hist() function of matplotlib library, 
# the only difference is that the Numpy histogram gives the numerical representation of the dataset
#  while the hist() gives graphical representation of the dataset.

def calc_histogram(img, nbins):
    ch1,bins = np.histogram(img[:,:,0], nbins)
    ch2,bins = np.histogram(img[:,:,1], nbins)
    ch3,bins = np.histogram(img[:,:,2], nbins)

    # Concatenate the histograms into a single feature vector
    histogram_features = np.concatenate((ch1, ch2, ch3))
    # Return the individual histograms, bin_centers and feature vector
    return histogram_features

def bin_spatial(image, size=(64, 64)):
    #use ravel to make it 1D, why ? :(
    color1 = cv2.resize(image[:,:,0], size).ravel()
    color2 = cv2.resize(image[:,:,1], size).ravel() 
    color3 = cv2.resize(image[:,:,2], size).ravel() 
    return np.hstack((color1, color2, color3))

#extract_features() function combines
#the features from HOG along with spatial features and color histogram features 
def extract_features(imgs, hog_channel):
    features = []
    for img in imgs:
        img_featurs = []
        #image = mpimg.imread(img)
        feature_image = np.copy(img)
        spatial_features = bin_spatial(feature_image, size=(64, 64))
        img_featurs.append(spatial_features)
        # Apply color_hist()
        hist_features = calc_histogram(feature_image, nbins=32)
        img_featurs.append(hist_features)
        hogged_feature, hogged_image = get_hog_features(feature_image, 9)
        img_featurs.append(hogged_feature)
    
        features.append(np.concatenate(img_featurs))

    return  hogged_image, features
