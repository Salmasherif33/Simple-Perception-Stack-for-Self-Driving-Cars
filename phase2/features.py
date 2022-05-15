from cv2 import resize
import cv2
from skimage.feature import hog
import numpy as np
import matplotlib.image as mpimg



def get_hog_features(img, orientations, feat_vec):
    features, hog_image = hog(img, orientations, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=False, visualize=True,feature_vector=feat_vec)

    return features, hog_image


# The Numpy histogram function is similar to the hist() function of matplotlib library, 
# the only difference is that the Numpy histogram gives the numerical representation of the dataset
#  while the hist() gives graphical representation of the dataset.

def calc_histogram(img, nbins):
    ch1,bins = np.histogram(img[:,:,0], nbins, range=(0,256))
    ch2,bins = np.histogram(img[:,:,1], nbins,range=(0,256))
    ch3,bins = np.histogram(img[:,:,2], nbins,range=(0,256))

    # Concatenate the histograms into a single feature vector
    histogram_features = np.concatenate((ch1, ch2, ch3))
    # Return the individual histograms, bin_centers and feature vector
    return histogram_features

def bin_spatial(image, size=(32, 32)):
    #use ravel to make it 1D, why ? :(
    color1 = resize(image[:,:,0], size).ravel()
    color2 = resize(image[:,:,1], size).ravel() 
    color3 = resize(image[:,:,2], size).ravel() 
    return np.hstack((color1, color2, color3))

#extract_features() function combines
#the features from HOG along with spatial features and color histogram features 
def extract_features(imgs, hog_channel):
    features = []
    for file in imgs:
        img = mpimg.imread(file)
        img_featurs = []
        #image = mpimg.imread(img)
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        #feature_image = np.copy(img)
        spatial_features = bin_spatial(feature_image, size=(32, 32))
        img_featurs.append(spatial_features)
        # Apply color_hist()
        hist_features = calc_histogram(feature_image, nbins=32)
        img_featurs.append(hist_features)

        if hog_channel == 'ALL':
            hog_features = []
            for ch in range(feature_image.shape[2]):
                hogged_feature, hogged_image = get_hog_features(feature_image[:,:,ch], 9, True)
                hog_features.append(hogged_feature)
            hog_features = np.ravel(hog_features)
            img_featurs.append(hog_features)

        if hog_channel == 0:
            hogged_feature, hogged_image = get_hog_features(feature_image[:,:,0], 9, True)
            img_featurs.append(hogged_feature)
    
        features.append(np.concatenate(img_featurs))

    return  hogged_image, features

def extract_features_single(img, hog_channel):
    img_featurs = []
    #image = mpimg.imread(img)
    feature_image = np.copy(img)
    spatial_features = bin_spatial(feature_image, size=(32, 32))
    img_featurs.append(spatial_features)
    # Apply color_hist()
    hist_features = calc_histogram(feature_image, nbins=32)
    img_featurs.append(hist_features)
    if hog_channel == 'ALL':
        hog_features = []
        for ch in range(feature_image.shape[2]):
            hogged_feature, hogged_image = get_hog_features(feature_image[:,:,ch], 9, True)
            hog_features.append(hogged_feature)
        hog_features = np.ravel(hog_features)
        img_featurs.append(hog_features)

    if hog_channel == 0:
        hogged_feature, hogged_image = get_hog_features(feature_image[:,:,0], 9, True)
        img_featurs.append(hogged_feature)

    return   np.concatenate(img_featurs)

