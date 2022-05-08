from cv2 import imread
from load import *
from features import *
import matplotlib.pyplot as plt

def main ():

    #cars,not_cars = load('D:"\\"faculty"\\"image_designPattern"\\"Simple-Perception-Stack-for-Self-Driving-Cars"\\"phase2')
    """ img = imread('image0009.png')
    r_image = bin_spatial(img)
    f, image = get_hog_features(r_image, 9)

    plt.imshow(image, cmap="gray")
    plt.show() """

    img = imread('vehicles/GTI_Far/image0126.png')
    #img = imread('non-vehicles/Extras/extra17.png')
    #creating hog features
    hog_image = extract_features(img, 0)
    plt.axis("off")
    plt.imshow(hog_image)
    plt.show()




if __name__== "__main__":
    main()

