from cv2 import imread
from load import *
from features import *
import matplotlib.pyplot as plt
from windows import *
from svm import *
import sys
from vehicle_detection import *
from scipy.ndimage import label
def main ():
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

    elif (type_ == "img"):
        img = cv.imread(path)

        ## PHASE II ##
        cars,not_cars = load(train_path)
        
        hogged_car, car_features = extract_features(cars[4000:5000],'ALL' )
        
        hogged_not_car, not_car_features = extract_features(not_cars[0:1000],'ALL' )

        

        windows_list  = sliding_windows(np.copy(img))
        y_start_stop = [800, 1000] 
        overlap = 0.5
        windows_list = sliding_windows(np.copy(img))
        #windows_list = slidingWindow(img )                   
        svc , X_scaler = train(car_features,not_car_features)
        #hot_windows = search_windows(np.copy(img), windows_list, svc, X_scaler)

        
        all_hot_win = detect_cars(np.copy(img.astype(np.float32)/255), 400, 656, 1.5, X_scaler, svc)
        
        result = vis_windows(np.copy(img),all_hot_win)
       
        heat = np.zeros_like(np.copy(img)[:,:,0])
        # Add heat to each box in box list
        heat = add_heat(heat, all_hot_win)
       
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)  
        # Visualize the heatmap when displaying  
        heatmap = np.clip(heat, 0, 255)
       
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        cv.imshow('Output Image',draw_img)
        cv.waitKey(0)





    #cars,not_cars = load('D:"\\"faculty"\\"image_designPattern"\\"Simple-Perception-Stack-for-Self-Driving-Cars"\\"phase2')
    """ img = imread('image0009.png')
    r_image = bin_spatial(img)
    f, image = get_hog_features(r_image, 9)

    plt.imshow(image, cmap="gray")
    plt.show() """

    """ img = imread('vehicles/GTI_Far/image0126.png')
    #img = imread('non-vehicles/Extras/extra17.png')
    #creating hog features
    hog_image = extract_features(img, 0)
    plt.axis("off")
    plt.imshow(hog_image)
    plt.show() """


if __name__== "__main__":
    main()

