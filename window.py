import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




## More simpler sliding window fn ##
## only detects the right line ## :))
def sliding_window_polyfit(exampleImg,img):
    # Take a histo of the bottom half of the image
    histo = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histo
    # These will be the starting point for the left and right lines
    midpoint = np.int(histo.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histo
    # this changes it so that only a quarter of the histo (directly to the left/right) is considered
    leftx_base = np.argmax(histo[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histo[midpoint:(midpoint+quarter_point)]) + midpoint
    
    #print('base pts:', leftx_base, rightx_base)


    
    window_height = np.int(img.shape[0]/10)     ##assumingg no. of sliding windows = 10 ##
    # get x & y for white pixels 
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    

    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_ends = []
    right_lane_ends = []
    # Rectangle data for visualization
    rectangle_data = []


    for window in range(10):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - 90      ##some marginal fitt to make sure each window alligned same as avg windows##
        win_xleft_high = leftx_current + 90 
        win_xright_low = rightx_current - 90
        win_xright_high = rightx_current + 90
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        valid_lefts = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        valid_rights = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_ends.append(valid_lefts)
        right_lane_ends.append(valid_rights)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(valid_lefts) > minpix:
            leftx_current = np.int(np.mean(nonzerox[valid_lefts]))
        if len(valid_rights) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[valid_rights]))

    # Concatenate the arrays of indices
    left_lane_ends = np.concatenate(left_lane_ends)
    right_lane_ends = np.concatenate(right_lane_ends)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_ends]
    lefty = nonzeroy[left_lane_ends] 
    rightx = nonzerox[right_lane_ends]
    righty = nonzeroy[right_lane_ends] 

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    visualization_data = (rectangle_data, histo)
    h = exampleImg.shape[0]
    left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
   

    rectangles = visualization_data[0]
    histo = visualization_data[1]

    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((img, img, img))*255)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for rect in rectangles:
    # Draw the windows on the visualization image
        cv.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
        cv.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_ends], nonzerox[left_lane_ends]] = [255, 255, 255]
    out_img[nonzeroy[right_lane_ends], nonzerox[right_lane_ends]] = [255, 255, 255]
    return left_fit, right_fit, left_lane_ends, right_lane_ends, visualization_data , out_img,ploty, leftx, lefty,rightx,righty





