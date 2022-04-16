import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def lane_line_markings(img):
    src = np.float32(
        [
            (572, 454),  # Top-left corner
            (250, 660),  # Bottom-left corner
            (1100, 684),  # Bottom-right corner
            (734, 454)  # Top-right corner
        ])

    detected_plot = cv.polylines(np.copy(img), np.int32([
        src]), True, (147, 20, 255), 3)

    return detected_plot



def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    #temp = binary_img[200:700  , 180:1200]
    #binary_img = cv.matchTemplate(binary_img,temp,cv.TM_CCOEFF_NORMED)
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Recast the x and y points into usable format for cv.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    ## Draw the using poly lines ##

    cv.fillPoly(color_warp, np.int_([pts]), (255,255, 0))
    cv.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,255), thickness=15)
    cv.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp perspective using given using Minv 
    newwarp = cv.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result