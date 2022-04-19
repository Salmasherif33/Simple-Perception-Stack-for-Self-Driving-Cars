import sys
import cv2 as cv
import numpy as np
import glob



def gaussian_blur(img, kernel_size):
    return cv.gaussian_blur(img,(kernel_size, kernel_size),0)


def grayscale(img):
    #returning the saturation part only (could by L)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    return gray

def threshold(img):
    cann_thresh = (0,1)
    binary = np.zeros_like(img)
    binary[(img > cann_thresh[0]) & (img < cann_thresh[1])] = 1

    return binary

def sobel_transform(img, dir, th):
    if(dir == 'x'):
        abs_sobel = np.absolute(cv.Sobel(img, cv.CV_64F, 1, 0))
    if(dir == 'x'):
        abs_sobel = np.absolute(cv.Sobel(img, cv.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= th[0]) & (scaled_sobel <= th[1])] = 1
    sobel_mask = binary_output
    # Return the result
    return sobel_mask
    
def mag_gradient(img, filter_size, th):
    # Take both Sobel x and y gradients
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=filter_size)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=filter_size)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= th[0]) & (gradmag <= th[1])] = 1
    mag_mask = binary_output
    # Return the binary image
    return mag_mask

def threshold_grad_dir(img, filter_size,thr):
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=filter_size)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=filter_size)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thr[0]) & (absgraddir <= thr[1])] = 1
    dir_mask = binary_output
    # Return the binary image
    return dir_mask

def canny(img,s_thresh,l_thresh):
    
    gray_img = grayscale(img)
    height, width = gray_img.shape
    grad_x = sobel_transform(gray_img, 'x', th=(10, 200))
    dir_binary = threshold_grad_dir(gray_img, 3,thr=(np.pi/6, np.pi/2))
    combined = ((grad_x == 1) & (dir_binary == 1))
     # R & G thresholds so that yellow lanes are detected well.
    color_threshold = 150
    R = img[:,:,0]
    G = img[:,:,1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)

    # Apply color threshold for better detection of yello and white lines in all environmental condition
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    # Select S channel because it is usually the best performant
    # for this task. R channel also performs similarly.
    s_channel = hls[:,:,2] 
    l_channel = hls[:,:,1]

     # S channel performs well for detecting bright yellow and white lanes
    s_condition = (s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])

     # We put a threshold on the L channel to avoid pixels which have shadows and as a result darker.   
    l_condition = (l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])

    # combine all the thresholds
    # A pixel should either be a yellowish or whiteish
    # And it should also have a gradient, as per our thresholds
    color_combined[(r_g_condition & l_condition) & (s_condition | combined)] = 1

    # apply the region of interest mask
    mask = np.zeros_like(color_combined)
    region_of_interest_vertices = np.array([[0,height-1], [width/2, int(0.5*height)], [width-1, height-1]], dtype=np.int32)
    cv.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv.bitwise_and(color_combined, mask)

    return thresholded

    """ img = grayscale(img)
    img = cv.Canny(img,150,200, L2gradient = True) 
    h = img.shape[0]
    w = img.shape[1]
    (cX, cY) = (w // 4 , h )
    img[0:cY , 0:cX] = 0 ##hard-setting all values at this rectangle to be zeroes 
    img[0:h , w-cX:w] = 0 ##hard-setting all values at this rectangle to be zeroes 
    #img = threshold(img) """
    return img


