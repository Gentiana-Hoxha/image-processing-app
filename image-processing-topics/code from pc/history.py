from skimage import img_as_float, img_as_ubyte, io
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = img_as_float(io.imread(cells_image_path))

image_cv_grey = cv.imread(cells_image_path, 0);

#cv.imshow("Grey Image", image_cv_grey);

#cv.waitKey(0)
#cv.destroyAllWindows()

plt.hist(img, bins=100, range=(0,80))
runfile('C:/Users/STORM/egmentation.py', wdir='C:/Users/STORM')
runfile('C:/Users/STORM/color segmentation.py', wdir='C:/Users/STORM')

## ---(Sat Aug 17 18:03:02 2024)---
runfile('C:/Users/STORM/edge-filter.py', wdir='C:/Users/STORM')

## ---(Sun Aug 18 19:27:56 2024)---
runfile('C:/Users/STORM/untitled0.py', wdir='C:/Users/STORM')
runfile('C:/Users/STORM/edge-filter.py', wdir='C:/Users/STORM')
runfile('C:/Users/STORM/pwert and snobel.py', wdir='C:/Users/STORM')
clear
runfile('C:/Users/STORM/gaussian.py', wdir='C:/Users/STORM')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scipy.signal import convolve2d
from scipy.ndimage import convolve


# Read the input image
images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = io.imread(cells_image_path)

#Best to turn it into float to keep the image intact, since unit8 can loose some information
img_float = img_as_float(img, as_gray=True);
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scipy.signal import convolve2d
from scipy.ndimage import convolve


# Read the input image
images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = io.imread(cells_image_path)

#Best to turn it into float to keep the image intact, since unit8 can loose some information
img = img_as_float(io.imread(cells_image_path))
# Convert the image to grayscale
img_gray = color.rgb2gray(img)
io.imshow(img_gray)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from scipy.signal import convolve2d
from scipy.ndimage import convolve


# Read the input image
images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = io.imread(cells_image_path)

#Best to turn it into float to keep the image intact, since unit8 can loose some information
img = img_as_float(io.imread(cells_image_path))
# Convert the image to grayscale
img_gray = color.rgb2gray(img)
io.imshow(img_gray)

kernel = np.ones((5, 5), np.float32)/25

#normalized Gaussian Kernel 
#pagging, stride, kernel, all equal one,
gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                            [1/8, 1/4, 1/8],
                            [1/16, 1/8, 1/16]])
runfile('C:/Users/STORM/gaussian.py', wdir='C:/Users/STORM')

## ---(Sun Aug 25 16:42:29 2024)---
runfile('C:/Users/STORM/canny edge detection.py', wdir='C:/Users/STORM')

## ---(Wed Sep 18 20:20:52 2024)---
runfile('C:/Users/STORM/image-hist-equal.py', wdir='C:/Users/STORM')
images_folder_path = "C:/Users/STORM/Desktop/images/"
image = images_folder_path + "image_hist_equal.png"

import cv2
from skimage import io
from matplotlib import pyplot as plt

img = cv2.imread(image, 1)
#img = cv2.imread('images/retina.jpg', 1)

#Converting image to LAB Color so CLAHE can be applied to the luminance channel
lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#Splitting the LAB image to L, A and B channels, respectively
l, a, b = cv2.split(lab_img)

#plt.hist(l.flat, bins=100, range=(0,255))
###########Histogram Equlization#############
#Apply histogram equalization to the L channel
equ = cv2.equalizeHist(l)

#plt.hist(equ.flat, bins=100, range=(0,255))
#Combine the Hist. equalized L-channel back with A and B channels
updated_lab_img1 = cv2.merge((equ,a,b))

#Convert LAB image back to color (RGB)
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)

###########CLAHE#########################
#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
#plt.hist(clahe_img.flat, bins=100, range=(0,255))

#Combine the CLAHE enhanced L-channel back with A and B channels
updated_lab_img2 = cv2.merge((clahe_img,a,b))

#Convert LAB image back to color (RGB)
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)


cv2.imshow("Original image", img)
cv2.imshow("Equalized image", hist_eq_img)
cv2.imshow('CLAHE Image', CLAHE_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

def equalize_hist(img):
    # Get the image histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(hist)
    
    # Normalize the CDF
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Map intensities using the normalized CDF
    result = np.interp(img.flatten(), np.arange(256), cdf_normalized)
    
    return result.reshape(img.shape)

# Example usage
img = cv2.imread(image)
equ = equalize_hist(img)

cv2.imshow('Original', img)
cv2.imshow('Equalized', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()

images_folder_path = "C:/Users/STORM/Desktop/images/"
image = images_folder_path + "low-contrast-image.jpeg"
images_folder_path = "C:/Users/STORM/Desktop/images/"
image = images_folder_path + "low-contrast-image.jpeg"
#save_path = images_folder_path + "image_hist_equal-output.png"
#change_contrast(image, save_path)

import cv2
import numpy as np

def equalize_hist(img):
    # Get the image histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(hist)
    
    # Normalize the CDF
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Map intensities using the normalized CDF
    result = np.interp(img.flatten(), np.arange(256), cdf_normalized)
    
    return result.reshape(img.shape)

# Example usage
img = cv2.imread(image)
equ = equalize_hist(img)

cv2.imshow('Original', img)
cv2.imshow('Equalized', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()
images_folder_path = "C:/Users/STORM/Desktop/images/"
image = images_folder_path + "low-contrast-image.jpg"
#save_path = images_folder_path + "image_hist_equal-output.png"
#change_contrast(image, save_path)

import cv2
import numpy as np

def equalize_hist(img):
    # Get the image histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(hist)
    
    # Normalize the CDF
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Map intensities using the normalized CDF
    result = np.interp(img.flatten(), np.arange(256), cdf_normalized)
    
    return result.reshape(img.shape)

# Example usage
img = cv2.imread(image)
equ = equalize_hist(img)

cv2.imshow('Original', img)
cv2.imshow('Equalized', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()
runfile('C:/Users/STORM/image-hist-equal.py', wdir='C:/Users/STORM')
images_folder_path = "C:/Users/STORM/Desktop/images/"
image = images_folder_path + "low-contrast-image.jpg"

import cv2
from skimage import io
from matplotlib import pyplot as plt

img = cv2.imread(image, 1)
#img = cv2.imread('images/retina.jpg', 1)

#Converting image to LAB Color so CLAHE can be applied to the luminance channel
lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#Splitting the LAB image to L, A and B channels, respectively
l, a, b = cv2.split(lab_img)

#plt.hist(l.flat, bins=100, range=(0,255))
###########Histogram Equlization#############
#Apply histogram equalization to the L channel
equ = cv2.equalizeHist(l)

#plt.hist(equ.flat, bins=100, range=(0,255))
#Combine the Hist. equalized L-channel back with A and B channels
updated_lab_img1 = cv2.merge((equ,a,b))

#Convert LAB image back to color (RGB)
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)

###########CLAHE#########################
#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
#plt.hist(clahe_img.flat, bins=100, range=(0,255))

#Combine the CLAHE enhanced L-channel back with A and B channels
updated_lab_img2 = cv2.merge((clahe_img,a,b))

#Convert LAB image back to color (RGB)
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)


cv2.imshow("Original image", img)
cv2.imshow("Equalized image", hist_eq_img)
cv2.imshow('CLAHE Image', CLAHE_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## ---(Mon Nov  4 21:28:23 2024)---
runfile('C:/Users/STORM/pwert and snobel.py', wdir='C:/Users/STORM')

## ---(Tue Nov  5 19:54:45 2024)---
runfile('C:/Users/STORM/pwert and snobel.py', wdir='C:/Users/STORM')
runfile('C:/Users/STORM/gaussian.py', wdir='C:/Users/STORM')

## ---(Tue Nov  5 20:02:57 2024)---
runfile('C:/Users/STORM/gaussian.py', wdir='C:/Users/STORM')
runfile('C:/Users/STORM/canny edge detection.py', wdir='C:/Users/STORM')

## ---(Tue Nov  5 21:10:37 2024)---
runfile('C:/Users/STORM/gaussian.py', wdir='C:/Users/STORM')

## ---(Tue Nov  5 21:24:43 2024)---
runfile('C:/Users/STORM/gaussian.py', wdir='C:/Users/STORM')
runfile('C:/Users/STORM/canny edge detection.py', wdir='C:/Users/STORM')
runfile('C:/Users/STORM/.spyder-py3/canny edge detection.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/segmentation.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/edge-filter.py', wdir='C:/Users/STORM')

## ---(Tue Nov  5 21:38:14 2024)---
runfile('C:/Users/STORM/edge-filter.py', wdir='C:/Users/STORM')
runfile('C:/Users/STORM/.spyder-py3/segmentation.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/sharpning.py', wdir='C:/Users/STORM/.spyder-py3')
clear
runfile('C:/Users/STORM/.spyder-py3/sharpning.py', wdir='C:/Users/STORM/.spyder-py3')

## ---(Mon Dec 16 11:40:49 2024)---
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('C:/Users/STORM/Desktop/images/mount.png', 0)
h, w = template.shape[::] 

#methods available: ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
# For TM_SQDIFF, Good match yields minimum value; bad match yields large values
# For all others it is exactly opposite, max value = good fit.
plt.imshow(res, cmap='gray')

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)  #White rectangle with thickness 2. 

cv2.imshow("Matched image", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()
import cv2
import numpy as np
from matplotlib import pyplot as plt


img_rgb = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('C:/Users/STORM/Desktop/images/mount.png', 0)
h, w = template.shape[::]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')

threshold = 0.8 #Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.

loc = np.where( res >= threshold)  
#Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

#Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
#then the second item and then third, etc. 

for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
    #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  #Red rectangles with thickness 2. 

#cv2.imwrite('images/template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()
runfile('C:/Users/STORM/.spyder-py3/object detection template.py', wdir='C:/Users/STORM/.spyder-py3')
import cv2
import numpy as np
from matplotlib import pyplot as plt


img_rgb = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('C:/Users/STORM/Desktop/images/mount.png', 0)
h, w = template.shape[::]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')

threshold = 0.9 #Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.

loc = np.where( res >= threshold)  
#Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

#Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
#then the second item and then third, etc. 

for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
    #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  #Red rectangles with thickness 2. 

#cv2.imwrite('images/template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt


img_rgb = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('C:/Users/STORM/Desktop/images/mount.png', 0)
h, w = template.shape[::]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')

threshold = 0.7 #Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.

loc = np.where( res >= threshold)  
#Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

#Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
#then the second item and then third, etc. 

for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
    #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)  #Red rectangles with thickness 2. 

#cv2.imwrite('images/template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()
import cv2
import numpy as np
from matplotlib import pyplot as plt


img_rgb = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('C:/Users/STORM/Desktop/images/mount.png', 0)
h, w = template.shape[::]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')

threshold = 0.6 #Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.

loc = np.where( res >= threshold)  
#Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

#Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
#then the second item and then third, etc. 

for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
    #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)  #Red rectangles with thickness 2. 

#cv2.imwrite('images/template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()
runfile('C:/Users/STORM/.spyder-py3/object detection template.py', wdir='C:/Users/STORM/.spyder-py3')
import cv2
import numpy as np
from matplotlib import pyplot as plt


img_rgb = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('C:/Users/STORM/Desktop/images/mount.png', 0)
h, w = template.shape[::]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')

threshold = 0.55 #Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.

loc = np.where( res >= threshold)  
#Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

#Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
#then the second item and then third, etc. 

for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
    #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)  #Red rectangles with thickness 2. 

#cv2.imwrite('images/template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('C:/Users/STORM/Desktop/images/mount.png', 0)
h, w = template.shape[::] 

#methods available: ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
# For TM_SQDIFF, Good match yields minimum value; bad match yields large values
# For all others it is exactly opposite, max value = good fit.
plt.imshow(res, cmap='gray')

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)  #White rectangle with thickness 2. 

cv2.imshow("Matched image", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()
runfile('C:/Users/STORM/.spyder-py3/object detection template.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/untitled1.py', wdir='C:/Users/STORM/.spyder-py3')

"""
Created on Mon Dec 16 13:50:50 2024

@author: STORM
"""

from __future__ import print_function
import sys
import cv2 as cv
use_mask = False
img = None
templ = None
mask = None
image_window = "Source Image"
result_window = "Result window"
match_method = 0
max_Trackbar = 5
def main(argv):
    if (len(sys.argv) < 3):
        print('Not enough parameters')
        print('Usage:\nmatch_template_demo.py <image_name> <template_name> [<mask_name>]')
        return -1
    
    global img
    global templ
    img = cv.imread(sys.argv[1], cv.IMREAD_COLOR)
    templ = cv.imread(sys.argv[2], cv.IMREAD_COLOR)
    if (len(sys.argv) > 3):
        global use_mask
        use_mask = True
        global mask
        mask = cv.imread( sys.argv[3], cv.IMREAD_COLOR )
    if ((img is None) or (templ is None) or (use_mask and (mask is None))):
        print('Can\'t read one of the images')
        return -1
    
    
    cv.namedWindow( image_window, cv.WINDOW_AUTOSIZE )
    cv.namedWindow( result_window, cv.WINDOW_AUTOSIZE )
    
    
    trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
    cv.createTrackbar( trackbar_label, image_window, match_method, max_Trackbar, MatchingMethod )
    
    MatchingMethod(match_method)
    
    cv.waitKey(0)
    return 0

def MatchingMethod(param):
    global match_method
    match_method = param
    
    img_display = img.copy()
    
    method_accepts_mask = (cv.TM_SQDIFF == match_method or match_method == cv.TM_CCORR_NORMED)
    if (use_mask and method_accepts_mask):
        result = cv.matchTemplate(img, templ, match_method, None, mask)
    else:
        result = cv.matchTemplate(img, templ, match_method)
    
    
    cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
    
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
    
    
    if (match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    
    
    cv.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    cv.rectangle(result, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    cv.imshow(image_window, img_display)
    cv.imshow(result_window, result)
    
    pass
if __name__ == "__main__":
    main(sys.argv[1:])

"""
Created on Mon Dec 16 13:50:50 2024

@author: STORM
"""

from __future__ import print_function
import sys
import cv2 as cv
use_mask = False
img = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png')
templ = cv2.imread('C:/Users/STORM/Desktop/images/mount2.png', 0)
mask = None
image_window = "Source Image"
result_window = "Result window"
match_method = 0
max_Trackbar = 5
def main(argv):
    if (len(sys.argv) < 3):
        print('Not enough parameters')
        print('Usage:\nmatch_template_demo.py <image_name> <template_name> [<mask_name>]')
        return -1
    
    global img
    global templ
    img = cv.imread(sys.argv[1], cv.IMREAD_COLOR)
    templ = cv.imread(sys.argv[2], cv.IMREAD_COLOR)
    if (len(sys.argv) > 3):
        global use_mask
        use_mask = True
        global mask
        mask = cv.imread( sys.argv[3], cv.IMREAD_COLOR )
    if ((img is None) or (templ is None) or (use_mask and (mask is None))):
        print('Can\'t read one of the images')
        return -1
    
    
    cv.namedWindow( image_window, cv.WINDOW_AUTOSIZE )
    cv.namedWindow( result_window, cv.WINDOW_AUTOSIZE )
    
    
    trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
    cv.createTrackbar( trackbar_label, image_window, match_method, max_Trackbar, MatchingMethod )
    
    MatchingMethod(match_method)
    
    cv.waitKey(0)
    return 0

def MatchingMethod(param):
    global match_method
    match_method = param
    
    img_display = img.copy()
    
    method_accepts_mask = (cv.TM_SQDIFF == match_method or match_method == cv.TM_CCORR_NORMED)
    if (use_mask and method_accepts_mask):
        result = cv.matchTemplate(img, templ, match_method, None, mask)
    else:
        result = cv.matchTemplate(img, templ, match_method)
    
    
    cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
    
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
    
    
    if (match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    
    
    cv.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    cv.rectangle(result, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    cv.imshow(image_window, img_display)
    cv.imshow(result_window, result)
    
    pass
if __name__ == "__main__":
    main(sys.argv[1:])
runfile('C:/Users/STORM/.spyder-py3/object detection template.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/CrossCorrelation.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/ncc.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/object detection template.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/edge-based.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/jpeg.py', wdir='C:/Users/STORM/.spyder-py3')

## ---(Wed Dec 18 12:27:18 2024)---

"""
Created on Wed Dec 18 12:29:29 2024

@author: STORM
"""

####################################### Imports >

#open_cv is a library of programming functions used to manipulate matrices(images videos)
import cv2 as cv
#ploting library #pyplot to create 2d plots and graphs #similar as matlab
import matplotlib.pyplot as plt 

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

image_cv_grey = cv.imread(cells_image_path, 0);
image_cv = cv.imread(cells_image_path, 1);

print(type(image_cv)) #<class 'numpy.ndarray'>


#By default it import images as BRG and not RGB that's why display cv image with plt looks different
plt.imshow(cv.cvtColor(image_cv, cv.COLOR_BGR2RGB))# change color space
#plt.imshow(image_cv) #shows with different colors

cv.imshow("Grey Image", image_cv_grey);
cv.imshow("Colored Image", image_cv);

cv.waitKey(0)
cv.destroyAllWindows()

import cv2 as cv
#ploting library #pyplot to create 2d plots and graphs #similar as matlab
import matplotlib.pyplot as plt 

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

image_cv_grey = cv.imread(cells_image_path, 0);
image_cv = cv.imread(cells_image_path, 1);

print(type(image_cv))

#By default it import images as BRG and not RGB that's why display cv image with plt looks different
plt.imshow(cv.cvtColor(image_cv, cv.COLOR_BGR2RGB))# change color space
#plt.imshow(image_cv) #shows with different colors

cv.imshow("Grey Image", image_cv_grey);
cv.imshow("Colored Image", image_cv);


import cv2 as cv
#ploting library #pyplot to create 2d plots and graphs #similar as matlab
import matplotlib.pyplot as plt 

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

image_cv_grey = cv.imread(cells_image_path, 0);
image_cv = cv.imread(cells_image_path, 1);

print(type(image_cv))

#By default it import images as BRG and not RGB that's why display cv image with plt looks different
plt.imshow(cv.cvtColor(image_cv))# change color space
#plt.imshow(image_cv) #shows with different colors

cv.imshow("Grey Image", image_cv_grey);
cv.imshow("Colored Image", image_cv);
import cv2 as cv
#ploting library #pyplot to create 2d plots and graphs #similar as matlab
import matplotlib.pyplot as plt 

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

image_cv_grey = cv.imread(cells_image_path, 0);
image_cv = cv.imread(cells_image_path, 1);

print(type(image_cv))

#By default it import images as BRG and not RGB that's why display cv image with plt looks different
plt.imshow(image_cv)# change color space
#plt.imshow(image_cv) #shows with different colors

cv.imshow("Grey Image", image_cv_grey);
cv.imshow("Colored Image", image_cv);
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from scipy.signal import convolve2d
from scipy.ndimage import convolve


# Read the input image
images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = io.imread(cells_image_path)

#Best to turn it into float to keep the image intact, since unit8 can loose some information
img = img_as_float(io.imread(cells_image_path))
# Convert the image to grayscale
img_gray = color.rgb2gray(img)
io.imshow(img_gray)


#normalized Gaussian Kernel 
#pagging, stride, kernel, all equal one, convolution
filtering_kernel = np.array([[0, 1/5, 0],
                            [1/5, 1/5, 1/5],
                            [0, 1/5, 0]])

#what the -1 depth?/ REPLICATE
conv_blur = cv2.filter2D(img_gray, -1, filtering_kernel, borderType=cv2.BORDER_REPLICATE);


identity_kernel = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

sharpening_constant = 10
sharpening_diff = np.subtract(identity_kernel, filtering_kernel)*sharpening_constant
sharpening_kernel = np.add(sharpening_diff, identity_kernel)

conv_sharpen = cv2.filter2D(img_gray, -1, sharpening_kernel, borderType=cv2.BORDER_REPLICATE);

cv2.imshow('Original Image', img_gray)
cv2.imshow('Blur filter', conv_blur)
cv2.imshow('sharpen filter', conv_sharpen)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from scipy.signal import convolve2d
from scipy.ndimage import convolve


# Read the input image
images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = io.imread(cells_image_path)

#Best to turn it into float to keep the image intact, since unit8 can loose some information
img = img_as_float(io.imread(cells_image_path))
# Convert the image to grayscale
img_gray = color.rgb2gray(img)
io.imshow(img_gray)

kernel = np.ones((3, 3), np.float32)/9 #25 bc 5 by of ones, and 25 of them make 1.0 so it's normalized

#normalized Gaussian Kernel 
#pagging, stride, kernel, all equal one, convolution
gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                            [1/8, 1/4, 1/8],
                            [1/16, 1/8, 1/16]])

#what the -1 depth?/ REPLICATE
conv_gaussian = cv2.filter2D(img_gray, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE);
conv_kernel = cv2.filter2D(img_gray, -1, kernel, borderType=cv2.BORDER_REPLICATE);

cv2.imshow('Original Image', img_gray)
cv2.imshow('simple filter', conv_kernel)
cv2.imshow('gaussian filter', conv_gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from scipy.signal import convolve2d
from scipy.ndimage import convolve

import albumentations as albu
from PIL import Image

# Read the input image
images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = io.imread(cells_image_path)

#Best to turn it into float to keep the image intact, since unit8 can loose some information
img = img_as_float(io.imread(cells_image_path))
# Convert the image to grayscale
img_gray = color.rgb2gray(img)
io.imshow(img_gray)

kernel = np.ones((3, 3), np.float32)/9 #25 bc 5 by of ones, and 25 of them make 1.0 so it's normalized

#normalized Gaussian Kernel 
#pagging, stride, kernel, all equal one, convolution
gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                            [1/8, 1/4, 1/8],
                            [1/16, 1/8, 1/16]])

transform = albu.GaussianNoise(var_limit=(10,50),mean=0,p=0.5)

image_noise = transform(image=img_gray)['img_gray']


#what the -1 depth?/ REPLICATE
conv_gaussian = cv2.filter2D(img_gray, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE);
conv_kernel = cv2.filter2D(img_gray, -1, kernel, borderType=cv2.BORDER_REPLICATE);

cv2.imshow('Original Image', img_gray)
cv2.imshow('simple filter', conv_kernel)
cv2.imshow('gaussian filter', conv_gaussian)
cv2.imshow('image_noise filter', image_noise)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
Created on Wed Dec 18 12:29:29 2024

@author: STORM
"""

#open_cv is a library of programming functions used to manipulate matrices(images videos)
import cv2 as cv
#ploting library #pyplot to create 2d plots and graphs #similar as matlab
import matplotlib.pyplot as plt 

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread(cells_image_path)
plt.figure(0)
imgplot = plt.imshow(img)

height = img.shape[0]
width = img.shape[1]

#Create an array of noise by drawing from a Gaussian distribution
#Add to source image
#Clip at [0, 255]
n_20 = np.random.normal(0, 20, [height, width, 3])
noisy_20_img = (img+n_20).clip(0,255)

n_50 = np.random.normal(0, 50, [height, width, 3])
noisy_50_img = (img+n_50).clip(0,255)

plt.figure(1)
imgplot = plt.imshow(noisy_20_img.astype('uint8'))

plt.figure(2)
imgplot = plt.imshow(noisy_50_img.astype('uint8'))
runfile('C:/Users/STORM/.spyder-py3/gaussian-noise.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/untitled1.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/gaussian-matrix.py', wdir='C:/Users/STORM/.spyder-py3')
runfile('C:/Users/STORM/.spyder-py3/gaussian-noise.py', wdir='C:/Users/STORM/.spyder-py3')

"""
Created on Wed Dec 18 15:48:19 2024

@author: STORM
"""

#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/StX_1iEO3ck

"""
Spyder Editor

See how median is much better at cleaning salt and pepper noise compared to Gaussian
"""
import cv2
import numpy as np
from scipy.ndimage.filters import convolve
from skimage import io
from skimage.filters import median

#img = io.imread('images/einstein.jpg', as_gray=True)

#Needs 8 bit, not float.

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "gaussian-noise.png"

img_salt_pepper_noise = cv2.imread(cells_image_path, 0)

img = img_salt_pepper_noise


median_using_cv2 = cv2.medianBlur(img, 3)

from skimage.morphology import disk
median_using_skimage = median(img, disk(3), mode='constant', cval=0.0)


cv2.imshow("Original", img)
cv2.imshow("cv2 median", median_using_cv2)
cv2.imshow("Using skimage median", median_using_skimage)

cv2.waitKey(0)          
cv2.destroyAllWindows()
runfile('C:/Users/STORM/.spyder-py3/untitled2.py', wdir='C:/Users/STORM/.spyder-py3')