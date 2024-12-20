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