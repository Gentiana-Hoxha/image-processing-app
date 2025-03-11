import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color
from scipy.signal import convolve2d
from scipy.ndimage import convolve

import albumentations as albu
from PIL import Image

# Read the input image
images_folder_path = "/Users/g.hoxha/Documents/GitHub/image-processing-app/image-processing-topics/assets/images/"
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