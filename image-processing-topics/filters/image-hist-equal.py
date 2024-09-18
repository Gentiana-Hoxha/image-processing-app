# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:24:21 2024

@author: STORM
"""

import cv2
import numpy as np

#images_folder_path = "C:/Users/STORM/Desktop/images/"
#image = images_folder_path + "image_hist_equal.png"

def change_contrast(image_path, save_path):
    img = cv2.imread(image_path)

    height, width, _ = img.shape

    # Convert the image to a NumPy array
    img_array = img.astype(np.uint8)

    # Separate channels
    r_channel, g_channel, b_channel = cv2.split(img_array)

    # Calculate probabilities
    probability_r = np.bincount(r_channel.flatten()) / (width * height)
    probability_g = np.bincount(g_channel.flatten()) / (width * height)
    probability_b = np.bincount(b_channel.flatten()) / (width * height)

    # Calculate cumulative probabilities
    cumulative_r = np.cumsum(probability_r)
    cumulative_g = np.cumsum(probability_g)
    cumulative_b = np.cumsum(probability_b)

     Map intensities
    new_r = np.interp(r_channel.flatten(), np.arange(256), cumulative_r * 255)
    new_g = np.interp(g_channel.flatten(), np.arange(256), cumulative_g * 255)
    new_b = np.interp(b_channel.flatten(), np.arange(256), cumulative_b * 255)

     Combine channels
    new_img = cv2.merge([new_r.reshape(height, width), new_g.reshape(height, width), new_b.reshape(height, width)])

     Save the image
    cv2.imwrite(save_path, new_img)

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


