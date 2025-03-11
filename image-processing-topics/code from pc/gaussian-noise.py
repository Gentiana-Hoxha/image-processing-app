# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:29:29 2024

@author: STORM
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images_folder_path = "/Users/g.hoxha/Documents/GitHub/image-processing-app/image-processing-topics/assets/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

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

plt.figure(3)
imgplot = plt.imshow(n_50.astype('uint8'))


cv2.imshow("Noise", cv2.cvtColor(noisy_50_img.astype('uint8'), cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()