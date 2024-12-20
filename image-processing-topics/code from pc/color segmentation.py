# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:24:28 2024

@author: STORM
"""

from skimage import io, measure
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = io.imread(cells_image_path)
plt.imshow(img)

hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
maskCellOne = cv.inRange(hsv, (120,90,90), (140,255,255))
maskCellTwo = cv.inRange(hsv, (150,60,60), (170,255,255))

#from scipy import ndimage as nd
#closed_maskRedCells = nd.binary_closing(maskRedCells, np.ones((11,11)))

labled_cells_image_one = measure.label(maskCellOne);
labled_cells_image_two = measure.label(maskCellTwo);

from skimage.color import label2rgb
image_one_overlay = label2rgb(labled_cells_image_one, image=img)
image_two_overlay = label2rgb(labled_cells_image_two, image=img)

#props = measure.regionprops_table(image_one_overlay)

io.imshow(image_one_overlay)
#io.imshow(image_two_overlay)

#io.imshow(maskCellOne);
#io.imshow(maskCellTwo);
#io.imshow(img);