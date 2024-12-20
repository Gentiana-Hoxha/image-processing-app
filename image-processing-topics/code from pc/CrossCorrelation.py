# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:18:03 2024

@author: STORM
"""

import cv2
import numpy as np

def cross_correlation(image, template):
    # Get dimensions
    img_h, img_w = image.shape
    temp_h, temp_w = template.shape
    
    # Output result matrix
    result = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1))

    # Slide the template across the image
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            # Extract region of interest (ROI)
            roi = image[y:y+temp_h, x:x+temp_w]
            
            # Cross-Correlation: Sum of element-wise multiplication
            correlation = np.sum(roi * template)
            result[y, x] = correlation

    return result


# Load the image and template in grayscale
image = cv2.imread('C:/Users/STORM/Desktop/images/mounts.png', cv2.IMREAD_GRAYSCALE) 
template = cv2.imread('C:/Users/STORM/Desktop/images/mount2.png', cv2.IMREAD_GRAYSCALE)
# Call the custom function
result = cross_correlation(image, template)

# Normalize result for visualization
result_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display the result
cv2.imshow("Cross-Correlation Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()