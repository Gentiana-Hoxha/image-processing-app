# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:37:23 2024

@author: STORM
"""

import cv2
import numpy as np

def normalized_cross_correlation(image, template):
    # Get dimensions
    img_h, img_w = image.shape
    temp_h, temp_w = template.shape

    # Compute template mean and standard deviation
    template_mean = np.mean(template)
    template_std = np.std(template)

    # Ensure template_std is not zero (avoid division by zero)
    if template_std == 0:
        raise ValueError("Standard deviation of template is zero, cannot normalize.")

    # Output result matrix
    result = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1))

    # Slide the template across the image
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            # Extract region of interest (ROI)
            roi = image[y:y+temp_h, x:x+temp_w]

            # Calculate mean and std of the current ROI
            roi_mean = np.mean(roi)
            roi_std = np.std(roi)

            # Ensure roi_std is not zero
            if roi_std == 0:
                result[y, x] = 0  # Avoid division by zero
                continue

            # Compute NCC using the formula
            ncc = np.sum(((roi - roi_mean) * (template - template_mean)) / (roi_std * template_std))
            result[y, x] = ncc

    return result

# Load the image and template in grayscale
image = cv2.imread('/Users/g.hoxha/Documents/GitHub/image-processing-app/image-processing-topics/assets/images//mounts.png', cv2.IMREAD_GRAYSCALE) 
template = cv2.imread('/Users/g.hoxha/Documents/GitHub/image-processing-app/image-processing-topics/assets/images//mount2.png', cv2.IMREAD_GRAYSCALE)

# Call the custom NCC function
ncc_result = normalized_cross_correlation(image, template)

# Normalize result for visualization
ncc_result_norm = cv2.normalize(ncc_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display the result
cv2.imshow("NCC Result", ncc_result_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()
