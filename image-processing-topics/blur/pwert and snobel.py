import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Read the input image
images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = io.imread(cells_image_path)

# Convert the truecolor RGB image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the image to double (floating-point type)
gray_image = gray_image.astype(float)

# Sobel operator masks
Mx = np.array([[-1, -2, -1], 
               [ 0,  0,  0], 
               [ 1,  2,  1]])

My = np.array([[-1,  0,  1], 
               [-2,  0,  2], 
               [-1,  0,  1]])

# Pre-allocate the filtered_image matrix with zeros
filtered_image = np.zeros_like(gray_image)

# Edge detection process
for i in range(1, gray_image.shape[0] - 1):
    for j in range(1, gray_image.shape[1] - 1):
        Gx = np.sum(Mx * gray_image[i-1:i+2, j-1:j+2])
        Gy = np.sum(My * gray_image[i-1:i+2, j-1:j+2])
        filtered_image[i, j] = np.sqrt(Gx**2 + Gy**2)
        
# Convert the filtered image to uint8 type
filtered_image = np.uint8(filtered_image)

# Display the filtered image
plt.figure(figsize=(10, 10))
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.show()

# Define a threshold value
threshold_value = 100

# Apply thresholding to get the output image
output_image = np.maximum(filtered_image, threshold_value)
output_image[output_image == threshold_value] = 0

# Convert the output image to binary (black and white)
_, output_image = cv2.threshold(output_image, 1, 255, cv2.THRESH_BINARY)

# Display the output image
plt.figure(figsize=(10, 10))
plt.imshow(output_image, cmap='gray')
plt.title('Edge Detected Image - Sobel')
plt.show()
