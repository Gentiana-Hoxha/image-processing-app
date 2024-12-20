import numpy as np
import cv2

windowsize_r = 8
windowsize_c = 8

# Read and convert image to grayscale
img = cv2.imread('C:/Users/STORM/Desktop/images/High-res-1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate padding sizes
pad_r = (windowsize_r - (gray.shape[0] % windowsize_r)) % windowsize_r
pad_c = (windowsize_c - (gray.shape[1] % windowsize_c)) % windowsize_c

# Pad the image to make its dimensions divisible by 8
gray_padded = cv2.copyMakeBorder(gray, 0, pad_r, 0, pad_c, cv2.BORDER_CONSTANT, value=0)

# Loop through the padded image in 8x8 blocks
for r in range(0, gray_padded.shape[0], windowsize_r):
    for c in range(0, gray_padded.shape[1], windowsize_c):
        # Extract 8x8 block
        window = gray_padded[r:r + windowsize_r, c:c + windowsize_c]
        
        # Display each 8x8 block
        cv2.imshow("8x8 Block", window)
        cv2.waitKey(100)  # Show each block for 100ms

# Display the padded grayscale image
cv2.imshow("Padded Grayscale Image", gray_padded)
cv2.waitKey(0)
cv2.destroyAllWindows()
