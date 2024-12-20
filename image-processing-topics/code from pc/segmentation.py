from skimage import img_as_float, img_as_ubyte, io, color
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

img = img_as_float(io.imread(cells_image_path))
#img = img_as_float(io.imread(cells_image_path))

#image_cv_grey = cv.imread(cells_image_path, 0);

#cv.imshow("Grey Image", image_cv_grey);
# Convert the image to grayscale
img_gray = color.rgb2gray(img)
io.imshow(img_gray)

#cv.waitKey(0)
#cv.destroyAllWindows()

#plt.hist(img_gray.falt, bins=100, range=(0,80))

# Plot the histogram
plt.figure()
plt.hist(img_gray.ravel(), bins=612, range=(0.1, 0.9), fc='black', ec='black')
plt.title('Histogram of Grayscale Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

#img_gray.ravel() flattens the 2D grayscale image into a 1D array, which is required for plt.hist().
#bins=256 specifies the number of bins for the histogram. Since the image is grayscale, the intensity values range from 0 to 1, and 256 bins will give a detailed distribution.
#range=(0.0, 1.0) sets the range of pixel intensities for the histogram.
#fc='black' and ec='black' set the face and edge color of the histogram bars to black.

seg0 = (img_gray <= 0.1)
seg1 = (img_gray > 0.1) & (img_gray < 0.6)
seg2 = (img_gray >= 0.4)

print(img_gray.shape[1])

all_segments = np.zeros((img_gray.shape[0], img_gray.shape[1], 3))


all_segments[seg0] = (1, 1, 1)
all_segments[seg1] = (0, 1, 0)
all_segments[seg2] = (1, 1, 1)

io.imshow(all_segments)
