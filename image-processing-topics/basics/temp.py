####################################### Imports >

from skimage import io
import numpy as np

#ploting library #pyplot to create 2d plots and graphs #similar as matlab
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
 
import numpy as np
#pillow - The Python Imaging Library adds image processing capabilities to your Python interpreter.
from PIL import Image

#image processing library, image segemntation, feature detection, a lot other stuff
from skimage import io #read images, shows the same as matplot 

#open_cv
#open_cv is a library of programming functions used to manipulate matrices(images videos)
import cv2 as cv


####################################### File >

images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

####################################### < File 

#skimage lib - reading image file
cells_image = io.imread(cells_image_path)
print(type(cells_image)) #<class 'numpy.ndarray'>
#print(cells_image.format) AttributeError:'numpy.ndarray' object has no attribute 'format'

img = Image.open(cells_image_path);
print(type(img)) #<class 'PIL.JpegImagePlugin.JpegImageFile'>
print(img.format) #JPEG

img.show() #opens new image tab

#convert Pil Image into a numpy array
numpy_image = np.asarray(img);
print(type(numpy_image)) #<class 'numpy.ndarray'>

#convert image to matplot
matplot_image = mpimg.imread(cells_image_path)
print(type(matplot_image)) #<class 'numpy.ndarray'>
print(matplot_image.shape) #(331, 504, 3)
plt.imshow(matplot_image) #shows scales on the logs
plt.colorbar() 

#scikit-image
image_ski = io.imread(cells_image_path)
print(type(image_ski)) #<class 'numpy.ndarray'>

#can be used for converting
#from skimage import img_as_float, img_as_ubyte
# img_float = img_as_float(image_ski)
# img_float = io.imread("...").astype(np.float) doest work correctly, exp 92 -> 92.0

image_cv_grey = cv.imread(cells_image_path, 0);
image_cv = cv.imread(cells_image_path, 1);

print(type(image_cv)) #<class 'numpy.ndarray'>

#By default it import images as BRG and not RGB that's why display cv image with plt looks different
plt.imshow(cv.cvtColor(image_cv, cv.COLOR_BGR2RGB))# change color space
#plt.imshow(image_cv) #shows with different colors

cv.imshow("Grey Image", image_cv_grey);
cv.imshow("Colored Image", image_cv);

cv.waitKey(0)
cv.destroyAllWindows()


#czifile, for microscope images
#OME-TIFF type
import glob

path = images_folder_path + "*"
for file in glob.glob(path):
    print(file)
    