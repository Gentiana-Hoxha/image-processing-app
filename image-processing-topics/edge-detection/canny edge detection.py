import cv2 as cv

# Read the input image
images_folder_path = "C:/Users/STORM/Desktop/images/"
cells_image_path = images_folder_path + "images_of_cells.jpg"

def callback(input):
    pass

def cannyEdgeDetection():
    img = cv.imread(cells_image_path, 0)
    
    windowName = 'Canny Edge Detection'
    cv.namedWindow(windowName)
    cv.createTrackbar('minTreshhold', windowName, 0, 255, callback)
    cv.createTrackbar('maxTreshhold', windowName, 0, 255, callback)
    
    while True:
        if cv.waitKey(1) == ord('q'):
            break
        
        minThresh = cv.getTrackbarPos('minTreshhold', windowName)
        maxThresh = cv.getTrackbarPos('maxTreshhold', windowName)
        cannyEdge = cv.Canny(img, minThresh, maxThresh)
        cv.imshow(windowName, cannyEdge)
        
    cv.destroyAllWindows()

if __name__ == '__main__':
    cannyEdgeDetection()
    