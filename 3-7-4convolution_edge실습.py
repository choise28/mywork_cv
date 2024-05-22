import cv2 as cv
import numpy as np

img=cv.imread('girl_laughing.jpg')
img=cv.resize(img,dsize=(0,0),fx=0.4,fy=0.4)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Edge filter
edge = np.array([[-1, 0, 1],
                  [ -1, 0, 1],
                  [ -1, 0, 1]])

# edge = np.array([[-1, -1, -1],
#                   [ 0, 0, 0],
#                   [ 1, 1, 1]])


# Convolution
gray_edge = cv.filter2D(gray,-1,edge)

cv.imshow('Original', img)
cv.imshow('gray image', gray)
cv.imshow('Sharpening filtered image', gray_edge)

cv.waitKey()
cv.destroyAllWindows()
