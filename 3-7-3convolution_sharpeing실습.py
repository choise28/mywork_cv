import cv2 as cv
import numpy as np

img=cv.imread('girl_laughing.jpg')
img=cv.resize(img,dsize=(0,0),fx=0.5,fy=0.5)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

gray16 = gray

# Sharpening filter
sharp = np.array([[-1, -1, -1],
                  [ -1, 9, -1],
                  [ -1, -1, -1]])


# Convolution
gray_sharp = cv.filter2D(gray16,-1,sharp)

cv.imshow('Original', img)
cv.imshow('gray image', gray)
cv.imshow('Sharpening filtered image', gray_sharp)

cv.waitKey()
cv.destroyAllWindows()
