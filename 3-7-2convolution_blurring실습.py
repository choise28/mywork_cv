import cv2 as cv
import numpy as np

img=cv.imread('girl_laughing.jpg')
img=cv.resize(img,dsize=(0,0),fx=0.4,fy=0.4)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Averaging filter
avg = np.array([[1/9, 1/9, 1/9],
                  [ 1/9, 1/9, 1/9],
                  [ 1/9, 1/9, 1/9]])

# Convolution
gray_avr = cv.filter2D(gray,-1,avg)

gray_avr2 = cv.filter2D(gray_avr,-1,avg)

cv.imshow('Original',img)
cv.imshow('gray image',gray)
cv.imshow('Averaging filtered image 1',gray_avr)
cv.imshow('Averaging filtered image 2',gray_avr2)

cv.waitKey()
cv.destroyAllWindows()
