import cv2 as cv
import numpy as np

img_color = cv.imread('soccer.jpg')

src_color = cv.resize(img_color, dsize=(0,0), fx=0.25, fy=0.25)
src_gray = cv.cvtColor(src_color, cv.COLOR_BGR2GRAY)

a = -50

### gray image
dst_gray = cv.add(src_gray, a)

cv.imshow('src gray image', src_gray)
cv.imshow('dst gray image', dst_gray)


### color image
dst_color = cv.add(src_color, (-50, -50, -50, 0))

cv.imshow('src color image', src_color)
cv.imshow('dst color image', dst_color)


### 색상 반전
dst_inv = src_color.copy()
# dst_inv = 255 - dst_inv # 직접 계산
dst_inv = np.invert(src_color) # numpy
# dst_inv = cv.bitwise_not(src_color) # opencv

cv.imshow('src image', src_color)
cv.imshow('dst inv image', dst_inv)

cv.waitKey()
cv.destroyAllWindows()

