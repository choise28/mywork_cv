import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# read image
gray=cv.imread('abnormal.jpg', cv.IMREAD_GRAYSCALE)

# histogram stretching
stretch = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)

# histogram equalization
equal = cv.equalizeHist(gray)


gc=np.hstack((gray, stretch, equal))

cv.imshow('original image, histogram stretch, histogram equalization ', gc)


# plot histogram
fig = plt.figure()
rows = 1
cols = 3

ax1 = fig.add_subplot(rows, cols, 1)
h = cv.calcHist([gray], [0], None, [256], [0,256])
ax1.plot(h,color='r', linewidth=1)

ax2 = fig.add_subplot(rows, cols, 2)
h = cv.calcHist([stretch], [0], None, [256], [0,256])
ax2.plot(h,color='r', linewidth=1)

ax3 = fig.add_subplot(rows, cols, 3)
h = cv.calcHist([equal], [0], None, [256], [0,256])
ax3.plot(h,color='r', linewidth=1)
plt.show()

cv.waitKey()
cv.destroyAllWindows()
