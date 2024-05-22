import cv2 as cv

src = cv.imread('apples.jpg', cv.IMREAD_GRAYSCALE)
rows, cols = src.shape[0:2]

src = cv.resize(src, (int(cols*0.5), int(rows*0.5)), interpolation=cv.INTER_CUBIC)

# canny edge
dst = cv.Canny(src, 50, 150)

cv.imshow('src', src)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

