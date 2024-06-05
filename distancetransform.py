import cv2 as cv
import numpy as np

# 이미지를 읽어서 바이너리 스케일로 변환
img = cv.imread('rectangle.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, biimg = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 거리 변환
dst = cv.distanceTransform(biimg, cv.DIST_L2, 5)

# 거리 값을 0 ~ 255 범위로 정규화
dst = (dst/(dst.max()-dst.min()) * 255).astype(np.uint8)

# 결과 출력
cv.imshow('origin', img)
cv.imshow('distance_transform', dst)

cv.waitKey(0)
cv.destroyAllWindows()

