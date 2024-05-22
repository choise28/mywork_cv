import cv2 as cv
import numpy as np

src = cv.imread('apples.jpg', cv.IMREAD_GRAYSCALE)
rows, cols = src.shape[0:2]

src = cv.resize(src, (int(cols*0.5), int(rows*0.5)), interpolation=cv.INTER_CUBIC)

# 라플라시안 필터 적용 ---①
edge = cv.Laplacian(src, -1)

# 결과 출력
merged = np.hstack((src, edge))
cv.imshow('Laplacian', merged)
cv.waitKey(0)
cv.destroyAllWindows()

