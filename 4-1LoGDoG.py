import cv2 as cv
import numpy as np

src = cv.imread('apples.jpg', cv.IMREAD_GRAYSCALE)
rows, cols = src.shape[0:2]

src = cv.resize(src, (int(cols*0.5), int(rows*0.5)), interpolation=cv.INTER_CUBIC)

# LoG
gaus = cv.GaussianBlur(src, (3, 3), 0, 0)            # 가우시안 마스크 적용
log_edge = cv.Laplacian(gaus, -1)             # 라플라시안 수행

# DoG
gaus1 = cv.GaussianBlur(src, (3, 3), 0)          # 가우시안 블러링
gaus2 = cv.GaussianBlur(src, (9, 9), 0)
dog_edge = cv.absdiff(gaus1, gaus2)          # DoG 수행


# 결과 출력
merged = np.hstack((src, log_edge, dog_edge))
cv.imshow('LoG, DoG', merged)
cv.waitKey(0)
cv.destroyAllWindows()

