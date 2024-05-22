import cv2 as cv
import numpy as np

src = cv.imread('apples.jpg', cv.IMREAD_GRAYSCALE)
rows, cols = src.shape[0:2]

src = cv.resize(src, (int(cols*0.5), int(rows*0.5)), interpolation=cv.INTER_CUBIC)

# prewitt

# 프리윗 커널 생성
gx_k = np.array([[-1,0,1], [-1,0,1],[-1,0,1]])
gy_k = np.array([[-1,-1,-1],[0,0,0], [1,1,1]])

# 프리윗 커널 필터 적용
edge_gx = cv.filter2D(src, -1, gx_k)
edge_gy = cv.filter2D(src, -1, gy_k)

# 결과 출력
merged = np.hstack((src, edge_gx, edge_gy, edge_gx+edge_gy))

cv.imshow('prewitt', merged)
cv.waitKey(0)
cv.destroyAllWindows()

