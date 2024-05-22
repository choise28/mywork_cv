import cv2 as cv
import numpy as np

src = cv.imread('apples.jpg', cv.IMREAD_GRAYSCALE)
rows, cols = src.shape[0:2]

src = cv.resize(src, (int(cols*0.5), int(rows*0.5)), interpolation=cv.INTER_CUBIC)

# 소벨 커널을 직접 생성해서 엣지 검출 ---①
## 소벨 커널 생성
gx_k = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
gy_k = np.array([[-1,-2,-1],[0,0,0], [1,2,1]])
## 소벨 필터 적용
edge_gx = cv.filter2D(src, -1, gx_k)
edge_gy = cv.filter2D(src, -1, gy_k)

# 소벨 함수 이용한 엣지 검출
sobelx = cv.Sobel(src, -1, 1, 0, ksize=3)
sobely = cv.Sobel(src, -1, 0, 1, ksize=3)

# 결과 출력
merged1 = np.hstack((src, edge_gx, edge_gy, edge_gx+edge_gy))
merged2 = np.hstack((src, sobelx, sobely, sobelx+sobely))
merged = np.vstack((merged1, merged2))
cv.imshow('sobel', merged)
cv.waitKey(0)
cv.destroyAllWindows()

