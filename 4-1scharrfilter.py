import cv2 as cv
import numpy as np

src = cv.imread('apples.jpg', cv.IMREAD_GRAYSCALE)
rows, cols = src.shape[0:2]

src = cv.resize(src, (int(cols*0.5), int(rows*0.5)), interpolation=cv.INTER_CUBIC)

# 샤르 커널을 직접 생성해서 엣지 검출 ---①
gx_k = np.array([[-3,0,3], [-10,0,10],[-3,0,3]])
gy_k = np.array([[-3,-10,-3],[0,0,0], [3,10,3]])
edge_gx = cv.filter2D(src, -1, gx_k)
edge_gy = cv.filter2D(src, -1, gy_k)

# 샤르 함수 이용한 엣지 검출 ---②
scharrx = cv.Scharr(src, -1, 1, 0)
scharry = cv.Scharr(src, -1, 0, 1)

# 결과 출력
merged1 = np.hstack((src, edge_gx, edge_gy))
merged2 = np.hstack((src, scharrx, scharry))
merged = np.vstack((merged1, merged2))
cv.imshow('Scharr', merged)
cv.waitKey(0)
cv.destroyAllWindows()

