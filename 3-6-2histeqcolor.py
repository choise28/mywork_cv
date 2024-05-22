import cv2 as cv

src = cv.imread('beatles01.jpg')

src_ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb) # YCbCr 색공간으로 변경
ycrcb_planes = list(cv.split(src_ycrcb)) # Y, Cr, Cb 나누기

# 밝기성분(Y)에 대해서만 히스토그램 평활화 수행
ycrcb_planes[0] = cv.equalizeHist(ycrcb_planes[0])

dst_ycrcb = cv.merge(ycrcb_planes) # Y, Cr, Cb 합치기
dst = cv.cvtColor(dst_ycrcb, cv.COLOR_YCrCb2BGR) # YCbCr --> RGB 색공간으로 변경

cv.imshow('src', src)
cv.imshow('dst', dst)
cv.waitKey()

cv.destroyAllWindows()