import cv2
import numpy as np

img = cv2.imread("house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SimpleBlobDetector 생성
detector = cv2.SimpleBlobDetector_create()

# 키 포인트 검출
keypoints = detector.detect(gray)

# 키 포인트를 빨간색으로 표시
img = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255), \
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Blob", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

