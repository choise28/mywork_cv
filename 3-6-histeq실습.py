import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('beatles01.jpg')

fig = plt.figure()
rows = 3
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)		# 명암 영상으로 변환하고 출력
ax1.imshow(gray,cmap='gray'), plt.xticks([]), plt.yticks([])
ax1.set_title('gray image')

ax2 = fig.add_subplot(rows, cols, 2)
h=cv.calcHist([gray],[0],None,[256],[0,256])		# 히스토그램을 구해 출력
ax2.plot(h,color='r',linewidth=1)

ax3 = fig.add_subplot(rows, cols, 3)
equal=cv.equalizeHist(gray)			# 히스토그램을 평활화하고 출력
ax3.imshow(equal,cmap='gray'), plt.xticks([]), plt.yticks([])
ax3.set_title('Histogram equalization')

ax4 = fig.add_subplot(rows, cols, 4)
h=cv.calcHist([equal],[0],None,[256],[0,256])		# 히스토그램을 구해 출력
ax4.plot(h,color='r',linewidth=1)

# 히스토그램 스트래칭
ax5 = fig.add_subplot(rows, cols, 5)
stretch = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
ax5.imshow(stretch, cmap='gray'), plt.xticks([]), plt.yticks([])
ax5.set_title('Histogram normalization')

ax6 = fig.add_subplot(rows, cols, 6)
h=cv.calcHist([stretch],[0],None,[256],[0,256])		# 히스토그램을 구해 출력
ax6.plot(h,color='r',linewidth=1)
plt.show()
