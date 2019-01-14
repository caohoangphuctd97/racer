import numpy as np
import cv2



image = cv2.imread("anh.png")
image = image[200:1000,470:1500]
ima_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
imgHSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

lower = np.array([40])
upper = np.array([80])
threshold = cv2.inRange(ima_gray, lower, upper)
h,w = threshold.shape
axisYX = []
for i in range(h):
    for j in range(w):
        if threshold[i][j] == 255:
            axisYX.append([i, j])


for x in range(len(axisYX)):
    image[axisYX[x][0]][axisYX[x][1]][0] = 66
    image[axisYX[x][0]][axisYX[x][1]][1] = 169
    image[axisYX[x][0]][axisYX[x][1]][2] = 132
img = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
cv2.imshow("image_processed.jpg", img)
#ret, thre = cv2.threshold(ima_gray,150,255, cv2.THRESH_BINARY_INV)

cv2.imshow("image",threshold)
cv2.imshow("threshold",image)
cv2.waitKey(0)