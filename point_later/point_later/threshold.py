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
############Chap anh############################################
kernel = np.ones((3, 3), np.float32)/9
image = cv2.filter2D(image_gray, -1, kernel)


########### Loc Gaussian ######################################

image = cv2.GaussianBlur(image_gray, (3, 3), 0)

########### Loc Trung vi ######################################

image = cv2.medianBlur(image_gray, 5)

########### Loc song phuong ####################################

blur = cv2.bilateralFilter(img,9,75,75)

############### bien doi hinh thai hoc #######  anh nhi phan
img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1) # thu nho

dilation = cv2.dilate(img,kernel,iterations = 1) # phong to

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

laplace = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

edges = cv2.Canny(img,100,200)

################# CONTOUR ##############################
_, contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
x, y, w, h = cv2.boundingRect(contour[0])

cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
