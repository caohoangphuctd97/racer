import cv2
import numpy as np


def get_point(image1,image2,arr_point):
    point_later = []

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image1,None)
    kp2, des2 = orb.detectAndCompute(image2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)[0:10]
    pixelX=np.int32(0)
    pixelY=np.int32(0)
    for i in range(len(matches)):

        point = np.uint32(kp1[matches[i].queryIdx].pt)


        point1 = np.uint32(kp2[matches[i].trainIdx].pt)


        pixelX = pixelX + point1[0]-point[0]
        pixelY = pixelY + point1[1] - point[1]


    pixelX= pixelX/len(matches)
    pixelY = pixelY/len(matches)
    for i in range(len(arr_point)):
        X = arr_point[i][0] + pixelX
        Y = arr_point[i][1] + pixelY
        point_later.append([X,Y])
    return point_later
while(1):
    img2 = cv2.imread("895.jpeg")
    img1 = cv2.imread("883.jpeg")
    point = [[154,111],[7,215],[197,75],[271,75],[288,106],[319,148]]
    array_point = get_point(img1,img2,point)
    for i in range(len(point)):
        cv2.circle(img1, (point[i][0],point[i][1]), 7, (0, 0, 0), -1)
    for i in range(len(array_point)):

        cv2.circle(img2, (array_point[i][0],array_point[i][1]), 7, (0, 0, 0), -1)
    cv2.imshow("2",img2)
    cv2.imshow("1",img1)
    cv2.waitKey(0)



