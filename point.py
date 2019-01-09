import cv2
import numpy as np


img = cv2.imread("922.jpeg")
h,w,c = img.shape
img1 = cv2.imread("941.jpeg")
img2 = cv2.imread("1017.jpeg")
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)[:10]

for i in range(len(matches)):
    b = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    r = np.random.randint(0, 255)
    point = np.uint32(kp1[matches[i].queryIdx].pt)

    cv2.circle(img1, (point[0], point[1]), 5, (b, g, r), -1)

    point1 = np.uint32(kp2[matches[i].trainIdx].pt)

    cv2.circle(img2, (point1[0], point1[1]), 5, (b, g, r), -1)

#print matches[1].trainIdx
    #print matches[i].queryIdx
    #print matches[i].distance

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)

cv2.imshow("2",img3)
cv2.imshow("1",img1)
cv2.imshow("3",img2)
cv2.waitKey(0)



