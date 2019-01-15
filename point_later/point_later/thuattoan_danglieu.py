import numpy as np
import cv2
import matplotlib.pyplot as plt
import math 
im = cv2.imread('image1.jpg',0)
histr = cv2.calcHist([im],[0],None,[256],[0,256])
t = 256/2
epst = 1
mL=0
mH=0


while 1:
	a =0
	sum = 0
	for i in range(0,t+1):
		sum = sum + i*histr[i]
		a = a +histr[i]		
	mL = sum/a
	a=0
	sum = 0
	for i in range(t+1,256):
		sum = sum + i*histr[i]
		a = a +histr[i]
	mH = sum/a
	t_new = (mL+mH)/2
	if math.fabs(t-t_new) < epst:
		break
	t = np.uint8(t_new)           #gia tri nguong
_,thres = cv2.threshold(im,t,255,cv2.THRESH_BINARY)
cv2.imshow("thres",thres)
cv2.waitKey(0)