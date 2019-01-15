import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
#im = cv2.imread("1017.jpeg",cv2.IMREAD_GRAYSCALE)
#plt.hist(im.ravel(),256,[0,256])


#histr = cv2.calcHist([im],[0],None,[256],[0,256]) # i: channel, None : Mask(100:200,200:300)
#print len(histr)
#plt.plot(histr)
#plt.xlim([0,256])




I = cv2.imread("1017.jpeg")
b,g,r = cv2.split(I)
rgb_img = cv2.merge([r,g,b])
r,c,n = I.shape
S = np.ones((r,c,3), np.uint8)
for i in range(r):
    for j in range(c):
        if j==1:
            S[i][j][0] = I[i][j][0]
            S[i][j][1] = I[i][j][1]
            S[i][j][2] = I[i][j][2]
        else:
            S[i][j][0] = 2*I[i][j][0] - I[i][j-1][0]
            S[i][j][1] = 2*I[i][j][1] - I[i][j-1][1]
            S[i][j][2] = 2*I[i][j][2] - I[i][j-1][2]

b,g,r = cv2.split(S)
fig = plt.figure()
gs = plt.GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlabel('volts')
ax1.set_title('HISTOGRAM',)
ax1.imshow(rgb_img)
ax1.set_xticks([])
ax1.set_yticks([])

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('histogram')
ax2.hist(r.ravel(),256,[0,256])



plt.show()




G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
kernel = np.ones((4,4),np.float32)/16
J = cv2.filter2D(G,-1,kernel)
G = np.double(G)
K = G-J
U = 0.5*K+J
U = np.uint8(U)

plt.hist(U.ravel(),256,[0,256])

#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv2.calcHist([img],[2],None,[256],[0,256]) # i: channel, None : Mask(100:200,200:300)
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])


#xoay anh

img = cv2.imread('1017.jpeg')
rows,cols,c = img.shape
rows1 = np.uint32(math.sqrt(rows*rows+cols*cols))
cols1 = np.uint32(math.sqrt(rows*rows+cols*cols))
M = cv2.getRotationMatrix2D((280,110),45,1)
dst = cv2.warpAffine(img,M,(rows1,cols1))
cv2.imshow("1",dst)
cv2.waitKey(0)
