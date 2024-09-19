from prova_BONNIE import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
cap = cv2.VideoCapture('/home/bonnie/Desktop/computer vision/project/Computer_Vision_project/23_09_23 amichevole trento volley/out1.mp4')

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error reading video file")

# Convert the frame to grayscale
img1 = cv2.cv2tColor(frame, cv2.COLOR_BGR2GRAY)

# Release the video capture object
cap.release()

cap = cv2.VideoCapture('/home/bonnie/Desktop/computer vision/project/Computer_Vision_project/23_09_23 amichevole trento volley/out2.mp4')

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error reading video file")

# Convert the frame to grayscale
img2 = cv2.cv2tColor(frame, cv2.COLOR_BGR2GRAY)

# Release the video capture object
cap.release()


pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
 
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Next we find the epilines. Epilines corresponding to the points in first image is drawn on second image. So mentioning of correct images are important here. We get an array of lines. So we define a new function to draw these lines on the images.
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Now we find the epilines in both the images and draw them.
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
 
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
 
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()