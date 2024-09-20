from prova_BONNIE import *
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
 
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt

def find_points_on_line(img, pt1, pt2):
    ''' Find points on a line segment in an image '''
    pts = []
    rows, cols = img.shape[:2]
    # Define line segment endpoints
    x0, y0 = pt1
    x1, y1 = pt2
    # Calculate the line equation parameters
    A = y1 - y0
    B = x0 - x1
    C = x1*y0 - x0*y1
    for x in range(cols):
        for y in range(rows):
            if abs(A*x + B*y + C) < 1:  # Tolerance for the line
                pts.append((x, y))
    return pts

def project_points_on_line(pt1, pt2, img):
    ''' Project points from one image to another along a line '''
    # Draw the line in the first image
    line_img = cv2.line(np.zeros_like(img), pt1, pt2, (255, 255, 255), 1)
    points_on_line = np.argwhere(line_img[:, :, 0] == 255)
    return points_on_line

# Load the first video
cap = cv2.VideoCapture('/home/bonnie/Desktop/computer vision/project/Computer_Vision_project/23_09_23 amichevole trento volley/out1.mp4')
ret, frame = cap.read()
if not ret:
    print("Error reading video file")
img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

# Load the second video
cap = cv2.VideoCapture('/home/bonnie/Desktop/computer vision/project/Computer_Vision_project/23_09_23 amichevole trento volley/out2.mp4')
ret, frame = cap.read()
if not ret:
    print("Error reading video file")
img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

# Load the JSON file
with open('/home/bonnie/Desktop/computer vision/project/Computer_Vision_project/data/world_points_all_cameras.json', 'r') as file:
    points = json.load(file)

num_cam1 = 1
num_cam2 = 2

# Create dictionaries for image coordinates
image_coords_1 = {tuple(item['world_coordinate']): item['image_coordinate'] for item in points[str(num_cam1)]}
image_coords_2 = {tuple(item['world_coordinate']): item['image_coordinate'] for item in points[str(num_cam2)]}

# Find common points
common_points = sorted(set(image_coords_1.keys()) & set(image_coords_2.keys()))

# Crea liste ordinate di coordinate delle immagini
pts1 = [image_coords_1[point] for point in common_points]
pts2 = [image_coords_2[point] for point in common_points]

# Stampa le coordinate ordinate
print("Ordered Image Coordinates for Camera 1:")
for coord in pts1:
    print(coord)

print("\nOrdered Image Coordinates for Camera 2:")
for coord in pts2:
    print(coord)


# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in ascending order of distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
def thick_lines(img_matches, matches, kp1, kp2, thickness=3):
    for match in matches:
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
        cv2.line(img_matches, pt1, (pt2[0] + img1.shape[1], pt2[1]), (0, 255, 0), thickness)
    return img_matches

# Amplia le linee delle corrispondenze
img_matches_thick = thick_lines(img_matches, matches[:10], kp1, kp2)

# Show the result
plt.imshow(img_matches)
plt.show()



pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
 
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