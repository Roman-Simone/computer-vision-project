import os
import cv2
import numpy as np
from utils import *
from cameraInfo import *
from homographyMatrix import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)

path_video_1 = os.path.join(parent_path, "data/dataset/video/out1.mp4")
path_video_2 = os.path.join(parent_path, "data/dataset/video/out2.mp4")
calibration_file = os.path.join(parent_path, "data/calibrationMatrix/calibration.pkl")
homographic_file = os.path.join(parent_path, "data/homographyMatrix/H_12.pkl")
fundamental_file = os.path.join(parent_path, "data/fundamentalMatrix/F_23.pkl")

def find_points_sift(img1, img2):
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches


def transform_points(kp1, kp2, good_matches):
    pts1 = []
    pts2 = []
    for i, m in enumerate(good_matches):
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2


def calculate_F_matrix():

    video_capture_1 = cv2.VideoCapture(path_video_1)
    video_capture_2 = cv2.VideoCapture(path_video_2)
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    camera_number_1 = 1
    camera_number_2 = 2

    camera_info_1 = find_cameraInfo(camera_number_1, calibration_file)
    camera_info_2 = find_cameraInfo(camera_number_2, calibration_file)

    frame1, frame2 = undistortedAndCrop(frame1, frame2, camera_info_1, camera_info_2)

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)   

    H = load_pickle(homographic_file)
    _, frame2_gray = apply_homography(frame1_gray, frame2_gray, H)


    plt.figure(figsize=(20, 10))
    plt.subplot(121), plt.imshow(frame1_gray, cmap='gray'), plt.title('Image 1')
    plt.subplot(122), plt.imshow(frame2_gray, cmap='gray'), plt.title('Image 2')
    plt.show()

    kp1, kp2, good_matches = find_points_sift(frame1_gray, frame2_gray)

    pts1, pts2 = transform_points(kp1, kp2, good_matches)
        
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    
    print(F)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    save_pickle(F, fundamental_file)


if __name__ == '__main__':
    calculate_F_matrix()

