import os
import cv2
import numpy as np
from utils import *
from matplotlib import pyplot as plt

#GLOBAL VARIABLES
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)

path_video_1 = os.path.join(parent_path, "data/dataset/video/out1.mp4")
path_video_2 = os.path.join(parent_path, "data/dataset/video/out2.mp4")
calibration_file = os.path.join(parent_path, "data/calibrationMatrix/calibration.pkl")
homographic_file = os.path.join(parent_path, "data/homographyMatrix/H_23.pkl")

camera_number_1 = 4
camera_number_2 = 3

def find_homographic(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Compute the inverse homography to warp image 2 to image 1 space
    H_inv = np.linalg.inv(H)
    
    return H, H_inv


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

def apply_homography(img1, img2, H):

    # Warp image 1 to image 2 space
    h1, w1 = img1.shape

    # Compute the inverse homography to warp image 2 to image 1 space
    H_inv = np.linalg.inv(H)
    img_warped = cv2.warpPerspective(img2, H_inv, (w1, h1))

    # Calculate intersection
    intersection = cv2.bitwise_and(img1, img_warped)

    return intersection, img_warped

def calculate_H_matrix():
    video_capture_1 = cv2.VideoCapture(path_video_1)
    video_capture_2 = cv2.VideoCapture(path_video_2)
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    camera_info_1 = find_cameraInfo(camera_number_1, calibration_file)
    camera_info_2 = find_cameraInfo(camera_number_2, calibration_file)

    frame1, frame2 = undistortedAndCrop(frame1, frame2, camera_info_1, camera_info_2)

    
    # cv2.imshow('frame1', frame1)
    # cv2.imshow('frame2', frame2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    kp1, kp2, good_matches = find_points_sift(frame1_gray, frame2_gray)

    # Find common area using homography
    H, H_inv= find_homographic(kp1, kp2, good_matches)
    save_pickle(H, homographic_file)

    intersection1, img2_warped = apply_homography(frame1_gray, frame2_gray, H)
    intersection2, img1_warped = apply_homography(frame2_gray, frame1_gray, H_inv)


    # Visualize the warped images, intersections, and warped back image
    plt.figure(figsize=(20, 10))
    plt.subplot(231), plt.imshow(frame1, cmap='gray'), plt.title('Image 1')
    plt.subplot(232), plt.imshow(img1_warped, cmap='gray'), plt.title('Warped Image 1')
    plt.subplot(233), plt.imshow(intersection1, cmap='gray'), plt.title('Intersection 1')

    plt.subplot(234), plt.imshow(frame2, cmap='gray'), plt.title('Image 2')
    plt.subplot(235), plt.imshow(img2_warped, cmap='gray'), plt.title('Warped Image 2')
    plt.subplot(236), plt.imshow(intersection2, cmap='gray'), plt.title('Intersection 2')

    plt.show()


if __name__ == '__main__':
    calculate_H_matrix()
