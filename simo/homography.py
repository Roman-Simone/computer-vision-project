import os
import re
import cv2
import itertools
import numpy as np
from utils import *
from cameraInfo import *


current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)
path_videos = os.path.join(parent_path, 'data/dataset/video')
path_calibrationMTX = os.path.join(parent_path, 'data/calibrationMatrix/calibration.pkl')

valid_camera_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]


def find_homography(frame1, frame2):
    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Match the descriptors
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # Draw the matches
    img_matches = cv2.drawMatches(frame1, kp1, frame2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matches
    cv2.imshow('Matches', img_matches)

    # Find the homography
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Draw the homography
        h, w = frame1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)

        frame2 = cv2.polylines(frame2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    cv2.imshow('Homography', frame2)


def homography():
    videos = find_file_mp4(path_videos)
    camera_infos = load_pickle(path_calibrationMTX)


    # all possible combinations of camera numbers
    combinations = list(itertools.combinations(valid_camera_numbers, 2))

    

    for camera1, camera2 in combinations:
        camera_info1 = next((cam for cam in camera_infos if cam.camera_number == camera1), None)
        camera_info2 = next((cam for cam in camera_infos if cam.camera_number == camera2), None)

        path_video1 = path_videos + f'/out{camera1}.mp4'
        path_video2 = path_videos + f'/out{camera2}.mp4'

        video_capture1 = cv2.VideoCapture(path_video1)
        video_capture2 = cv2.VideoCapture(path_video2)

         # Show the video
        while True:
            ret1, frame1 = video_capture1.read()
            ret2, frame2 = video_capture2.read()
            if not ret1 or not ret2:
                break

            find_homography(frame1, frame2)

            # undistorted_frame1 = undistorted(frame1, camera_info1)
            # undistorted_frame2 = undistorted(frame2, camera_info2)

            # undistorted_frame2 = cv2.resize(undistorted_frame2, (undistorted_frame1.shape[1], undistorted_frame1.shape[0]))
            # comparison_frame = np.hstack((undistorted_frame1, undistorted_frame2))

            # cv2.imshow('Original (Left) vs Undistorted (Right)', comparison_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == '__main__':
    
    homography()