import os
import cv2
import numpy as np
import pandas as pd
from utils import *
from cameraInfo import *
import matplotlib.pyplot as plt

def find_common_area(path_video_1, path_video_2, camera_info_1, camera_info_2):
    # Load frames from videos
    video_capture_1 = cv2.VideoCapture(path_video_1)
    video_capture_2 = cv2.VideoCapture(path_video_2)
    if not video_capture_1.isOpened() or not video_capture_2.isOpened():
        print("Error opening video file.")
        return None, None, None, None

    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()
    if not ret1 or not ret2:
        print("Error reading frames.")
        return None, None, None, None

    # Close videos
    video_capture_1.release()
    video_capture_2.release()

    # Remove distortion from images
    undistorted_frame_1 = cv2.undistort(frame1, camera_info_1.mtx, camera_info_1.dist, None, camera_info_1.newcameramtx)
    undistorted_frame_2 = cv2.undistort(frame2, camera_info_2.mtx, camera_info_2.dist, None, camera_info_2.newcameramtx)

    x1, y1, w1, h1 = camera_info_1.roi
    undistorted_frame_1 = undistorted_frame_1[y1:y1+h1, x1:x1+w1]

    x2, y2, w2, h2 = camera_info_2.roi
    undistorted_frame_2 = undistorted_frame_2[y2:y2+h2, x2:x2+w2]

    # Convert to grayscale
    gray1 = cv2.cvtColor(undistorted_frame_1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(undistorted_frame_2, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches using KNN
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print("Not enough matches found.")
        return None, None, None, None

    # Extract match points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography could not be computed.")
        return None, None, None, None

    # Transform the original frame corners
    h, w = gray1.shape
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    # Find the common area
    x_min = max(0, np.min(transformed_corners[:, 0, 0]))
    y_min = max(0, np.min(transformed_corners[:, 0, 1]))
    x_max = min(w, np.max(transformed_corners[:, 0, 0]))
    y_max = min(h, np.max(transformed_corners[:, 0, 1]))

    common_corners = [(int(x_min), int(y_min)), (int(x_max), int(y_min)), (int(x_min), int(y_max)), (int(x_max), int(y_max))]
    return common_corners, undistorted_frame_1, undistorted_frame_2, (x_min, y_min, x_max-x_min, y_max-y_min)

def display_cropped_frames(frame1, frame2, crop_rect):
    x, y, w, h = crop_rect
    # Ensure the rectangle coordinates are integers
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    cropped_frame1 = frame1[y:y+h, x:x+w]
    cropped_frame2 = frame2[y:y+h, x:x+w]

    # Display using matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(cropped_frame1, cv2.COLOR_BGR2RGB))
    ax1.set_title("Cropped Frame 1")
    ax2.imshow(cv2.cvtColor(cropped_frame2, cv2.COLOR_BGR2RGB))
    ax2.set_title("Cropped Frame 2")
    plt.show()

# Example usage
def main():
    # Paths and CameraInfo objects setup

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.join(current_path, os.pardir)
    parent_path = os.path.abspath(parent_path)

    calibration_file = os.path.join(parent_path, "data/calibrationMatrix/calibration.pkl")
    path_csv_file = os.path.join(parent_path, "data/fundamentalMatrix/points4Fundamentals.csv")
    path_video_1 = os.path.join(parent_path, "data/dataset/video/out1.mp4")
    path_video_2 = os.path.join(parent_path, "data/dataset/video/out2.mp4")
    output_path = os.path.join(parent_path, "data/fundamentalMatrix/fundamentalMatrix_1_2_manual.pkl")
    camera_infos = load_pickle(calibration_file)
    camera_number_1 = 1
    camera_number_2 = 2

    camera_info_1 = next((cam for cam in camera_infos if cam.camera_number == camera_number_1), None)
    camera_info_2 = next((cam for cam in camera_infos if cam.camera_number == camera_number_2), None)
    
    common_corners, frame1, frame2, crop_rect = find_common_area(path_video_1, path_video_2, camera_info_1, camera_info_2) 
    if frame1 is not None and frame2 is not None:
        display_cropped_frames(frame1, frame2, crop_rect)
    else:
        print("Could not display frames due to an error in processing.")

if __name__ == "__main__":
    main()
