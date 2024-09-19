import os
import re
import cv2
import numpy as np
from utils import *
from cameraInfo import *


current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)
path_videos_calibration = os.path.join(parent_path, 'data/dataset/calibration')
path_videos = os.path.join(parent_path, 'data/dataset/video')
path_calibrationMTX = os.path.join(parent_path, 'data/calibrationMatrix/calibration.pkl')

valid_camera_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
all_chessboard_sizes = {1: (5, 7), 2: (5, 7), 3: (5, 7), 4: (5, 7), 5: (6, 9), 6: (6, 9), 7: (5, 7), 8: (6, 9), 12: (5, 7), 13: (5, 7)}

SKIP_FRAME = 15


def findPoints(path_video, cameraInfo, debug=True):

    chess_width = cameraInfo.chessboard_size[0]
    chess_height = cameraInfo.chessboard_size[1]

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_width * chess_height,3), np.float32)
    objp[:,:2] = np.mgrid[0:chess_height , 0:chess_width].T.reshape(-1,2)


    # Arrays to store object points and image points from all the images.
    retObjpoints = [] # 3d point in real world space
    retImgpoints = [] # 2d points in image plane.

    video_capture = cv2.VideoCapture(path_video)


    # Print the number of frames in the video
    numberOf_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames in the video: ", numberOf_frame)

    if debug:
        # Create a directory to save the screen captures named as the video file
        output_dir = f"samples/Camera{cameraInfo.camera_number}"
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("Saving frames to ", output_dir)
        # Remove all files in the directory
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        

    frame_count = 0

    while True:

        frame_count += 1
        # Read a frame from the video
        ret, img = video_capture.read()
        if not ret:
            break  # Break the loop if we've reached the end of the video
        if frame_count % SKIP_FRAME  != 0:
            continue


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (chess_height, chess_width), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            ret_gray = gray
            
            retObjpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            retImgpoints.append(corners2)

            if debug:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (chess_height, chess_width), corners2, ret)
                # Save the image with detected corners
                cv2.imwrite(f"{output_dir}/frame{frame_count}.jpg", img)
    
    return retObjpoints, retImgpoints, ret_gray


def compute_calibration(camerasInfo):

    videosCalibration = find_file_mp4(path_videos_calibration)
    
    for video in videosCalibration:
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])
        # pos_camera = [camera.camera_number for camera in camerasInfo].index(numero_camera)
        pos_camera = numero_camera - 1

        # print("Starting calibration for camera ", numero_camera, pos_camera)
        
        path_video = os.path.join(path_videos_calibration, video)

        camerasInfo[pos_camera].objpoints, camerasInfo[pos_camera].imgpoints, gray = findPoints(path_video, camerasInfo[pos_camera], debug=False)

        ret, camerasInfo[pos_camera].mtx, camerasInfo[pos_camera].dist, camerasInfo[pos_camera].rvecs, camerasInfo[pos_camera].tvecs = cv2.calibrateCamera(camerasInfo[pos_camera].objpoints, camerasInfo[pos_camera].imgpoints, gray.shape[::-1], None, None)

        h,  w = gray.shape[:2]

        camerasInfo[pos_camera].newcameramtx, camerasInfo[pos_camera].roi = cv2.getOptimalNewCameraMatrix(camerasInfo[pos_camera].mtx, camerasInfo[pos_camera].dist, (w,h), 1, (w,h))

    
    save_pickle(camerasInfo, "calibration.pkl")
    return camerasInfo


def calibrate():
    camerasInfo = []
    
    for camera_number in all_chessboard_sizes.keys():
        camera = CameraInfo(camera_number)
        camera.chessboard_size = all_chessboard_sizes[camera_number]
        camerasInfo.append(camera)
    
    camerasInfo = compute_calibration(camerasInfo)


def test_calibration():

    videos = find_file_mp4(path_videos)
    camera_infos = load_pickle(path_calibrationMTX)

    for video in videos:

        camera_number = re.findall(r'\d+', video.replace(".mp4", ""))
        camera_number = int(camera_number[0])
        if camera_number not in valid_camera_numbers:
            continue

        # open the video
        camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)
        path_video = os.path.join(path_videos, video)
        video_capture = cv2.VideoCapture(path_video)

        # Show the video
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            undistorted_frame = undistorted(frame, camera_info)
            undistorted_frame = cv2.resize(undistorted_frame, (frame.shape[1], frame.shape[0]))
            comparison_frame = np.hstack((frame, undistorted_frame))

            cv2.imshow('Original (Left) vs Undistorted (Right)', comparison_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':

    # CALIBRATE THE CAM
    # calibrate()

    # ONLY FOR TESTING
    test_calibration()

