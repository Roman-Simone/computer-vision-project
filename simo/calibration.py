
import os
import numpy as np
import cv2
import re
import csv
from pathlib import Path
from cameraInfo import CameraInfo
import pickle

def salva_camerasInfo_pickle(camerasInfo, filename):
    with open(filename, 'wb') as file:
        pickle.dump(camerasInfo, file)


def carica_camerasInfo_pickle(filename):
    with open(filename, 'rb') as file:
        camerasInfo = pickle.load(file)
    return camerasInfo


def trova_file_mp4(cartella):
    file_mp4 = []
    for file in os.listdir(cartella):
        if file.endswith(".mp4"):
            file_mp4.append(file)
    return file_mp4


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
    skip_frames = 15

    while True:

        frame_count += 1
        # Read a frame from the video
        ret, img = video_capture.read()
        if not ret:
            break  # Break the loop if we've reached the end of the video
        if frame_count % skip_frames  != 0:
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

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.join(current_path, os.pardir)
    parent_path = os.path.abspath(parent_path)

    path_videos = os.path.join(parent_path, 'dataset/calibration')


    videosCalibration = trova_file_mp4(path_videos)
    
    for video in videosCalibration:
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])
        pos_camera = [camera.camera_number for camera in camerasInfo].index(numero_camera)

        print("Starting calibration for camera ", numero_camera)
        
        path_video = os.path.join(path_videos, video)

        camerasInfo[pos_camera].objpoints, camerasInfo[pos_camera].imgpoints, gray = findPoints(path_video, camerasInfo[pos_camera], debug=True)

        ret, camerasInfo[pos_camera].mtx, camerasInfo[pos_camera].dist, camerasInfo[pos_camera].rvecs, camerasInfo[pos_camera].tvecs = cv2.calibrateCamera(camerasInfo[pos_camera].objpoints, camerasInfo[pos_camera].imgpoints, gray.shape[::-1], None, None)

        h,  w = gray.shape[:2]

        camerasInfo[pos_camera].newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camerasInfo[pos_camera].mtx, camerasInfo[pos_camera].dist, (w,h), 1, (w,h))

    
    salva_camerasInfo_pickle(camerasInfo, "calibration.pkl")



if __name__ == '__main__':

    all_chessboard_sizes = {1: (5, 7), 2: (5, 7), 3: (5, 7), 4: (5, 7), 5: (6, 9), 6: (6, 9), 7: (5, 7), 8: (6, 9), 12: (5, 7), 13: (5, 7)}
    
    camerasInfo = []
    
    for camera_number in all_chessboard_sizes.keys():
        camera = CameraInfo(camera_number)
        camera.chessboard_size = all_chessboard_sizes[camera_number]
        camerasInfo.append(camera)
    


    compute_calibration(camerasInfo)

