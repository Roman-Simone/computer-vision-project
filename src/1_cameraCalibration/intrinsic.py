import re
import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
from cameraInfo import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

all_chessboard_sizes = {1: (5, 7), 2: (5, 7), 3: (5, 7), 4: (5, 7), 5: (6, 9), 6: (6, 9), 7: (5, 7), 8: (6, 9), 12: (5, 7), 13: (5, 7)}
SKIP_FRAME = 10

def findPoints(path_video, cameraInfo, debug=True):
    """
    Detects and collects chessboard corner points from frames in a video for camera calibration.

    Parameters:
        path_video (str): path to the video file.
        cameraInfo (CameraInfo): camera information object containing chessboard size and camera number.
        debug (bool): if True, saves frames with and without corners marked for debugging.

    Returns:
        tuple: tuple containing object points, image points, and the grayscale version of the last frame.
    """
    chess_width = cameraInfo.chessboard_size[0]
    chess_height = cameraInfo.chessboard_size[1]
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, 0.01)

    objp = np.zeros((chess_width * chess_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_height, 0:chess_width].T.reshape(-1, 2)

    retObjpoints = []  
    retImgpoints = []  

    video_capture = cv2.VideoCapture(path_video)
    numberOf_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nStarting calibrations for camera {cameraInfo.camera_number}")
    print("Number of frames in the video: ", numberOf_frame)

    if debug:
        output_dir_original = f"{PATH_FRAME_SAMPLES_CALIBRATION}/Camera{cameraInfo.camera_number}/originalImages"
        output_dir_original = os.path.join(os.path.dirname(__file__), output_dir_original)
        os.makedirs(output_dir_original, exist_ok=True)
        for file in os.listdir(output_dir_original):
            os.remove(os.path.join(output_dir_original, file))
        
        output_dir_corners = f"{PATH_FRAME_SAMPLES_CALIBRATION}/Camera{cameraInfo.camera_number}/withCorners"
        output_dir_corners = os.path.join(os.path.dirname(__file__), output_dir_corners)
        os.makedirs(output_dir_corners, exist_ok=True)
        for file in os.listdir(output_dir_corners):
            os.remove(os.path.join(output_dir_corners, file))

    frame_count = 0
    with tqdm(total=numberOf_frame, desc="Processing Video", unit="frame") as pbar:
        while True:
            frame_count += 1
            ret, img = video_capture.read()
            if not ret:
                break  
            if frame_count % SKIP_FRAME != 0:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (chess_height, chess_width), None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

            if ret:
                retObjpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
                retImgpoints.append(corners2)

                if debug:
                    cv2.imwrite(f"{output_dir_original}/frame{frame_count}.jpg", img)
                    cv2.drawChessboardCorners(img, (chess_height, chess_width), corners2, ret)
                    cv2.imwrite(f"{output_dir_corners}/frame{frame_count}.jpg", img)

            pbar.update(SKIP_FRAME)
    
    return retObjpoints, retImgpoints, gray


def compute_calibration_all(camerasInfo):
    """
    Computes the intrinsic calibration for all cameras by analyzing calibration videos.

    Parameters:
        camerasInfo (list): list of CameraInfo objects with camera metadata.

    Returns:
        list: updated list of CameraInfo objects with calibration parameters.
    """
    videosCalibration = find_files(PATH_VIDEOS_CALIBRATION)
    videosCalibration.sort()

    for video in videosCalibration:
        numero_camera = int(re.findall(r'\d+', video.replace(".mp4", ""))[0])
        _, pos_camera = take_info_camera(numero_camera, camerasInfo)
        path_video = os.path.join(PATH_VIDEOS_CALIBRATION, video)

        camerasInfo[pos_camera].objpoints, camerasInfo[pos_camera].imgpoints, gray = findPoints(path_video, camerasInfo[pos_camera], debug=True)

        if len(camerasInfo[pos_camera].objpoints) == 0:
            print(f"Camera {numero_camera} not calibrated - No points found")
            continue

        ret, camerasInfo[pos_camera].mtx, camerasInfo[pos_camera].dist, _, _ = cv2.calibrateCamera(
            camerasInfo[pos_camera].objpoints, camerasInfo[pos_camera].imgpoints, gray.shape[::-1], None, None
        )

        if ret:
            h, w = gray.shape[:2]
            camerasInfo[pos_camera].newcameramtx, camerasInfo[pos_camera].roi = cv2.getOptimalNewCameraMatrix(
                camerasInfo[pos_camera].mtx, camerasInfo[pos_camera].dist, (w, h), 1, (w, h)
            )
        else:
            print(f"Camera {numero_camera} not calibrated - Not enough points found")
    
    save_pickle(camerasInfo, PATH_CALIBRATION_MATRIX)
    return camerasInfo


def compute_calibration_single(cameraInfo):
    """
    Computes intrinsic calibration for a single camera using a calibration video.

    Parameters:
        cameraInfo (CameraInfo): camera information object for calibration.

    Returns:
        CameraInfo: updated camera information object with calibration parameters.
    """
    videosCalibration = find_files(PATH_VIDEOS_CALIBRATION)
    videosCalibration.sort()

    for video in videosCalibration:
        numero_camera = int(re.findall(r'\d+', video.replace(".mp4", ""))[0])

        if numero_camera != cameraInfo.camera_number:
            continue

        path_video = os.path.join(PATH_VIDEOS_CALIBRATION, video)
        cameraInfo.objpoints, cameraInfo.imgpoints, gray = findPoints(path_video, cameraInfo, debug=True)

        if len(cameraInfo.objpoints) == 0:
            print(f"Camera {numero_camera} not calibrated - No points found")
            continue

        ret, cameraInfo.mtx, cameraInfo.dist, cameraInfo.rvecs, cameraInfo.tvecs = cv2.calibrateCamera(
            cameraInfo.objpoints, cameraInfo.imgpoints, gray.shape[::-1], None, None
        )

        if ret:
            h, w = gray.shape[:2]
            cameraInfo.newcameramtx, cameraInfo.roi = cv2.getOptimalNewCameraMatrix(
                cameraInfo.mtx, cameraInfo.dist, (w, h), 1, (w, h)
            )
        else:
            print(f"Camera {numero_camera} not calibrated - Not enough points found")

    return cameraInfo


def calibrateAllIntrinsic():
    """
    Initiates the intrinsic calibration for all cameras using preset chessboard sizes.
    """
    camerasInfo = []
    for camera_number in all_chessboard_sizes.keys():
        camera = CameraInfo(camera_number)
        camera.chessboard_size = all_chessboard_sizes[camera_number]
        camerasInfo.append(camera)
    camerasInfo = compute_calibration_all(camerasInfo)

def calibrateCameraIntrinsic(camera_number):
    """
    Calibrates a specific camera using intrinsic parameters and saves the calibration.

    Parameters:
        camera_number (int): camera number to calibrate.
    """
    camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)
    cameraInfo = None

    for chessNumber in all_chessboard_sizes.keys():
        if chessNumber == camera_number:
            cameraInfo = CameraInfo(camera_number)
            cameraInfo.chessboard_size = all_chessboard_sizes[camera_number]

    cameraInfo = compute_calibration_single(cameraInfo)
    flagFind = False

    for i, camera in enumerate(camerasInfo):
        if camera.camera_number == camera_number:
            camerasInfo[i] = cameraInfo
            flagFind = True
            break

    if not flagFind:
        camerasInfo.append(cameraInfo)

    save_pickle(camerasInfo, PATH_CALIBRATION_MATRIX)


def check_errors():
    """
    Checks calibration errors for each camera and prints the mean reprojection error.
    """
    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    for elem in camera_infos:
        mean_error = 0
        for i in range(len(elem.objpoints)):
            imgpoints2, _ = cv2.projectPoints(elem.objpoints[i], elem.rvecs[i], elem.tvecs[i], elem.mtx, elem.dist)
            error = cv2.norm(elem.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        print("\ncamera number: {}".format(elem.camera_number))
        print("total error: {}".format(mean_error / len(elem.objpoints)))


def test_calibration():
    """
    Tests calibration by displaying undistorted frames side by side with original frames for visual validation.
    """
    videos = find_files(PATH_VIDEOS)
    videos.sort()
    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    for video in videos:
        camera_number = int(re.findall(r'\d+', video.replace(".mp4", ""))[0])
        if camera_number not in VALID_CAMERA_NUMBERS:
            continue

        camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)
        path_video = os.path.join(PATH_VIDEOS, video)
        video_capture = cv2.VideoCapture(path_video)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            undistorted_frame = undistorted(frame, camera_info)
            undistorted_frame = cv2.resize(undistorted_frame, (frame.shape[1], frame.shape[0]))
            comparison_frame = np.hstack((frame, undistorted_frame))
            comparison_frame = cv2.resize(comparison_frame, (int(comparison_frame.shape[1]/5), int(comparison_frame.shape[0]/5)))

            cv2.imshow('Original (Left) vs Undistorted (Right)', comparison_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    camera_number = input("Enter the camera number to calibrate (press Enter to calibrate all cameras): ")
    if camera_number == "":
        print("Calibrating all cameras...")
        calibrateAllIntrinsic()
    else:
        print(f"Calibrating camera {camera_number}...")
        calibrateCameraIntrinsic(int(camera_number))

    if input("Do you want to check calibration errors? (y/n): ") == "y":
        check_errors()

    if input("Do you want to test calibration? (y/n): ") == "y":
        test_calibration()