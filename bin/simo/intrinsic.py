import re
import cv2
import numpy as np
from utils import *
from config import *
from tqdm import tqdm
from cameraInfo import *

all_chessboard_sizes = {1: (5, 7), 2: (5, 7), 3: (5, 7), 4: (5, 7), 5: (6, 9), 6: (6, 9), 7: (5, 7), 8: (6, 9), 12: (5, 7), 13: (5, 7)}
SKIP_FRAME = 10


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
    print(f"\nStarting calibrations for camera {cameraInfo.camera_number}")
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

    # Initialize the progress bar
    with tqdm(total=numberOf_frame, desc="Processing Video", unit="frame") as pbar:

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
                retObjpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                retImgpoints.append(corners2)

                if debug:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (chess_height, chess_width), corners2, ret)
                    # Save the image with detected corners
                    cv2.imwrite(f"{output_dir}/frame{frame_count}.jpg", img)

            # Update the progress bar
            pbar.update(SKIP_FRAME)
    
    return retObjpoints, retImgpoints, gray


def compute_calibration_all(camerasInfo):

    videosCalibration = find_files(PATH_VIDEOS_CALIBRATION)
    videosCalibration.sort()
    
    for video in videosCalibration:
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])

        _, pos_camera = take_info_camera(numero_camera, camerasInfo)
        
        path_video = os.path.join(PATH_VIDEOS_CALIBRATION, video)

        camerasInfo[pos_camera].objpoints, camerasInfo[pos_camera].imgpoints, gray = findPoints(path_video, camerasInfo[pos_camera], debug=False)

        if len(camerasInfo[pos_camera].objpoints) == 0:
            print(f"Camera {numero_camera} not calibrated - No points found")
            continue

        ret, camerasInfo[pos_camera].mtx, camerasInfo[pos_camera].dist, _, _ = cv2.calibrateCamera(camerasInfo[pos_camera].objpoints, camerasInfo[pos_camera].imgpoints, gray.shape[::-1], None, None)

        if ret:
            h,  w = gray.shape[:2]

            camerasInfo[pos_camera].newcameramtx, camerasInfo[pos_camera].roi = cv2.getOptimalNewCameraMatrix(camerasInfo[pos_camera].mtx, camerasInfo[pos_camera].dist, (w,h), 1, (w,h))

        else:
            print(f"Camera {numero_camera} not calibrated - No enough points found")
    
    save_pickle(camerasInfo, PATH_CALIBRATION_MATRIX)
    
    return camerasInfo


def compute_calibration_single(cameraInfo):

    videosCalibration = find_files(PATH_VIDEOS_CALIBRATION)
    videosCalibration.sort()
    
    for video in videosCalibration:
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])
        if numero_camera != cameraInfo.camera_number:
            continue


        path_video = os.path.join(PATH_VIDEOS_CALIBRATION, video)

        cameraInfo.objpoints, cameraInfo.imgpoints, gray = findPoints(path_video, cameraInfo, debug=False)

        if len(cameraInfo.objpoints) == 0:
            print(f"Camera {numero_camera} not calibrated - No points found")
            continue

        ret, cameraInfo.mtx, cameraInfo.dist, cameraInfo.rvecs, cameraInfo.tvecs = cv2.calibrateCamera(cameraInfo.objpoints, cameraInfo.imgpoints, gray.shape[::-1], None, None)

        if ret:
            h,  w = gray.shape[:2]

            cameraInfo.newcameramtx, cameraInfo.roi = cv2.getOptimalNewCameraMatrix(cameraInfo.mtx, cameraInfo.dist, (w,h), 1, (w,h))
        else:
            print(f"Camera {numero_camera} not calibrated - No enough points found")

    return cameraInfo


def calibrateAllIntrinsic():
    camerasInfo = []
    
    for camera_number in all_chessboard_sizes.keys():
        camera = CameraInfo(camera_number)
        camera.chessboard_size = all_chessboard_sizes[camera_number]
        camerasInfo.append(camera)
    
    camerasInfo = compute_calibration_all(camerasInfo)


def calibrateCameraIntrinsic(camera_number):
    camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

    # Inizializza cameraInfo per la camera specifica
    for chessNumber in all_chessboard_sizes.keys():
        if chessNumber == camera_number:
            cameraInfo = CameraInfo(camera_number)
            cameraInfo.chessboard_size = all_chessboard_sizes[camera_number]
    
    # Esegui la calibrazione
    cameraInfo = compute_calibration_single(cameraInfo)
    
    flagFind = False
    
    # Sostituisci l'elemento nella lista usando l'indice
    for i, camera in enumerate(camerasInfo):
        if camera.camera_number == camera_number:
            camerasInfo[i] = cameraInfo  # Sostituisci l'elemento
            flagFind = True
    
    if not flagFind:
        camerasInfo.append(cameraInfo)
    
    # Salva la lista aggiornata
    save_pickle(camerasInfo, PATH_CALIBRATION_MATRIX)


def test_calibration():

    videos = find_files(PATH_VIDEOS)
    videos.sort()
    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    for camera_info in camera_infos:
        print(f"Camera {camera_info.camera_number} rvecs: {camera_info.rvecs}")
    
    for video in videos:

        camera_number = re.findall(r'\d+', video.replace(".mp4", ""))
        camera_number = int(camera_number[0])
        if camera_number not in VALID_CAMERA_NUMBERS:
            continue

        # open the video
        camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)
        path_video = os.path.join(PATH_VIDEOS, video)
        video_capture = cv2.VideoCapture(path_video)

        # Show the video
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

    # CALIBRATE ALL THE CAMS
    calibrateAllIntrinsic()

    # CALIBRATE SPECIFIC CAM
    # camera_number = 3
    calibrateCameraIntrinsic(camera_number)

    # ONLY FOR TESTING
    test_calibration()
