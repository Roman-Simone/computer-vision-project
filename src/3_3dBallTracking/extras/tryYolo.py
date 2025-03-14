import re
import os
import sys
import cv2
from yoloWindows import *

current_path = os.path.dirname(os.path.abspath(__file__))
grandparent_path = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
sys.path.append(grandparent_path)

from utils.utils import *
from utils.config import *

from cameraInfo import *

weight_path = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')

def applyModel(frame, windowFlag = False):
    """
    Apply YOLO model to the given frame.
    Parameters:
    frame (numpy.ndarray): The input image frame to process.
    windowFlag (bool): Flag to determine whether to use windowed YOLO detection or not. 
                    If True, uses windowed detection; otherwise, uses standard YOLO detection.
    Returns:
    numpy.ndarray: The processed frame with bounding boxes and labels drawn.
    """

    if windowFlag:
        window_yolo = yoloWindows(pathWeight=weight_path, windowSize=(640, 640), overlap=(0.1, 0.1))
        detections, processed_frame = window_yolo.detect(
            frame, visualizeBBox=True, visualizeWindows=True
        )

        return processed_frame
    else:
        model = YOLO(weight_path)
        size = 800

        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        originalSizeHeight, originalSizeWidth, _ = frame.shape

        results = model.track(frame, verbose = False, device=device)

        frameResized = cv2.resize(frame, (size, size))
        
        results = model.track(frameResized, verbose=False, device=device)
        
        for box in results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = x1 * originalSizeWidth / size
            y1 = y1 * originalSizeHeight / size
            x2 = x2 * originalSizeWidth / size
            y2 = y2 * originalSizeHeight / size

            confidence = box.conf[0]
            class_id = box.cls[0]

            if confidence < 0.5:
                continue

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{confidence:.2f}'
            label_position = (int(x1), int(y1) - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            cv2.circle(frame, (int(x_center), int(y_center)), 3, (0, 255, 0), -1)
        
        return frame


def testModel():
    """
    Tests the YOLO model on a set of videos.

    This function performs the following steps:
    1. Finds and sorts video files from a specified directory.
    2. Loads camera calibration information.
    3. Prompts the user to select a camera number.
    4. Processes each video corresponding to the selected camera number:
        - Reads the video file.
        - Undistorts each frame using the camera calibration information.
        - Applies the YOLO model to each undistorted frame.
        - Displays the processed frame with bounding boxes in a window.
        - Allows the user to stop the video processing by pressing the 's' key.
        
    Raises:
        ValueError: If the user inputs an invalid camera number.
    """
    
    videos = find_files(PATH_VIDEOS)
    videos.sort()
    cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

    cameraNumber = -1

    while cameraNumber not in VALID_CAMERA_NUMBERS:
        print("Select the number of camera (1 - 8) (12 - 13):")
        cameraNumber = int(input())
    
    window = input("Do you want to use windowed YOLO detection? (y/n): ")
    windowFlag = True if window == 'y' else False

    for video in videos:
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])

        if numero_camera not in VALID_CAMERA_NUMBERS or numero_camera != cameraNumber:
            continue

        print(f"Processing video {numero_camera}, windowFlag: {windowFlag}")

        cameraInfo, _ = take_info_camera(numero_camera, cameraInfos)

        pathVideo = os.path.join(PATH_VIDEOS, video)

        videoCapture = cv2.VideoCapture(pathVideo)

        while True:
            ret, frame = videoCapture.read()

            if not ret:
                break

            frameUndistorted = undistorted(frame, cameraInfo)

            frameWithBbox = applyModel(frameUndistorted, windowFlag = windowFlag)

            cv2.imshow('Frame', frameWithBbox)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break

        videoCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':


    testModel()
