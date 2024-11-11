import re
import os
import cv2
import sys
from tqdm import tqdm
from cameraInfo import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

SKIP_FRAME = 342

def extractFrame():
    """
    Extracts frames from videos, processes them, and saves them to a dataset.
    This function performs the following steps:
    1. Creates the dataset directory if it does not exist.
    2. Finds and sorts all video files in the specified directory.
    3. Loads camera calibration information.
    4. Calculates the total number of frames to process.
    5. Iterates through each video, extracts frames, undistorts them using camera calibration data, 
       and saves the processed frames to the dataset directory.
    The function skips frames based on the SKIP_FRAME parameter and only processes videos from valid cameras.
    
    Raises:
        FileNotFoundError: If the specified video or calibration files are not found.
        ValueError: If the camera number extracted from the video filename is not valid.
  
    Note:
        This function uses OpenCV for video processing and tqdm for progress display.
    """

    os.makedirs(PATH_DATASET, exist_ok=True)

    videos = find_files(PATH_VIDEOS)
    videos.sort()

    camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

    total_frames = 0
    for video in videos:
        path_video = os.path.join(PATH_VIDEOS, video)
        video_capture = cv2.VideoCapture(path_video)
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])

        if numero_camera not in VALID_CAMERA_NUMBERS:
            continue
        total_frames += int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()

    frame_count = 0

    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        for video in videos:
            numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
            numero_camera = int(numero_camera[0])

            if numero_camera not in VALID_CAMERA_NUMBERS:
                continue

            cameraInfo, _ = take_info_camera(numero_camera, camerasInfo)

            path_video = os.path.join(PATH_VIDEOS, video)
            video_capture = cv2.VideoCapture(path_video) 
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                if frame_count % SKIP_FRAME == 0:
                    frame = undistorted(frame, cameraInfo)
                    cv2.imwrite(f"{PATH_DATASET}/{int(frame_count/SKIP_FRAME)}_{numero_camera}.png", frame)

                frame_count += 1
                pbar.update(1)  

            video_capture.release()


if __name__ == "__main__":
    extractFrame()
