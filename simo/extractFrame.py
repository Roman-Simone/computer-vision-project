import re
import os
import cv2
from tqdm import tqdm
from utils import *
from config import *

SKIP_FRAME = 850

def extractFrame():

    # Create the dataset folder
    os.makedirs(PATH_DATASET, exist_ok=True)

    # Get list of video files
    videos = find_files(PATH_VIDEOS)
    videos.sort()

    # Load camera calibration information
    camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

    # Calculate total number of frames across all videos
    total_frames = 0
    for video in videos:
        path_video = os.path.join(PATH_VIDEOS, video)
        video_capture = cv2.VideoCapture(path_video)
        # Extract camera number from the video filename
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])

        if numero_camera not in VALID_CAMERA_NUMBERS:
            continue
        total_frames += int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()

    frame_count = 0

    # Initialize tqdm progress bar with the total number of frames
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        for video in videos:
            # Extract camera number from the video filename
            numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
            numero_camera = int(numero_camera[0])

            if numero_camera not in VALID_CAMERA_NUMBERS:
                continue

            # Get camera information
            cameraInfo, _ = take_info_camera(numero_camera, camerasInfo)

            # Open video file
            path_video = os.path.join(PATH_VIDEOS, video)
            video_capture = cv2.VideoCapture(path_video) 
            

            # Process video frame by frame
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                if frame_count % SKIP_FRAME == 0:
                    # Undistort frame
                    frame = undistorted(frame, cameraInfo)
                    # Save frame as an image file
                    cv2.imwrite(f"{PATH_DATASET}/{int(frame_count/SKIP_FRAME)}_{numero_camera}.png", frame)

                # Update frame count and progress bar
                frame_count += 1
                pbar.update(1)  # Update the progress bar by 1 for each processed frame

            video_capture.release()


if __name__ == "__main__":
    extractFrame()
