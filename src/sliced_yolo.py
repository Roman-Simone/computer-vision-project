import re
import os
import cv2
from yoloWindows import *



from utils.utils import *
from utils.config import *
from cameraInfo import *

# Update PATH_WEIGHT to point to the actual model file
weight_path = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
window_yolo = yoloWindow(pathWeight=weight_path, windowSize=(640, 640), overlap=(0.1, 0.1))


def applyModel(frame):
    detections, processed_frame = window_yolo.detect(
        frame, visualizeBBox=True, visualizeWindows=True
    )

    
    return processed_frame

def testModel():
    videos = find_files(PATH_VIDEOS)
    videos.sort()
    cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

    for video in videos:
        numero_camera = re.findall(r'\d+', video.replace(".mp4", ""))
        numero_camera = int(numero_camera[0])

        if numero_camera not in VALID_CAMERA_NUMBERS:
            continue
        print(f"Processing video {numero_camera}")

        cameraInfo, _ = take_info_camera(numero_camera, cameraInfos)

        pathVideo = os.path.join(PATH_VIDEOS, video)

        videoCapture = cv2.VideoCapture(pathVideo)

        while True:
            ret, frame = videoCapture.read()

            if not ret:
                break

            frameUndistorted = undistorted(frame, cameraInfo)

            frameWithBbox = applyModel(frameUndistorted)

            cv2.imshow('Frame', frameWithBbox)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                break

        videoCapture.release()
        cv2.destroyAllWindows()





if __name__ == '__main__':
    testModel()
