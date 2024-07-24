import os
import cv2
import numpy as np
from utils import load_calibration

def undistort_video(path_video, camera_info):
    video_capture = cv2.VideoCapture(path_video)
    if not video_capture.isOpened():
        print("Error opening video file.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        undistorted_frame = cv2.undistort(frame, camera_info.mtx, camera_info.dist, None, camera_info.newcameramtx)
        # print(camera_info.roi)
        x, y, w, h = camera_info.roi
        undistorted_frame = undistorted_frame[y:y+h, x:x+w]
        
        undistorted_frame_resized = cv2.resize(undistorted_frame, (frame.shape[1], frame.shape[0]))

        comparison_frame = np.hstack((frame, undistorted_frame_resized))

        cv2.imshow('Original (Left) vs Undistorted (Right)', comparison_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    calibration_file = "/Users/simoneroman/Desktop/CV/Computer_Vision_project/calibration.pkl"
    camera_infos = load_calibration(calibration_file)

    example_video_path = '/Users/simoneroman/Desktop/CV/Computer_Vision_project/dataset/calibration/out13F.mp4'


    camera_number = 13
    camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)
    print(camera_info.mtx)
    if camera_info:
        undistort_video(example_video_path, camera_info)
    else:
        print(f"No calibration data found for camera number {camera_number}")
