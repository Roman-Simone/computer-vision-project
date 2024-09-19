import os
import cv2
import pickle


def undistorted(frame1, camera_info):   

    undistorted_frame = cv2.undistort(frame1, camera_info.mtx, camera_info.dist, None, camera_info.newcameramtx)
    x1, y1, w1, h1 = camera_info.roi
    undistorted_frame = undistorted_frame[y1:y1+h1, x1:x1+w1]

    return undistorted_frame


def save_pickle(camerasInfo, filename):
    with open(filename, 'wb') as file:
        pickle.dump(camerasInfo, file)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        camerasInfo = pickle.load(file)
    return camerasInfo


def find_file_mp4(directory):
    file_mp4 = []
    for file in os.listdir(directory):
        if file.endswith(".mp4") or file.endswith(".png"):
            file_mp4.append(file)
    return file_mp4


