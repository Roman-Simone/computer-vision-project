import os
import cv2
import numpy as np
from utils import *
from homographyMatrix import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)

calibration_file = os.path.join(parent_path, "data/calibrationMatrix/calibration.pkl")
csv_file = os.path.join(parent_path, "data/fundamentalMatrix/points4Fundamentals.csv")
path_video_1 = os.path.join(parent_path, "data/dataset/video/out1.mp4")
path_video_2 = os.path.join(parent_path, "data/dataset/video/out2.mp4")
fundamental_file = os.path.join(parent_path, "data/fundamentalMatrix/provaF.pkl")
homographic_file = os.path.join(parent_path, "data/fundamentalMatrix/provaH.pkl")

def click_event(event, x, y, flags, param):
    global img1, img2, F
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if param == "left":
            print(f"Left Image Clicked at: {x}, {y}")
            pts1 = np.array([[x, y]], dtype='float32')
            pts1 = np.reshape(pts1, (1, 1, 2))
            lines = cv2.computeCorrespondEpilines(pts1, 1, F)
            lines = np.reshape(lines, (-1, 3))
            img2_with_lines = draw_lines(img2, lines)
            cv2.imwrite('epipolar_lines.jpg', img2_with_lines)
            # cv2.imshow('image2_crop', img2_with_lines)
        elif param == "right":
            print(f"Right Image Clicked at: {x}, {y}")
            pts2 = np.array([[x, y]], dtype='float32')
            pts2 = np.reshape(pts2, (1, 1, 2))
            lines = cv2.computeCorrespondEpilines(pts2, 2, F)
            lines = np.reshape(lines, (-1, 3))
            img1_with_lines = draw_lines(img1, lines)
            cv2.imwrite('epipolar_lines_2.jpg', img1_with_lines)
            # cv2.imshow('image1_crop', img1_with_lines)

def draw_lines(img, lines):
    img_with_lines = img.copy()
    r, c = img.shape
    img_with_lines = cv2.cvtColor(img_with_lines, cv2.COLOR_GRAY2BGR)
    for r in lines:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img_with_lines = cv2.line(img_with_lines, (x0, y0), (x1, y1), color, 10)
    return img_with_lines

def drawEpipolarlines():
    global F, img1, img2
    
    F = load_pickle(fundamental_file)
    H = load_pickle(homographic_file)

    video_capture_1 = cv2.VideoCapture(path_video_1)
    video_capture_2 = cv2.VideoCapture(path_video_2)
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    camera_number_1 = 1
    camera_number_2 = 2

    camera_info_1 = find_cameraInfo(camera_number_1, calibration_file)
    camera_info_2 = find_cameraInfo(camera_number_2, calibration_file)

    frame1, frame2 = undistortedAndCrop(frame1, frame2, camera_info_1, camera_info_2)

    img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)   

    H = load_pickle(homographic_file)
    _, img2 = apply_homography(img1, img2, H)

    cv2.namedWindow('image1')
    cv2.namedWindow('image2')
    cv2.setMouseCallback('image1', click_event, param="left")
    cv2.setMouseCallback('image2', click_event, param="right")

    while True:
        cv2.imshow('image1', img1)
        cv2.imshow('image2', img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    drawEpipolarlines()
