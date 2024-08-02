import cv2
import numpy as np
from utils import *

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
            cv2.imshow('image2', img2_with_lines)
        elif param == "right":
            print(f"Right Image Clicked at: {x}, {y}")
            pts2 = np.array([[x, y]], dtype='float32')
            pts2 = np.reshape(pts2, (1, 1, 2))
            lines = cv2.computeCorrespondEpilines(pts2, 2, F)
            lines = np.reshape(lines, (-1, 3))
            img1_with_lines = draw_lines(img1, lines)
            cv2.imshow('image1', img1_with_lines)

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

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.join(current_path, os.pardir)
    parent_path = os.path.abspath(parent_path)

    calibration_file = os.path.join(parent_path, "data/calibrationMatrix/calibration.pkl")
    csv_file = os.path.join(parent_path, "data/fundamentalMatrix/points4Fundamentals.csv")
    path_video_1 = os.path.join(parent_path, "data/dataset/video/out1.mp4")
    path_video_2 = os.path.join(parent_path, "data/dataset/video/out2.mp4")
    fundamental_file = os.path.join(parent_path, "data/fundamentalMatrix/fundamentalMatrix_1_2_auto.pkl")
    
    # mask_file = "/Users/simoneroman/Desktop/CV/Computer_Vision_project/mask.pkl"

    F = load_pickle(fundamental_file)
    # mask = load_calibration(mask_file)

    video_capture_1 = cv2.VideoCapture(path_video_1)
    video_capture_2 = cv2.VideoCapture(path_video_2)
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    camera_infos = load_pickle(calibration_file)
    camera_number_1 = 1
    camera_number_2 = 2

    camera_info_1 = next((cam for cam in camera_infos if cam.camera_number == camera_number_1), None)
    camera_info_2 = next((cam for cam in camera_infos if cam.camera_number == camera_number_2), None)
    
    undistorted_frame_1 = cv2.undistort(frame1, camera_info_1.mtx, camera_info_1.dist, None, camera_info_1.newcameramtx)
    undistorted_frame_2 = cv2.undistort(frame2, camera_info_2.mtx, camera_info_2.dist, None, camera_info_2.newcameramtx)

    x1, y1, w1, h1 = camera_info_1.roi
    img1 = undistorted_frame_1[y1:y1+h1, x1:x1+w1]

    x2, y2, w2, h2 = camera_info_2.roi
    img2 = undistorted_frame_2[y2:y2+h2, x2:x2+w2]
        

    if ret1 and ret2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        print("Error reading video files.")
        return

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
