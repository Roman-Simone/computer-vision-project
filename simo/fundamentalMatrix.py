import os
import cv2
import numpy as np
from utils import *
from cameraInfo import *

def find_points(path_video_1, path_video_2, camera_info_1, camera_info_2, path_csv_file):

    df = pd.read_csv(path_csv_file, delimiter=';')
    cam_to_check = f"{camera_info_1.camera_number}_{camera_info_2.camera_number}"

    # Inizializza le liste per memorizzare i valori
    corners_cam1 = []  # order: top_left, top_right, bottom_left, bottom_right
    corners_cam2 = []  # order: top_left, top_right, bottom_left, bottom_right
    first_time = True

    # Leggi il DataFrame riga per riga
    for index, row in df.iterrows():
        if row['cams_number'] == cam_to_check:
            if first_time:
                corners_cam1.append(convert_to_tuple(row['top_left_corner_cam1']))
                corners_cam1.append(convert_to_tuple(row['top_right_corner_cam1']))
                corners_cam1.append(convert_to_tuple(row['bottom_right_corner_cam1']))
                corners_cam1.append(convert_to_tuple(row['bottom_left_corner_cam1']))
                
                corners_cam2.append(convert_to_tuple(row['top_left_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['top_right_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['bottom_right_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['bottom_left_corner_cam2']))
                
                first_time = False

    video_capture_1 = cv2.VideoCapture(path_video_1)
    video_capture_2 = cv2.VideoCapture(path_video_2)
    if not video_capture_1.isOpened() or not video_capture_2.isOpened():
        print("Error opening video file.")
        return
    
    # Print the number of frames in the video
    numberOf_frame = int(video_capture_1.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames in the video: ", numberOf_frame)
    numberOf_frame = int(video_capture_2.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames in the video: ", numberOf_frame)

    pts1 = []
    pts2 = []
    count_frame = 0

    while True:
        ret1, frame1 = video_capture_1.read()
        ret2, frame2 = video_capture_2.read()
        if not ret1 or not ret2:
            break
        
        count_frame += 1
        if count_frame % 7000 != 0:
            continue
            
        print(f"Processing frame {count_frame}")

        undistorted_frame_1 = cv2.undistort(frame1, camera_info_1.mtx, camera_info_1.dist, None, camera_info_1.newcameramtx)
        undistorted_frame_2 = cv2.undistort(frame2, camera_info_2.mtx, camera_info_2.dist, None, camera_info_2.newcameramtx)

        x1, y1, w1, h1 = camera_info_1.roi
        undistorted_frame_1 = undistorted_frame_1[y1:y1+h1, x1:x1+w1]

        x2, y2, w2, h2 = camera_info_2.roi
        undistorted_frame_2 = undistorted_frame_2[y2:y2+h2, x2:x2+w2]
        

        if 'ALL' in corners_cam1:
            cropped_frame_1 = undistorted_frame_1
        else:
            # Ritaglia i frame utilizzando i vertici
            cropped_frame_1 = crop_polygon(undistorted_frame_1, corners_cam1)

        if 'ALL' in corners_cam2:
            cropped_frame_2 = undistorted_frame_2
        else:
            print(corners_cam2)
            corners_cam2 = [(int(x), int(y)) for x, y in corners_cam2]
            cropped_frame_2 = crop_polygon(undistorted_frame_2, corners_cam2)

        img1 = cv2.cvtColor(cropped_frame_1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cropped_frame_2, cv2.COLOR_BGR2GRAY)



        sift = cv2.SIFT_create()
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

    video_capture_1.release()
    video_capture_2.release()
    cv2.destroyAllWindows()

    return pts1, pts2


def take_points_from_csv(csv_file_path, camera_number_1, camera_number_2):

    df = pd.read_csv(csv_file_path, delimiter=';')
    cam_to_check = f"{camera_number_1}_{camera_number_2}"

    # Inizializza le liste per memorizzare i valori
    corners_cam1 = []  #order: top_left, top_right, bottom_left, bottom_right
    corners_cam2 = []  #order: top_left, top_right, bottom_left, bottom_right
    points_img1 = []
    points_img2 = []
    first_time = False

    # Leggi il DataFrame riga per riga
    for index, row in df.iterrows():

        if row['cams_number'] == cam_to_check:
            if first_time:
                corners_cam1.append(convert_to_tuple(row['top_left_corner_cam1']))
                corners_cam1.append(convert_to_tuple(row['top_right_corner_cam1']))
                corners_cam1.append(convert_to_tuple(row['bottom_left_corner_cam1']))
                corners_cam1.append(convert_to_tuple(row['bottom_right_corner_cam1']))
                corners_cam2.append(convert_to_tuple(row['top_left_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['top_right_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['bottom_left_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['bottom_right_corner_cam2']))
                first_time = False
            points_img1.append(convert_to_tuple(row['point_img1']))
            points_img2.append(convert_to_tuple(row['point_img2']))
    
    return points_img1, points_img2


def calculate_F_matrix():

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.join(current_path, os.pardir)
    parent_path = os.path.abspath(parent_path)

    calibration_file = os.path.join(parent_path, "data/calibrationMatrix/calibration.pkl")
    csv_file = os.path.join(parent_path, "data/fundamentalMatrix/points4Fundamentals.csv")
    video1 = os.path.join(parent_path, "data/dataset/video/out1.mp4")
    video2 = os.path.join(parent_path, "data/dataset/video/out2.mp4")
    output_path = os.path.join(parent_path, "data/fundamentalMatrix/fundamentalMatrix_1_2_auto.pkl")

    interCameraInfo = []

    camera_infos = load_pickle(calibration_file)
    camera_number_1 = 1
    camera_number_2 = 2

    camera_info_1 = next((cam for cam in camera_infos if cam.camera_number == camera_number_1), None)
    camera_info_2 = next((cam for cam in camera_infos if cam.camera_number == camera_number_2), None)
    
    if camera_info_1 and camera_info_2:
        pts1, pts2 = find_points(video1, video2 , camera_info_1, camera_info_2, csv_file)
        print(pts1, pts2)
        # pts1, pts2 = take_points_from_csv(csv_file, camera_number_1, camera_number_2)
        # print(pts1, pts2)
    else:
        print(f"No calibration data found for camera number {camera_number_1} or {camera_number_2}")

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    print(F)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    save_pickle(F, output_path)


if __name__ == '__main__':
    calculate_F_matrix()

