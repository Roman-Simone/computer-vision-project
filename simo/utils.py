import os
import pickle
import pandas as pd
import numpy as np
import cv2


def crop_polygon(img, vertices):
    # Creare una maschera nera con le stesse dimensioni dell'immagine
    mask = np.zeros_like(img)
    vertices = np.array([vertices], dtype=np.int32)
    # Riempire il poligono con il bianco
    cv2.fillPoly(mask, [vertices], (255,) * img.shape[2])
    
    # Applicare la maschera all'immagine
    masked_img = cv2.bitwise_and(img, mask)
    
    # Trova i confini del rettangolo che contiene il poligono
    x, y, w, h = cv2.boundingRect(vertices)
    
    # Ritagliare l'immagine utilizzando i confini trovati
    cropped_img = masked_img[y:y+h, x:x+w]
    
    return cropped_img


def findVertices(path_csv_file, camera_info_1, camera_info_2):
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
                corners_cam1.append(convert_to_tuple(row['bottom_left_corner_cam1']))
                corners_cam1.append(convert_to_tuple(row['bottom_right_corner_cam1']))
                corners_cam2.append(convert_to_tuple(row['top_left_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['top_right_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['bottom_right_corner_cam2']))
                corners_cam2.append(convert_to_tuple(row['bottom_left_corner_cam2']))
                
                first_time = False
    
    return corners_cam1, corners_cam2


def undistortedAndCrop(frame1, frame2, camera_info_1, camera_info_2, corners_cam1="ALL", corners_cam2="ALL"):

    undistorted_frame_1 = cv2.undistort(frame1, camera_info_1.mtx, camera_info_1.dist, None, camera_info_1.newcameramtx)
    undistorted_frame_2 = cv2.undistort(frame2, camera_info_2.mtx, camera_info_2.dist, None, camera_info_2.newcameramtx)

    x1, y1, w1, h1 = camera_info_1.roi
    undistorted_frame_1 = undistorted_frame_1[y1:y1+h1, x1:x1+w1]

    x2, y2, w2, h2 = camera_info_2.roi
    undistorted_frame_2 = undistorted_frame_2[y2:y2+h2, x2:x2+w2]
    

    if 'ALL' in corners_cam1:
        cropped_frame_1 = undistorted_frame_1
    else:
        corners_cam1 = np.array(corners_cam1, np.int32)
        # Crop the result with the bounding rectangle
        cropped_frame_1 = crop_polygon(undistorted_frame_1, corners_cam1)

    if 'ALL' in corners_cam2:
        cropped_frame_2 = undistorted_frame_2
    else:

        corners_cam2 = np.array(corners_cam2, np.int32)
        # Crop the result with the bounding rectangle
        cropped_frame_2 = crop_polygon(undistorted_frame_2, corners_cam2)

    return cropped_frame_1, cropped_frame_2


# Funzione per convertire le stringhe 'x_y' in tuple (x, y)
def convert_to_tuple(coord_str):
    if coord_str == 'ALL':
        return 'ALL'
    x, y = map(float, coord_str.split('_'))
    return (x, y)


def save_pickle(camerasInfo, filename):
    with open(filename, 'wb') as file:
        pickle.dump(camerasInfo, file)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        camerasInfo = pickle.load(file)
    return camerasInfo


def trova_file_mp4(cartella):
    file_mp4 = []
    for file in os.listdir(cartella):
        if file.endswith(".mp4"):
            file_mp4.append(file)
    return file_mp4


def find_cameraInfo(camera_number, path_calibration_file):
    camera_infos = load_pickle(path_calibration_file)
    camera_info = next((cam for cam in camera_infos if cam.camera_number == camera_number), None)
    return camera_info