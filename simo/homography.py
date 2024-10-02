import numpy as np
from utils import *
from config import *
from cameraInfo import *


coordinates_by_camera = read_json_file_and_structure_data(PATH_JSON)
camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)


def find_common_points(camera_number_1: int, camera_number_2: int):

    points_1 = []
    points_2 = []

    for camera_id, coords in coordinates_by_camera.items():
        if int(camera_id) == camera_number_1:
            world_points_1 = np.array(coords["world_coordinates"], dtype=np.float32)
            image_points_1 = np.array(coords["image_coordinates"], dtype=np.float32)
        elif int(camera_id) == camera_number_2:
            world_points_2 = np.array(coords["world_coordinates"], dtype=np.float32)
            image_points_2 = np.array(coords["image_coordinates"], dtype=np.float32)

    for pos1, elem1 in enumerate(world_points_1):
        for pos2, elem2 in enumerate(world_points_2):
            if elem1[0] == elem2[0] and elem1[1] == elem2[1]:
                points_1.append(image_points_1[pos1])
                points_2.append(image_points_2[pos2])
    
    points_1 = np.array(points_1, np.float32)
    points_2 = np.array(points_2, np.float32)
    
    return points_1, points_2


def homographyUndistortedCameras(points_1, points_2, camera_number_1, camera_number_2):

    homography_ret = None

    camera_info_1, _ = take_info_camera(camera_number_1, camera_infos)
    camera_info_2, _ = take_info_camera(camera_number_2, camera_infos)

    if points_1.shape[0] < 4 or points_2.shape[0] < 4:
        print("Too few points to compute the homography map between cam")
        hom = None
    else:
        points_1_undistorted = cv2.undistortPoints(points_1, camera_info_1.mtx, camera_info_1.dist, P=camera_info_1.newcameramtx)
        points_2_undistorted = cv2.undistortPoints(points_2, camera_info_2.mtx, camera_info_2.dist, P=camera_info_2.newcameramtx)
        hom, _ = cv2.findHomography(points_1_undistorted, points_2_undistorted, method=0)
    
    return hom


def calculateHomographyAllCameras():

    InterCameraInfolist = []

    for camera_number_1 in VALID_CAMERA_NUMBERS:
        
        for camera_number_2 in VALID_CAMERA_NUMBERS:

            if camera_number_1 == camera_number_2:
                continue


            points_1, points_2 = find_common_points(camera_number_1, camera_number_2)

            homographyMatrix = homographyUndistortedCameras(points_1, points_2, camera_number_1, camera_number_2)

            inter_camera_info = InterCameraInfo(camera_number_1, camera_number_2)

            inter_camera_info.homography = homographyMatrix

            InterCameraInfolist.append(inter_camera_info)
    
    save_pickle(InterCameraInfo, "inter.pkl")


if __name__ == '__main__':
    # Calculate Homography
    calculateHomographyAllCameras()
