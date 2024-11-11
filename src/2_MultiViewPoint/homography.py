import os
import cv2
import sys
import numpy as np
from cameraInfo import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

coordinates_by_camera = read_json_file_and_structure_data(PATH_JSON_DISTORTED)
camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

def find_common_points(camera_number_1: int, camera_number_2: int):
    """
    Finds common world points between two cameras and retrieves their corresponding image points.

    Parameters:
        camera_number_1 (int): first camera number.
        camera_number_2 (int): second camera number.

    Returns:
        tuple: two lists of image points for the common world points in the two cameras.
    """
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
    """
    Calculates the homography matrix between two cameras using their undistorted common points.

    Parameters:
        points_1 (numpy.ndarray): image points from the first camera.
        points_2 (numpy.ndarray): image points from the second camera.
        camera_number_1 (int): first camera number.
        camera_number_2 (int): second camera number.

    Returns:
        numpy.ndarray: homography matrix between the two cameras.
    """
    camera_info_1, _ = take_info_camera(camera_number_1, camera_infos)
    camera_info_2, _ = take_info_camera(camera_number_2, camera_infos)

    if points_1.shape[0] < 4 or points_2.shape[0] < 4:
        print(f"Too few points to compute the homography map between cam {camera_number_1} - {camera_number_2}")
        hom = None
    else:
        points_1_undistorted = cv2.undistortPoints(points_1, camera_info_1.mtx, camera_info_1.dist, P=camera_info_1.newcameramtx) if camera_number_1 != 0 else points_1
        points_2_undistorted = cv2.undistortPoints(points_2, camera_info_2.mtx, camera_info_2.dist, P=camera_info_2.newcameramtx) if camera_number_2 != 0 else points_2

        hom, _ = cv2.findHomography(points_1_undistorted, points_2_undistorted, method=0)
    
    return hom


def calculateHomographyAllCameras():
    """
    Computes homography matrices for all pairs of cameras and saves them in a pkl file.
    """
    HomographyInfolist = []
    cameras = VALID_CAMERA_NUMBERS.copy()
    cameras.append(0)  # Add the court as a virtual camera for homography calculation
    cameras.sort()

    for camera_number_1 in cameras:
        for camera_number_2 in cameras:
            if camera_number_1 == camera_number_2:
                continue

            points_1, points_2 = find_common_points(camera_number_1, camera_number_2)
            homographyMatrix = homographyUndistortedCameras(points_1, points_2, camera_number_1, camera_number_2)

            homographyInfo = HomographyInfo(camera_number_1, camera_number_2)
            homographyInfo.homography = homographyMatrix
            HomographyInfolist.append(homographyInfo)
    
    save_pickle(HomographyInfolist, PATH_HOMOGRAPHY_MATRIX)


def testHomography():
    """
    Loads homography matrices and allows user to interactively test homographies by selecting points on images.
    """
    homographyInfos = load_pickle(PATH_HOMOGRAPHY_MATRIX)
    cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)
    
    def mouse_callback(event, x, y, flags, param):
        """
        Mouse callback to show corresponding points in two camera views when clicked.
        
        Parameters:
            event (int): type of mouse event (e.g., left button down).
            x (int): x-coordinate of the mouse click.
            y (int): y-coordinate of the mouse click.
            flags (int): additional flags for the mouse event.
            param (tuple): additional parameters for the callback (unused).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            x_original = x / scale_factor_src
            y_original = y / scale_factor_src

            if camera_src != 0:
                point = np.array([[x_original + camera_info_1.roi[0], y_original + camera_info_1.roi[1]]], dtype=np.float32)
            else:
                point = np.array([[x_original, y_original]], dtype=np.float32)

            point_transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography).reshape(-1, 2)

            x_transformed = point_transformed[0][0] - (camera_info_2.roi[0] if camera_dst != 0 else 0)
            y_transformed = point_transformed[0][1] - (camera_info_2.roi[1] if camera_dst != 0 else 0)

            x_transformed_resized = int(x_transformed * scale_factor_dst)
            y_transformed_resized = int(y_transformed * scale_factor_dst)

            cv2.circle(img_src_resized, (int(x), int(y)), 15, (0, 255, 0), -1)
            cv2.circle(img_dst_resized, (x_transformed_resized, y_transformed_resized), 15, (0, 255, 0), -1)

            concatenated_image = cv2.hconcat([img_src_resized, img_dst_resized])
            cv2.imshow(f"Camera {camera_src} and {camera_dst}", concatenated_image)

    for homographyInfo in homographyInfos:
        camera_src = homographyInfo.camera_number_1
        camera_dst = homographyInfo.camera_number_2
        homography = homographyInfo.homography
        
        if homography is None:
            print(f"No homography available for cameras {camera_src} and {camera_dst}")
            continue
        
        img_src = cv2.imread(f"{PATH_FRAME_DISTORTED}/cam_{camera_src}.png")
        img_dst = cv2.imread(f"{PATH_FRAME_DISTORTED}/cam_{camera_dst}.png")
        
        if camera_src != 0:
            camera_info_1, _ = take_info_camera(camera_src, cameras_info)
            img_src = undistorted(img_src, camera_info_1)
        if camera_dst != 0:
            camera_info_2, _ = take_info_camera(camera_dst, cameras_info)
            img_dst = undistorted(img_dst, camera_info_2)
        
        if img_src is None or img_dst is None:
            print(f"Could not load images for cameras {camera_src} and {camera_dst}")
            continue

        height_src, width_src = img_src.shape[:2]
        height_dst, width_dst = img_dst.shape[:2]
        desired_height = max(height_src, height_dst)
        scale_factor_src = desired_height / height_src
        scale_factor_dst = desired_height / height_dst

        img_src_resized = cv2.resize(img_src, (int(width_src * scale_factor_src), desired_height))
        img_dst_resized = cv2.resize(img_dst, (int(width_dst * scale_factor_dst), desired_height))

        concatenated_image = cv2.hconcat([img_src_resized, img_dst_resized])

        cv2.namedWindow(f"Camera {camera_src} and {camera_dst}")
        cv2.setMouseCallback(f"Camera {camera_src} and {camera_dst}", mouse_callback)

        cv2.imshow(f"Camera {camera_src} and {camera_dst}", concatenated_image)
        
        print(f"Click on the image from Camera {camera_src} to see the corresponding point on Camera {camera_dst}")
        print("Press 'q' to move to the next pair of cameras or exit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    
    # need to round the field map
    
    calculateHomographyAllCameras()
    
    testHomography()
