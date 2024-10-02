import cv2
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
    
    save_pickle(InterCameraInfolist, PATH_HOMOGRAPHY_MATRIX)



def testHomography():
    interInfo = load_pickle(PATH_HOMOGRAPHY_MATRIX)
    cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = np.array([[x + camera_info_1.roi[0], y + camera_info_1.roi[1]]], dtype=np.float32)

            # Apply homography transformation
            point_transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography).reshape(-1, 2)
            
            # Draw the point on both images
            cv2.circle(img_src, (int(x), int(y)), 15, (0, 255, 0), -1)
            cv2.circle(img_dst, (int(point_transformed[0][0] - camera_info_2.roi[0]), int(point_transformed[0][1] - camera_info_2.roi[1])), 15, (0, 255, 0), -1)

            # Update the displays
            cv2.imshow(f"Camera {camera_src}", img_src)
            cv2.imshow(f"Camera {camera_dst}", img_dst)

    for inter in interInfo:
        camera_src = inter.camera_number_1
        camera_dst = inter.camera_number_2
        homography = inter.homography
        
        if homography is None:
            print(f"No homography available for cameras {camera_src} and {camera_dst}")
            continue
        
        # Load images for both cameras
        img_src = cv2.imread(f"{PATH_FRAME}/cam_{camera_src}.png")
        img_dst = cv2.imread(f"{PATH_FRAME}/cam_{camera_dst}.png")
        
        camera_info_1, _ = take_info_camera(camera_src, cameras_info)
        camera_info_2, _ = take_info_camera(camera_dst, cameras_info)
        img_src = undistorted(img_src, camera_info_1)
        img_dst = undistorted(img_dst, camera_info_2)
        
        if img_src is None or img_dst is None:
            print(f"Could not load images for cameras {camera_src} and {camera_dst}")
            continue
        
        # Create windows and set mouse callback
        cv2.namedWindow(f"Camera {camera_src}")
        cv2.setMouseCallback(f"Camera {camera_src}", mouse_callback)
        
        cv2.imshow(f"Camera {camera_src}", img_src)
        cv2.imshow(f"Camera {camera_dst}", img_dst)
        
        print(f"Click on the image from Camera {camera_src} to see the corresponding point on Camera {camera_dst}")
        print("Press 'q' to move to the next pair of cameras or exit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Calculate Homography
    calculateHomographyAllCameras()

    # Test Homography
    testHomography()
