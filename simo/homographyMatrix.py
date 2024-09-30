import numpy as np
from utils import *
from config import *
from cameraInfo import InterCameraInfo


def calculate_homography_matrix(camera_info1, camera_info2):

    # data from camera 1
    intrinsic1 = camera_info1.newcameramtx
    rotation1 = camera_info1.extrinsic_matrix[:3, :3]
    translation1 = camera_info1.extrinsic_matrix[:3, 3]

    h1 = intrinsic1 @ np.hstack((rotation1, translation1.reshape(3,1)))

    # data from camera 2
    intrinsic2 = camera_info2.newcameramtx
    rotation2 = camera_info2.extrinsic_matrix[:3, :3]
    translation2 = camera_info2.extrinsic_matrix[:3, 3]

    h2 = intrinsic2 @ np.hstack((rotation2, translation2.reshape(3,1)))

    homography = h2 @ np.linalg.pinv(h1)


    return homography


def calculateHomographyMatrix():

    InterCameraInfolist = []

    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    for camera_number1 in VALID_CAMERA_NUMBERS:

        camera_info1, _ = take_info_camera(camera_number1, camera_infos)

        for camera_number2 in VALID_CAMERA_NUMBERS:

            if camera_number1 == camera_number2:
                continue  

            camera_info2, _ = take_info_camera(camera_number2, camera_infos)    

            homography_matrix = calculate_homography_matrix(camera_info1, camera_info2)

            inter_camera_info = InterCameraInfo(camera_number1, camera_number2)

            inter_camera_info.homography = homography_matrix

            InterCameraInfolist.append(inter_camera_info)

    save_pickle(InterCameraInfolist, "inter_camera_info.pkl")


def test_all_homography_matrix():
    inter_camera_info_list = load_pickle("inter_camera_info.pkl")

    for inter_camera_info in inter_camera_info_list:

        camera_number_1 = inter_camera_info.camera_number_1
        camera_number_2 = inter_camera_info.camera_number_2
        homography = inter_camera_info.homography
        print("Camera number 1: ", inter_camera_info.camera_number_1)
        print("Camera number 2: ", inter_camera_info.camera_number_2)
        print("Homography matrix: ", inter_camera_info.homography)
        print("\n")

        img1 = cv2.imread(f"{PATH_FRAME}/cam_{camera_number_1}.png")
        img2 = cv2.imread(f"{PATH_FRAME}/cam_{camera_number_2}.png")

        # find common part of image 1 and image 2 using homography matrix

        
        




if __name__ == "__main__":
    calculateHomographyMatrix()

    # test homography matrix
    test_all_homography_matrix()