import numpy as np
import cv2
from utils import *
from config import *
from cameraInfo import InterCameraInfo

clicked_point = None

def calculate_homography_matrix(camera_info1, camera_info2):
    h1 = camera_info1.newcameramtx @ np.hstack((camera_info1.extrinsic_matrix[:3, :3], camera_info1.extrinsic_matrix[:3, 3].reshape(3,1)))
    h2 = camera_info2.newcameramtx @ np.hstack((camera_info2.extrinsic_matrix[:3, :3], camera_info2.extrinsic_matrix[:3, 3].reshape(3,1)))
    return h2 @ np.linalg.pinv(h1), h1, h2

def calculateHomographyMatrix():
    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)
    InterCameraInfolist = []

    for camera_number1 in VALID_CAMERA_NUMBERS:
        camera_info1, _ = take_info_camera(camera_number1, camera_infos)
        for camera_number2 in VALID_CAMERA_NUMBERS:
            if camera_number1 == camera_number2:
                continue  
            camera_info2, _ = take_info_camera(camera_number2, camera_infos)    
            homography_matrix, h1, h2 = calculate_homography_matrix(camera_info1, camera_info2)
            inter_camera_info = InterCameraInfo(camera_number1, camera_number2)
            inter_camera_info.homography = homography_matrix
            inter_camera_info.h1 = h1
            inter_camera_info.h2 = h2
            InterCameraInfolist.append(inter_camera_info)
            print(f"Homography matrix shape for cameras {camera_number1} to {camera_number2}: {homography_matrix.shape}")

    save_pickle(InterCameraInfolist, "inter_camera_info.pkl")

def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")

def apply_homography(point, homography):
    # Convert to homogeneous coordinates
    point_homogeneous = np.array([point[0], point[1], 1.0])
    # Apply homography
    transformed_point = homography @ point_homogeneous
    # Convert back to image coordinates
    transformed_point = transformed_point / transformed_point[2]
    return transformed_point[:2]

def test_all_homography_matrix():
    global clicked_point
    inter_camera_info_list = load_pickle("inter_camera_info.pkl")

    for inter_camera_info in inter_camera_info_list:
        camera_number_1 = inter_camera_info.camera_number_1
        camera_number_2 = inter_camera_info.camera_number_2
        homography = inter_camera_info.homography
        print(f"Testing homography: Camera {camera_number_1} to Camera {camera_number_2}")

        img1 = cv2.imread(f"{PATH_FRAME}/cam_{camera_number_1}.png")
        img2 = cv2.imread(f"{PATH_FRAME}/cam_{camera_number_2}.png")

        # Ensure both images have the same height
        max_height = max(img1.shape[0], img2.shape[0])
        img1 = cv2.copyMakeBorder(img1, 0, max_height - img1.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        img2 = cv2.copyMakeBorder(img2, 0, max_height - img2.shape[0], 0, 0, cv2.BORDER_CONSTANT)

        # Create a window with both images side by side
        combined_img = np.hstack((img1, img2))
        window_name = f"Camera {camera_number_1} and Camera {camera_number_2}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            display_img = combined_img.copy()

            if clicked_point:
                # Check if the click is on the left image
                if clicked_point[0] < img1.shape[1]:
                    cv2.circle(display_img, clicked_point, 5, (0, 255, 0), -1)
                    
                    # Transform the clicked point to the second camera's image
                    point_src = np.array(clicked_point)
                    point_dst = apply_homography(point_src, homography)
                    
                    # Adjust x-coordinate for the right image
                    point_dst = (int(point_dst[0]) + img1.shape[1], int(point_dst[1]))
                    
                    cv2.circle(display_img, point_dst, 5, (0, 0, 255), -1)
                    cv2.line(display_img, clicked_point, point_dst, (255, 0, 0), 2)

            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    calculateHomographyMatrix()
    test_all_homography_matrix()