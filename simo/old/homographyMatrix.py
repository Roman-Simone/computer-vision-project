import numpy as np
from utils import *
from config import *
from cameraInfo import *

clicked_point = None

# calculate homography from points previously selected
def calculateHomographyMatrix():

    InterCameraInfolist = []

    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

    for camera_number1 in VALID_CAMERA_NUMBERS:

        camera_info1, _ = take_info_camera(camera_number1, camera_infos)

        for camera_number2 in VALID_CAMERA_NUMBERS:

            if camera_number1 == camera_number2:
                continue

            camera_info2, _ = take_info_camera(camera_number2, camera_infos)

            # Read the data
            coordinates_by_camera = read_json_file_and_structure_data(PATH_JSON)

            for camera_id, coords in coordinates_by_camera.items():
                if int(camera_id) == camera_number1:
                    world_points1 = np.array(coords["world_coordinates"], dtype=np.float32)
                    image_points1 = np.array(coords["image_coordinates"], dtype=np.float32)
                
                if int(camera_id) == camera_number2:
                    world_points2 = np.array(coords["world_coordinates"], dtype=np.float32)
                    image_points2 = np.array(coords["image_coordinates"], dtype=np.float32)


            # find common points
            common_points1 = []
            common_points2 = []
            for i in range(len(world_points1)):
                for j in range(len(world_points2)):
                    if world_points1[i][0] == world_points2[j][0] and world_points1[i][1] == world_points2[j][1] and world_points1[i][2] == world_points2[j][2]:
                        
                        common_points1.append(image_points1[i])
                        common_points2.append(image_points2[j])

            if len(common_points1) < 4 or len(common_points2) < 4:
                print("No sufficient points to calculate homography matrix")
                continue

            common_points1 = np.array(common_points1, dtype=np.float32)
            common_points2 = np.array(common_points2, dtype=np.float32)
            
            # calculate homography matrix

            homography_matrix = cv2.findHomography(common_points1, common_points2, cv2.RANSAC, 5.0)[0]

            inter_camera_info = InterCameraInfo(camera_number1, camera_number2)

            inter_camera_info.homography = homography_matrix

            InterCameraInfolist.append(inter_camera_info)

    save_pickle(InterCameraInfolist, "inter_camera_info.pkl")



def on_mouse(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")
    


def test_all_homography_matrix():

    global clicked_point

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
        
        window_name = f"Camera {camera_number_1}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        cv2.imshow(window_name, img1)
        key = cv2.waitKey(0) & 0xFF
            
if __name__ == "__main__":
    
    calculateHomographyMatrix()

    # test homography matrix
    test_all_homography_matrix()