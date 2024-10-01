import numpy as np
from utils import *
from config import *
from cameraInfo import InterCameraInfo


clicked_point = None

def calculate_homography_matrix(camera_info1, camera_info2):

    intrinsic1 = camera_info1.newcameramtx
    rotation1 = camera_info1.extrinsic_matrix[:3, :3]
    translation1 = camera_info1.extrinsic_matrix[:3, 3]

    h1 = intrinsic1 @ np.hstack((rotation1, translation1.reshape(3,1)))

    intrinsic2 = camera_info2.newcameramtx
    rotation2 = camera_info2.extrinsic_matrix[:3, :3]
    translation2 = camera_info2.extrinsic_matrix[:3, 3]

    h2 = intrinsic2 @ np.hstack((rotation2, translation2.reshape(3,1)))

    homography = h2 @ np.linalg.pinv(h1)


    return homography, h1, h2


def calculateHomographyMatrix():

    InterCameraInfolist = []

    camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)

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

            print("h1 shape: ", h1.shape)
            print("h2 shape: ", h2.shape)
            print("Homography matrix shape: ", homography_matrix.shape)
            print("\n")

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
        # camera_info1, _ = take_info_camera(camera_number_1, load_pickle(PATH_CALIBRATION_MATRIX))
        # img1 = undistorted(img1, camera_info1)
        img2 = cv2.imread(f"{PATH_FRAME}/cam_{camera_number_2}.png")
        # camera_info2, _ = take_info_camera(camera_number_2, load_pickle(PATH_CALIBRATION_MATRIX))
        # img2 = undistorted(img2, camera_info2)
        
        window_name = f"Camera {camera_number_1}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            cv2.imshow(window_name, img1)

            if clicked_point:
                # Trasforma il punto cliccato nell'immagine della seconda telecamera usando l'omografia
                point_src = np.array([clicked_point[0], clicked_point[1], 1.0], dtype=np.float32)
                point_dst = homography @ point_src

                # Normalizza le coordinate omogenee
                point_dst = point_dst / point_dst[2]
                point_dst = tuple(map(int, point_dst[:2]))

                # Disegna il punto nell'immagine della seconda telecamera
                img2_with_point = img2.copy()
                cv2.circle(img2_with_point, point_dst, 10, (0, 0, 255), -1)

                # Mostra l'immagine della seconda telecamera con il punto proiettato
                window_name2 = f"Camera {camera_number_2} (Proiezione del punto)"
                cv2.imshow(window_name2, img2_with_point)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


        
        




if __name__ == "__main__":
    calculateHomographyMatrix()

    # test homography matrix
    test_all_homography_matrix()