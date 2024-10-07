import cv2
import numpy as np
from config import *
from utils import *

available_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
interInfo = load_pickle(PATH_HOMOGRAPHY_MATRIX)
cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)
resize = False

def ret_homography(camera_src, camera_dst):
    inter_camera_info = next((inter for inter in interInfo if inter.camera_number_1 == camera_src and inter.camera_number_2 == camera_dst), None)
    return inter_camera_info.homography

def show_videos(camera_src, camera_dst):
    while True:
        
        window_name = "Camera " + str(camera_src) + " and " + str(camera_dst) + " - Click on the image to see the corresponding point"

        title_height = 100
        text_height = 80  
        
        def mouse_callback(event, x, y, flags, param):

            if event == cv2.EVENT_LBUTTONDOWN:

                y_adjusted = y - title_height

                if y_adjusted >= 0:

                    point = np.array([[x + camera_info_1.roi[0], y_adjusted + camera_info_1.roi[1]]], dtype=np.float32)

                    point_transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), homography).reshape(-1, 2)

                    cv2.circle(img_src_resized, (int(x), int(y_adjusted)), 15, (0, 255, 0), -1)

                    scale_x = img_dst_resized.shape[1] / img_dst.shape[1]
                    scale_y = img_dst_resized.shape[0] / img_dst.shape[0]

                    x_transformed = int((point_transformed[0][0] - camera_info_2.roi[0]) * scale_x)
                    y_transformed = int((point_transformed[0][1] - camera_info_2.roi[1]) * scale_y)

                    cv2.circle(img_dst_resized, (x_transformed, y_transformed), 15, (0, 255, 0), -1)

                    combined_img_right = np.vstack((title_source, img_src_resized))
                    combined_img_left = np.vstack((title_destination, img_dst_resized))
    
                    combined = np.hstack((combined_img_right, combined_img_left))

                    # combined = np.hstack((img_src_resized, img_dst_resized))

                    cv2.imshow(window_name, combined)
        
        homography = ret_homography(camera_src, camera_dst)
        
        if homography is None:
            print(f"No homography available for cameras {camera_src} and {camera_dst}")
            continue
        
        img_src = cv2.imread(f"{PATH_FRAME_DISTORTED}/cam_{camera_src}.png")
        img_dst = cv2.imread(f"{PATH_FRAME_DISTORTED}/cam_{camera_dst}.png")  
        
        camera_info_1, _ = take_info_camera(camera_src, cameras_info)
        camera_info_2, _ = take_info_camera(camera_dst, cameras_info)
        
        img_src = undistorted(img_src, camera_info_1)
        img_dst = undistorted(img_dst, camera_info_2)
        
        # cv2.imshow("img_src", img_src)
        # cv2.imshow("img_dst", img_dst)
        # cv2.waitKey(0)
        
        if img_src is None or img_dst is None:
            print(f"Could not load images for cameras {camera_src} and {camera_dst}")
            continue
        
        if resize:
            target_width = 900

            h_src, w_src, _ = img_src.shape
            h_dst, w_dst, _ = img_dst.shape
            new_height_src = int(target_width * h_src / w_src)
            new_height_dst = int(target_width * h_dst / w_dst)

            img_src = cv2.resize(img_src, (target_width, new_height_src))
            img_dst = cv2.resize(img_dst, (target_width, new_height_dst))

       
        height_src, width_src = img_src.shape[:2]
        height_dst, width_dst = img_dst.shape[:2]
        
        if height_src != height_dst:
            if height_src > height_dst:
                img_dst_resized = cv2.resize(img_dst, (width_dst * height_src // height_dst, height_src))
                img_src_resized = img_src
            else:
                img_src_resized = cv2.resize(img_src, (width_src * height_dst // height_src, height_dst))
                img_dst_resized = img_dst
        else:
            img_src_resized = img_src
            img_dst_resized = img_dst

        title_source = np.zeros((title_height, img_src_resized.shape[1], 3), dtype=np.uint8)
        title_destination = np.zeros((title_height, img_dst_resized.shape[1], 3), dtype=np.uint8)

        text_source = f"SOURCE: camera {camera_src}"
        text_destination = f"DESTINATION: camera {camera_dst}"

        text_size_source = cv2.getTextSize(text_source, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_size_destination = cv2.getTextSize(text_destination, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

        text_x_source = (img_src_resized.shape[1] - text_size_source[0]) // 2
        text_x_destination = (img_dst_resized.shape[1] - text_size_destination[0]) // 2

        cv2.putText(title_source, text_source, (text_x_source, text_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(title_destination, text_destination, (text_x_destination, text_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        combined_img_right = np.vstack((title_source, img_src_resized))
        combined_img_left = np.vstack((title_destination, img_dst_resized))

        combined = np.hstack((combined_img_right, combined_img_left))
        # combined = np.hstack((img_src_resized, img_dst_resized))

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        cv2.imshow(window_name, combined)
        
        print(f"Click on the image from Camera {camera_src} to see the corresponding point on Camera {camera_dst}")
        print("Press 'c' to change cameras or 'q' to exit")

        key = cv2.waitKey(0)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            cv2.destroyAllWindows()
            camera_src = int(input("Insert new source camera (1-8, 12, 13): "))
            while camera_src not in available_cameras:
                print("Invalid input. Please choose from available cameras.")
                camera_src = int(input("Insert new source camera (1-8, 12, 13): "))

            camera_dst = int(input("Insert new destination camera (1-8, 12, 13), different from source: "))
            while camera_dst not in available_cameras or camera_dst == camera_src:
                print("Invalid input. Please choose from available cameras, different from source.")
                camera_dst = int(input("Insert new destination camera (1-8, 12, 13), different from source: "))

if __name__ == "__main__":
    camera_src = int(input("Insert new source camera (1-8, 12, 13): "))
    while camera_src not in available_cameras:
        print("Invalid input. Please choose from available cameras.")
        camera_src = int(input("Insert new source camera (1-8, 12, 13): "))

    camera_dst = int(input("Insert new destination camera (1-8, 12, 13), different from source: "))
    while camera_dst not in available_cameras or camera_dst == camera_src:
        print("Invalid input. Please choose from available cameras, different from source.")
        camera_dst = int(input("Insert new destination camera (1-8, 12, 13), different from source: "))

    show_videos(camera_src, camera_dst)