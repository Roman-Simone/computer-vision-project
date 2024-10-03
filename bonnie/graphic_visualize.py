import cv2
import numpy as np
from config import *

available_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

def show_videos(camera_num_src, camera_num_dst):
    while True:
        video_path_1 = f"{path_videos}/out{camera_num_src}.mp4"
        video_path_2 = f"{path_videos}/out{camera_num_dst}.mp4"

        cap1 = cv2.VideoCapture(video_path_1)
        cap2 = cv2.VideoCapture(video_path_2)

        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()

        if not ret1 or img1 is None:
            print(f"Error reading video file for camera {camera_num_src}")
            cap1.release()
            return

        if not ret2 or img2 is None:
            print(f"Error reading video file for camera {camera_num_dst}")
            cap2.release()
            return

        img1 = cv2.resize(img1, (640, 480))
        img2 = cv2.resize(img2, (640, 480))

        title_source = np.zeros((50, 640, 3), dtype=np.uint8)
        title_destination = np.zeros((50, 640, 3), dtype=np.uint8)

        cv2.putText(title_source, "SOURCE", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(title_destination, "DESTINATION", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        combined_img = np.vstack((title_source, img1))
        combined_img2 = np.vstack((title_destination, img2))

        combined = np.hstack((combined_img, combined_img2))

        cv2.imshow(f"Video Feed camera {camera_num_src} and {camera_num_dst}", combined)

        key = cv2.waitKey(0)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            cv2.destroyAllWindows()
            camera_num_src = int(input("Insert new source camera (1-8, 12, 13): "))
            while camera_num_src not in available_cameras:
                print("Invalid input. Please choose from available cameras.")
                camera_num_src = int(input("Insert new source camera (1-8, 12, 13): "))

            camera_num_dst = int(input("Insert new destination camera (1-8, 12, 13), different from source: "))
            while camera_num_dst not in available_cameras or camera_num_dst == camera_num_src:
                print("Invalid input. Please choose from available cameras, different from source.")
                camera_num_dst = int(input("Insert new destination camera (1-8, 12, 13), different from source: "))

        cap1.release()
        cap2.release()

if __name__ == "__main__":
    camera_num_src = int(input("Insert new source camera (1-8, 12, 13): "))
    while camera_num_src not in available_cameras:
        print("Invalid input. Please choose from available cameras.")
        camera_num_src = int(input("Insert new source camera (1-8, 12, 13): "))

    camera_num_dst = int(input("Insert new destination camera (1-8, 12, 13), different from source: "))
    while camera_num_dst not in available_cameras or camera_num_dst == camera_num_src:
        print("Invalid input. Please choose from available cameras, different from source.")
        camera_num_dst = int(input("Insert new destination camera (1-8, 12, 13), different from source: "))

    show_videos(camera_num_src, camera_num_dst)
