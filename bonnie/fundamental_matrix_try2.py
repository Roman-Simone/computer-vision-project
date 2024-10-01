import numpy as np 
import json 
import pickle
import os
import matplotlib.pyplot as plt
import cv2
from config import *
from utils import *

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)

def draw_epipolar_line(img, epipolar_line, color):
    """Draw epipolar line on an image, taking scale factors into account."""
    r, c = img.shape[:2]
    
    # Scale the epipolar line coordinates
    x0, y0 = map(int, [0, (-epipolar_line[2] / epipolar_line[1])])
    x1, y1 = map(int, [c, (-(epipolar_line[2] + epipolar_line[0] * c) / epipolar_line[1])])
    
    print(f"Drawing line from ({x0}, {y0}) to ({x1}, {y1}), dimensions: {img.shape}")
    
    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    return img

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def compute_relative_motion(R1, T1, R2, T2):
    
    R_relative = R2 @ R1.T                  # rotation from camera 1's coordinates to camera 2's coordinates
    T_relative = T2 - R_relative @ T1        # relative translation from camera 1's origin to camera 2's origin, accounting for their orientations.
    return R_relative, T_relative

def compute_essential_matrix(R, T):
    
    # skew-symmetric matrix of the translation vector
    T_skew = skew_symmetric(T)
    
    # Essential matrix E = [T]x * R
    E = T_skew @ R
    return E

def load_calibration_data(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        camera_infos = pickle.load(f)  
    return camera_infos


def compute_fundamental_matrix(E, K1, K2):
    # F = K2^-T * E * K1^-1
    K1_inv = np.linalg.inv(K1)
    K2_inv_T = np.linalg.inv(K2).T
    F = K2_inv_T @ E @ K1_inv
    return F

def compute_all(camera_number_1, camera_number_2):
    
    camera_data = load_calibration_data(PATH_CALIBRATION_MATRIX)
    
    camera_1_info = next((cam for cam in camera_data if cam.camera_number == camera_number_1), None)
    camera_2_info = next((cam for cam in camera_data if cam.camera_number == camera_number_2), None)

    # stored vectors are the inverses, it means the translation from the world to the camera, i want the opposite
    
    R1 = np.array(camera_1_info.inverse_rotation_matrix).T              
    T1 = -np.array(camera_1_info.inverse_translation_vector).flatten()  

    R2 = np.array(camera_2_info.inverse_rotation_matrix).T              
    T2 = -np.array(camera_2_info.inverse_translation_vector).flatten()  


    print("\nCamera ", camera_number_1, " Rotation Matrix:\n", R1, "\nTranslation Vector:\n", T1)
    print("Camera ", camera_number_2, " Rotation Matrix:\n", R2, "\nTranslation Vector:\n", T2)

    if len(R1) == 0 or len(T1) == 0:
        print("Error: Camera ", camera_number_1, " data not found.")
        return None
    elif len(R2) == 0 or len(T2) == 0:
        print("Error: Camera ", camera_number_2, " data not found.")
        return None

    R_relative, T_relative = compute_relative_motion(R1, T1, R2, T2)

    E = compute_essential_matrix(R_relative, T_relative)

    # print("Essential Matrix:\n", E)

    if camera_1_info and camera_2_info:
        K1 = camera_1_info.newcameramtx  
        K2 = camera_2_info.newcameramtx  

        # print(f"Camera 2 Intrinsic Matrix (K1): \n{K1}")
        # print(f"Camera 3 Intrinsic Matrix (K2): \n{K2}")

        F = compute_fundamental_matrix(E, K1, K2)

        print("\nFundamental Matrix:\n", F)

    else:
        if not camera_1_info:
            print("Camera ", str(camera_number_1), " not found in the calibration data.")
        if not camera_2_info:
            print("Camera ", str(camera_number_2), " not found in the calibration data.")
    return E, F
            
def resize_with_aspect_ratio(image, max_width, max_height):
    original_height, original_width = image.shape[:2]
    scale_width = max_width / original_width
    scale_height = max_height / original_height
    scale = min(scale_width, scale_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image
            
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Se si clicca con il pulsante sinistro
        param['clicked_point'] = (x, y)
        print(f"Mouse clicked at: {x}, {y}")

def epipolar_line(point1, F):
    """
        compute epipolar line of img2 given a point in img1 and the fundamental matrix.
    """
    print("-------------------------------> Point1: ", point1)

    point1_homogeneous = np.array([point1[0], point1[1], 1])
    
    line2 = F @ point1_homogeneous
    
    return line2
            

def take_points(img1, img2, camera_number, fundamental_mtx):
    clicked_point = {}
    print(f"Select a point in the image for camera {camera_number}")
    window_name = f"Select Point Camera {camera_number}"

    img1_copy = img1.copy()
    img2_copy = img2.copy()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_point, clicked_point)

    while True:
        combined_img = np.hstack((img1_copy, img2_copy))
        cv2.imshow(window_name, combined_img)
        key = cv2.waitKey(1) & 0xFF

        if 'clicked_point' in clicked_point:
            pt1_resized = clicked_point['clicked_point']
            print(f"---> Point selected at {pt1_resized}")

            # Adjust point to original image scale
            pt1_original = (int(pt1_resized[0]), int(pt1_resized[1]))
            print(f"Point in original img1: {pt1_original}")

            # Convert point to array format for cv2 operations
            pt1 = np.array(pt1_original, dtype=np.float32)
            
            # Compute epipolar line in img2 using the fundamental matrix
            # lines2 = cv2.computeCorrespondEpilines(pt1.reshape(-1, 1, 2), 1, fundamental_mtx)
            
            print("Point1: ", pt1)
            
            lines2 = epipolar_line(pt1, fundamental_mtx)
            
            print(f"\nEpipolar line: {lines2.shape}")

            lines2 = lines2.reshape(-1, 3)  # Epipolar line in img2 for point in img1

            print(f"\nEpipolar line in img2 for point in img1: {lines2}")

            # random color for drawing
            color = tuple(np.random.randint(0, 256, 3).tolist())

            # Draw the selected point on img1 and epipolar line on img2
            img1_copy = cv2.circle(img1_copy, pt1_resized, 5, color, -1)
            img2_copy = draw_epipolar_line(img2_copy, lines2[0], color)

            combined_img = np.hstack((img1_copy, img2_copy))
            
            cv2.imshow(window_name, combined_img)
            

            # Save the combined image
            cv2.imwrite(f'combined_image_camera_{camera_number}.png', combined_img)
            print(f"Combined image saved as combined_image_camera_{camera_number}.png")

            # Clear the clicked point to allow for the next selection
            clicked_point.clear()

        elif key == ord('q'):
            print("Exiting without selecting a point.")
            cv2.destroyWindow(window_name)
            return None


if __name__ == "__main__":
    
    plt.close('all')
    
    while True:
        try:
            camera_number_1 = input("Enter the first camera number to use (or type 'exit' to quit): ")

            if camera_number_1.lower() == 'exit':
                print("Exiting the program.")
                break

            camera_number_2 = input("Enter the second camera number to use (or type 'exit' to quit): ")

            if camera_number_2.lower() == 'exit':
                print("Exiting the program.")
                break

            camera_number_1 = int(camera_number_1)
            camera_number_2 = int(camera_number_2)

            essential_mtx, fundamental_mtx = compute_all(camera_number_1, camera_number_2)

            video_path_1 = path_videos + '/out' + str(camera_number_1) + '.mp4'
            print("Trying to open ", video_path_1)
            cap1 = cv2.VideoCapture(video_path_1)

            ret1, img1 = cap1.read()
            if not ret1 or img1 is None:
                print("Error reading video file for camera", camera_number_1)
                cap1.release()
                continue  

            cap1.release()  

            video_path_2 = path_videos + '/out' + str(camera_number_2) + '.mp4'
            print("Trying to open ", video_path_2)
            cap2 = cv2.VideoCapture(video_path_2)

            ret2, img2 = cap2.read()  
            if not ret2 or img2 is None:
                print("Error reading video file for camera", camera_number_2)
                cap2.release()
                continue

            cap2.release()

            
            camera_infos = load_calibration_data(PATH_CALIBRATION_MATRIX)
            
            camera1_info =  next((cam for cam in camera_infos if cam.camera_number == camera_number_1), None)
            camera2_info =  next((cam for cam in camera_infos if cam.camera_number == camera_number_1), None)
            
            if img1 is not None and img2 is not None:
                img1, _ = undistorted(img1, camera1_info)
                img2, _ = undistorted(img2, camera2_info)

                selected_point = take_points(img1, img2, camera_number_1, fundamental_mtx)
            else:
                print("One of the images could not be loaded.")


            if selected_point is not None:
                print(f"Selected point in img1: {selected_point}")

        except ValueError:
            print("Invalid input. Please enter a valid camera number.")

