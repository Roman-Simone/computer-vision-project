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

def draw_epipolar_line(img, line, color=(0, 255, 0)):
    """ Draw the epipolar line on an image. """
    
    h, w = img.shape[:2]
    # Calculate the points (x0, y0) and (x1, y1) of the line
    if line[1] != 0:  # To avoid division by zero
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [w, -(line[2] + line[0] * w) / line[1]])
        
        # Clip the points to be within the image boundaries
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h - 1))
    else:
        x0, y0 = map(int, [0, -line[2] / line[0]])
        x1, y1 = map(int, [w, -line[2] / line[0]])

    print(f"\n\nDrawing line from ({x0}, {y0}) to ({x1}, {y1})\n\n")
    # Draw the line
    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    return img

def find_epipolar_line(F, pt1):
    """ Given a point in img1, find the corresponding epipolar line in img2. """
    pt1_np = np.array(pt1)
    pt1_reshaped = pt1_np.reshape(1, 1, 2)     
    epilines = cv2.computeCorrespondEpilines(pt1_reshaped, 1, F)
    epipolar_line = epilines[0][0]  # For the first point
    return epipolar_line
    
def best_point_img2(epipolar_line, img2, pt1, window_size=5):
    """ Find the best corresponding point on the epipolar line using a window-based approach.
    Args:
        epipolar_line: The epipolar line equation (a, b, c) in img2.
        img2: The second image where we search for the corresponding point.
        pt1: The selected point in img1.
        window_size: The size of the square window (default is 5).
    Returns:
        best_pt2: The best corresponding point in img2.
    """
    h, w = img2.shape[:2]
    half_window = window_size // 2

    # Extract the window around the selected point in img1
    pt1_x, pt1_y = pt1
    window1 = img2[pt1_y - half_window: pt1_y + half_window + 1, 
                   pt1_x - half_window: pt1_x + half_window + 1]

    best_pt2 = None
    min_ssd = float('inf')

    # Scan along the epipolar line in img2
    for x in range(w):
        # Calculate the corresponding y-coordinate on the epipolar line
        y = int(-(epipolar_line[0] * x + epipolar_line[2]) / epipolar_line[1])
        
        # Ensure the window is within image bounds
        if y - half_window < 0 or y + half_window >= h or x - half_window < 0 or x + half_window >= w:
            continue
        
        # Extract the window around the current point on the epipolar line in img2
        window2 = img2[y - half_window: y + half_window + 1, 
                       x - half_window: x + half_window + 1]

        # Calculate the SSD between the two windows
        ssd = np.sum((window1 - window2) ** 2)

        # Update if the current SSD is the best so far
        if ssd < min_ssd:
            min_ssd = ssd
            best_pt2 = (x, y)

    return best_pt2

# def best_point_img2(epipolar_line, img2_shape):
#     """ Given an epipolar line, find the closest point on the line in img2. """
#     h, w = img2_shape
#     # grid of points in img2
#     points = np.array(np.meshgrid(np.arange(w), np.arange(h))).reshape(2, -1).T
#     distances = np.abs(epipolar_line[0] * points[:, 0] + epipolar_line[1] * points[:, 1] + epipolar_line[2])
#     closest_idx = np.argmin(distances)
#     closest_point = points[closest_idx]
#     return tuple(closest_point)

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def compute_relative_motion(R1, T1, R2, T2):
    # relative rotation and translation
    R_relative = R2 @ R1.T
    T_relative = T2 - R2 @ R1.T @ T1
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

    R1 = np.array(camera_1_info.inverse_rotation_matrix)
    T1 = np.array(camera_1_info.inverse_translation_vector).flatten()  
    R2 = np.array(camera_2_info.inverse_rotation_matrix)
    T2 = np.array(camera_2_info.inverse_translation_vector).flatten()  

    if len(R1) == 0 or len(T1) == 0:
        print("Error: Camera ", camera_number_1, " data not found.")
        return None
    elif len(R2) == 0 or len(T2) == 0:
        print("Error: Camera ", camera_number_2, " data not found.")
        return None

    R1 = R1.T  # Inverse of R1 (transpose of R1)
    R2 = R2.T  # Inverse of R2 (transpose of R2)

    # Compute the inverse of the translation vectors
    T1 = -T1  # Inverse of T1 (negative of T1)
    T2 = -T2

    R_relative, T_relative = compute_relative_motion(R1, T1, R2, T2)

    E = compute_essential_matrix(R_relative, T_relative)

    print("Essential Matrix:\n", E)

    if camera_1_info and camera_2_info:
        K1 = camera_1_info.newcameramtx  
        K2 = camera_2_info.newcameramtx  

        print(f"Camera 2 Intrinsic Matrix (K1): \n{K1}")
        print(f"Camera 3 Intrinsic Matrix (K2): \n{K2}")

        F = compute_fundamental_matrix(E, K1, K2)

        print("Fundamental Matrix:\n", F)

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
        
def take_points(img1, img2, camera_number, fundamental_mtx):
    clicked_point = {}
    print(f"Select a point in the image for camera {camera_number}")
    window_name = f"Select Point Camera {camera_number}"

    window_width = 1600
    window_height = 900

    # Get original dimensions
    orig_h1, orig_w1 = img1.shape[:2]  
    orig_h2, orig_w2 = img2.shape[:2]

    img1_resized = resize_with_aspect_ratio(img1, window_width // 2, window_height)
    img2_resized = resize_with_aspect_ratio(img2, window_width // 2, window_height)

    # Get resized dimensions for mapping
    resized_h1, resized_w1 = img1_resized.shape[:2]
    resized_h2, resized_w2 = img2_resized.shape[:2]

    img1_copy = img1_resized.copy()
    img2_copy = img2_resized.copy()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_point, clicked_point)

    while True:
        combined_img = np.hstack((img1_copy, img2_copy))
        cv2.imshow(window_name, combined_img)
        key = cv2.waitKey(1) & 0xFF

        if 'clicked_point' in clicked_point:
            pt1_resized = clicked_point['clicked_point']
            print(f"---> Point selected at {pt1_resized}")

            # Map the point back to original size
            scale_x1 = orig_w1 / resized_w1
            scale_y1 = orig_h1 / resized_h1

            # Adjust point to original image scale
            pt1_original = (int(pt1_resized[0] * scale_x1), int(pt1_resized[1] * scale_y1))
            print(f"Point in original img1: {pt1_original}")

            # Process the selected point
            epipolar_line = find_epipolar_line(fundamental_mtx, pt1_original)

            # The second image also needs to be mapped to the original coordinates
            best_pt2_resized = best_point_img2(epipolar_line, img2, pt1_original)
            if best_pt2_resized is None:
                print("No corresponding point found.")
                continue

            # Map the best_pt2 to the resized image for visualization
            scale_x2 = orig_w2 / resized_w2
            scale_y2 = orig_h2 / resized_h2
            best_pt2_resized = (int(best_pt2_resized[0] / scale_x2), int(best_pt2_resized[1] / scale_y2))

            # Use a random color for drawing
            color = tuple(np.random.randint(0, 256, 3).tolist())
            img1_copy = cv2.circle(img1_copy, pt1_resized, 5, color, -1)
            img2_copy = cv2.circle(img2_copy, best_pt2_resized, 5, color, -1)
            img2_copy = draw_epipolar_line(img2_copy, epipolar_line, color)

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

            ret1, img1 = cap1.read()  # Returns the status and the frame (image)
            if not ret1 or img1 is None:
                print("Error reading video file for camera", camera_number_1)
                cap1.release()
                continue  # Go back to re-prompt the user

            cap1.release()  # Don't forget to release the video capture object


            video_path_2 = path_videos + '/out' + str(camera_number_2) + '.mp4'
            print("Trying to open ", video_path_2)
            cap2 = cv2.VideoCapture(video_path_2)

            ret2, img2 = cap2.read()  # Same process for the second video
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