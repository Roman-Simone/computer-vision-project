import numpy as np 
import json 
import pickle
import os
import cv2

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)

def draw_epipolar_line(img, line, color=(255, 0, 0)):
    """ Draw the epipolar line on an image. """
    h, w = img.shape[:2]
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [w, -(line[2] + line[0] * w) / line[1]])
    img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
    
    return img

def find_corresponding_point(F, pt1):
    """ Given a point in img1, find the corresponding epipolar line in img2. """
    pt1_homogeneous = np.array([pt1[0], pt1[1], 1])  # homogeneous coordinates
    epipolar_line = np.dot(F, pt1_homogeneous)  # epipolar line in img2
    return epipolar_line

def closest_point_on_line(epipolar_line, img2_shape):
    """ Given an epipolar line, find the closest point on the line in img2. """
    h, w = img2_shape
    # grid of points in img2
    points = np.array(np.meshgrid(np.arange(w), np.arange(h))).reshape(2, -1).T
    distances = np.abs(epipolar_line[0] * points[:, 0] + epipolar_line[1] * points[:, 1] + epipolar_line[2])
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]
    return tuple(closest_point)

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
    
    json_file_path = 'camera_data.json'

    with open(json_file_path, 'r') as json_file:    
        camera_data = json.load(json_file)  # JSON string into a dictionary

    R1 = np.array(camera_data[str(camera_number_1)]["inverse_rotation_matrix"])
    T1 = np.array(camera_data[str(camera_number_1)]["inverse_translation_vector"]).flatten()  
    R2 = np.array(camera_data[str(camera_number_2)]["inverse_rotation_matrix"])
    T2 = np.array(camera_data[str(camera_number_2)]["inverse_translation_vector"]).flatten()  

    R_relative, T_relative = compute_relative_motion(R1, T1, R2, T2)

    E = compute_essential_matrix(R_relative, T_relative)

    print("Essential Matrix:\n", E)

    pickle_file_path = parent_path + '/data/calibrationMatrix/calibration.pkl'

    camera_infos = load_calibration_data(pickle_file_path)

    camera_1_info = next((cam for cam in camera_infos if cam.camera_number == camera_number_1), None)
    camera_2_info = next((cam for cam in camera_infos if cam.camera_number == camera_number_2), None)

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
            
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Se si clicca con il pulsante sinistro
        param['clicked_point'] = (x, y)
        print(f"Mouse clicked at: {x}, {y}")

def take_points(img1, img2, camera_number, fundamental_mtx):
    clicked_point = {}

    print(f"Select a point in the image for camera {camera_number}")
    window_name = f"Select Point Camera {camera_number}"
    cv2.namedWindow(window_name)
    
    # Definisci la larghezza e l'altezza desiderate per la finestra
    window_width = 1280
    window_height = 600
    cv2.resizeWindow(window_name, window_width, window_height)
    
    # Ridimensiona le immagini per adattarle alla finestra
    img1_resized = cv2.resize(img1, (window_width // 2, window_height))  # metà larghezza per ogni immagine
    img2_resized = cv2.resize(img2, (window_width // 2, window_height))

    img1_copy = img1_resized.copy()
    img2_copy = img2_resized.copy()
    
    cv2.setMouseCallback(window_name, select_point, clicked_point)

    while True:
        combined_img = np.hstack((img1_copy, img2_copy))  # combina le immagini ridimensionate
        cv2.imshow(window_name, combined_img)
        key = cv2.waitKey(1) & 0xFF

        if 'clicked_point' in clicked_point:
            pt1 = clicked_point['clicked_point']
            print(f"---> Point selected at {pt1}")

            # corrispondente linea epipolare e punto nella seconda immagine
            epipolar_line = find_corresponding_point(fundamental_mtx, pt1)
            closest_pt2 = closest_point_on_line(epipolar_line, img2_resized.shape)

            img1_copy = cv2.circle(img1_copy, pt1, 5, (255, 0, 0), -1)
            img2_copy = cv2.circle(img2_copy, closest_pt2, 5, (0, 255, 0), -1)

            img2_copy = draw_epipolar_line(img2_copy, epipolar_line)

            clicked_point.clear()
            
            combined_img = np.hstack((img1_copy, img2_copy))  
            cv2.imshow(window_name, combined_img)
            
            cv2.imwrite(f'combined_image_camera_{camera_number}.png', combined_img)
            print(f"Combined image saved as combined_image_camera_{camera_number}.png")
            
            cv2.waitKey(0)

            break
        elif key == ord('q'):
            print("Exiting without selecting a point.")
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    
    return pt1  

if __name__ == "__main__":
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

            video_path_1 = parent_path + '/23_09_23 amichevole trento volley/out' + str(camera_number_1) + '.mp4'
            print("Trying to open ", video_path_1)
            cap1 = cv2.VideoCapture(video_path_1)
            
            ret1, frame1 = cap1.read()
            if not ret1:
                print("Error reading video file")
            img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            cap1.release()

            video_path_2 = parent_path + '/23_09_23 amichevole trento volley/out' + str(camera_number_2) + '.mp4'
            print("Trying to open ", video_path_2)
            cap2 = cv2.VideoCapture(video_path_2)            
            
            ret2, frame2 = cap2.read()
            if not ret2:
                print("Error reading video file")
            img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            cap2.release()

            selected_point = take_points(img1, img2, camera_number_1, fundamental_mtx)

            if selected_point is not None:
                print(f"Selected point in img1: {selected_point}")

        except ValueError:
            print("Invalid input. Please enter a valid camera number.")