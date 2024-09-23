import numpy as np 
import json 
import pickle

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def compute_relative_motion(R1, T1, R2, T2):
    # Compute relative rotation and translation
    R_relative = R2 @ R1.T
    T_relative = T2 - R2 @ R1.T @ T1
    return R_relative, T_relative

def compute_essential_matrix(R, T):
    
    # Compute skew-symmetric matrix of the translation vector
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
    T1 = np.array(camera_data[str(camera_number_1)]["inverse_translation_vector"]).flatten()  # Convert to 1D array
    R2 = np.array(camera_data[str(camera_number_2)]["inverse_rotation_matrix"])
    T2 = np.array(camera_data[str(camera_number_2)]["inverse_translation_vector"]).flatten()  # Convert to 1D array

    R_relative, T_relative = compute_relative_motion(R1, T1, R2, T2)

    E = compute_essential_matrix(R_relative, T_relative)

    print("Essential Matrix:\n", E)

    pickle_file_path = '/home/bonnie/Desktop/computer vision/project/Computer_Vision_project/data/calibrationMatrix/calibration.pkl'

    # Load the camera calibration data
    camera_infos = load_calibration_data(pickle_file_path)

    # Access intrinsic parameters for a specific camera (e.g., camera number 2)
    camera_1_info = next((cam for cam in camera_infos if cam.camera_number == camera_number_1), None)
    camera_2_info = next((cam for cam in camera_infos if cam.camera_number == camera_number_2), None)

    if camera_1_info and camera_2_info:
        K1 = camera_1_info.newcameramtx  # Intrinsic matrix of Camera 2
        K2 = camera_2_info.newcameramtx  # Intrinsic matrix of Camera 3

        print(f"Camera 2 Intrinsic Matrix (K1): \n{K1}")
        print(f"Camera 3 Intrinsic Matrix (K2): \n{K2}")

        # Step 5: Compute the Fundamental matrix using the Essential matrix and camera intrinsics
        F = compute_fundamental_matrix(E, K1, K2)

        print("Fundamental Matrix:\n", F)

    else:
        if not camera_1_info:
            print("Camera ", str(camera_number_1), " not found in the calibration data.")
        if not camera_2_info:
            print("Camera ", str(camera_number_2), " not found in the calibration data.")
            
    return E, F
            
if __name__ == "__main__":
    while True:
        try:
            # Ask the user for the camera number
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

        except ValueError:
            print("Invalid input. Please enter a valid camera number.")