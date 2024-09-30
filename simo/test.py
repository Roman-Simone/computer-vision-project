import numpy as np
import cv2
from utils import *
from config import *
from cameraInfo import InterCameraInfo

def calculate_homography_matrix(camera_info1, camera_info2):
    # Extract camera matrices
    P1 = camera_info1.newcameramtx @ np.hstack((camera_info1.extrinsic_matrix[:3, :3], camera_info1.extrinsic_matrix[:3, 3].reshape(3,1)))
    P2 = camera_info2.newcameramtx @ np.hstack((camera_info2.extrinsic_matrix[:3, :3], camera_info2.extrinsic_matrix[:3, 3].reshape(3,1)))
    
    # Calculate homography
    H = P2 @ np.linalg.pinv(P1)
    
    return H[:3, :3]  # Return the 3x3 homography matrix

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
        
        if img1 is None or img2 is None:
            print(f"Error: Unable to read images for cameras {camera_number_1} and {camera_number_2}")
            continue
        
        # Resize images to have the same height
        height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))
        
        # find common part of image 1 and image 2 using homography matrix
        h, w = img1.shape[:2]
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, homography)
        
        # Draw the transformed corners on img2
        img2_with_corners = img2.copy()
        img2_with_corners = cv2.polylines(img2_with_corners,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
        
        # Warp img1 to align with img2
        result = cv2.warpPerspective(img1, homography, (img2.shape[1], img2.shape[0]))
        
        # Blend the two images
        alpha = 0.5
        beta = 1.0 - alpha
        blended = cv2.addWeighted(img2, alpha, result, beta, 0.0)
        
        # Create side-by-side comparisons
        comparison1 = np.hstack((img1, img2_with_corners))
        comparison2 = np.hstack((result, blended))
        
        # Display the results
        cv2.imshow(f"Original Images (Left: Camera {camera_number_1}, Right: Camera {camera_number_2} with projected corners)", comparison1)
        cv2.imshow(f"Results (Left: Warped Image, Right: Blended Result)", comparison2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calculateHomographyMatrix()
    
    # test homography matrix
    test_all_homography_matrix()