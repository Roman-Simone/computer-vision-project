from config import *
from utils import *
import os
import pickle

MAX_FRAME = 5100

pathPickle = os.path.join(PATH_DETECTIONS, 'detections.pkl')
detections = load_pickle(pathPickle)
camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

cam = [take_info_camera(n, camerasInfo) for n in VALID_CAMERA_NUMBERS]
proj_matrix = [get_projection_matrix(c) for c in cam]

def get_projection_matrix(cam):
    
    K = cam.newcameramtx  
    extrinsic_matrix = cam.extrinsic_matrix  
    
    # get the top 3x4 part (first 3 rows and 4 columns)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]  
    
    # return projection matrix P = K * [R | t]    
    return np.dot(K, extrinsic_matrix_3x4)

def triangulate(cam1, cam2, point2d1, point2d2):
    proj1 = cam1.get_projection_matrix(cam1)
    proj2 = cam2.get_projection_matrix(cam2)

    point2d1 = np.array([point2d1], dtype=np.float32)
    point2d2 = np.array([point2d2], dtype=np.float32)

    point4d = cv2.triangulatePoints(proj1, proj2, point2d1.T, point2d2.T)
    point3d = cv2.convertPointsFromHomogeneous(point4d.T)[0][0]
    return point3d
        
def main():
    for i in range(1, MAX_FRAME):
        det_frame = {}
        
        for cam in [1, 2, 3, 4, 5, 6, 7, 8]:
            det_frame[cam] = detections[(cam, i)]
            
        print("\n\n", det_frame)