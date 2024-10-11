from config import *
from utils import *
import os

MAX_FRAME = 5100

pathPickle = os.path.join(PATH_DETECTIONS, 'detections.pkl')
detections = load_pickle(pathPickle)
camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

cam = [take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS]

def get_projection_matrix(cam):
    
    K = cam.newcameramtx  
    extrinsic_matrix = cam.extrinsic_matrix  
    
    # get the top 3x4 part (first 3 rows and 4 columns)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]  
    
    # return projection matrix P = K * [R | t]    
    return np.dot(K, extrinsic_matrix_3x4)

proj_matrix = [get_projection_matrix(c) for c in cam]

def triangulate(cam1, cam2, point2d1, point2d2):
    proj1 = get_projection_matrix(cam1)
    proj2 = get_projection_matrix(cam2)

    point2d1 = np.array([point2d1], dtype=np.float32)
    point2d2 = np.array([point2d2], dtype=np.float32)

    point4d = cv2.triangulatePoints(proj1, proj2, point2d1.T, point2d2.T)
    point3d = cv2.convertPointsFromHomogeneous(point4d.T)[0][0]
    return point3d
        
def is_valid_point(point3d):
    
    
    ##############################################################
    ###################### NOT COMPLETE ##########################
    ##############################################################
    
    x, y, z = point3d
    
    if x < 0 or y < 0 or z < 0:
        return False
    else:
        print("Point3d: ", point3d)
    # implementing checks for valid 3D points
    return True
        
def main():
    det_3D = {}  # store valid 3D detections for each frame
    
    for i in range(1, MAX_FRAME):
        det_frame_3D = []  # 3D detections for the current frame
        
        detections_frame = {cam: detections.get((cam, i), []) for cam in VALID_CAMERA_NUMBERS}

        # iterate over pairs of cameras
        for idx1, cam1 in enumerate(VALID_CAMERA_NUMBERS):
            for idx2 in range(idx1 + 1, len(VALID_CAMERA_NUMBERS)):
                cam2 = VALID_CAMERA_NUMBERS[idx2]
                
                detections_cam1 = detections_frame[cam1]
                # print("Detections cam1: ", detections_cam1)
                detections_cam2 = detections_frame[cam2]
                # print("Detections cam2: ", detections_cam2)

                if not detections_cam1 or not detections_cam2:
                    continue

                for point2d1 in detections_cam1:
                    for point2d2 in detections_cam2:
                        point3d = triangulate(cam[idx1], cam[idx2], point2d1, point2d2)
                        # print("Point3d: ", point3d)

                        if is_valid_point(point3d):
                            det_frame_3D.append(point3d)
        
        det_3D[i] = det_frame_3D
    
    save_pickle(det_3D, os.path.join(PATH_DETECTIONS, 'detections_3D.pkl'))

if __name__ == "__main__":
    main()