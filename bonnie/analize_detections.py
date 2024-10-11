from config import *
from utils import *
from matplotlib import pyplot as plt
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
    
    x, y, z = point3d
        
    if z < 0 or x < -14 or y < -7.5 or x > 14 or y > 7.5:
        return False
    else:
        print("Point3d: ", point3d)
        return True
        
def get_positions():
    with open(PATH_CAMERA_POS, "r") as file:
        data = json.load(file)
        positions = data["positions"]
        field_corners = np.array(data["field_corners"]) * 1000
    return positions, field_corners
        
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")  # Create a 3D plot
    ax.set_box_aspect([1, 1, 1])  # Set aspect ratio for the 3D plot

    # Get real corner positions and field corners for comparison
    positions, field_corners = get_positions()

    # Plot real corners on the 3D plot
    ax.scatter(
        field_corners[:, 0],
        field_corners[:, 1],
        field_corners[:, 2],
        c="red",  # Color for real corners
        label="Real Corners",
    )
    
    x_coords, y_coords, z_coords = [], [], []
    
    
    for frame in det_3D:
        print("Points: ", len(det_3D[frame]))
        for point in det_3D[frame]:
            
            
            
            x, y, z = point
            
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

        
    
    # x_coords = [det_3D[i][0] for i in range(len(det_3D))]
    # y_coords = [det_3D[i][1] for i in range(len(det_3D))]
    # z_coords = [det_3D[i][2] for i in range(len(det_3D))]  

    ax.scatter(
        x_coords,  
        y_coords,  
        z_coords,  
        c='blue',  
        label="Tracked Points",
        s=50, 
        marker='o'
    )

    ax.plot(
        x_coords,  
        y_coords,  
        z_coords,  
        color='blue',  
        label="Tracked Path",
    )

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.set_xlim([np.min(x_coords) - 1, np.max(x_coords) + 1])  
    ax.set_ylim([np.min(y_coords) - 1, np.max(y_coords) + 1])  
    ax.set_zlim([np.min(z_coords) - 1, np.max(z_coords) + 1])  

    ax.set_title('3D Tracked Points and Real Corners (with Path)')

    ax.legend()

    set_axes_equal(ax)

    plt.show()


if __name__ == "__main__":
    main()