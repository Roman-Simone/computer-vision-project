from config import *
from utils import *
from matplotlib import pyplot as plt
import os

MAX_FRAME = 5100

pathPickle = os.path.join(PATH_DETECTIONS, 'detections.pkl')
detections = load_pickle(pathPickle)
camerasInfo = load_pickle(PATH_CALIBRATION_MATRIX)

cam = [take_info_camera(n, camerasInfo)[0] for n in VALID_CAMERA_NUMBERS]

count_valid = 0
count_not_valid = 0

def get_projection_matrix(cam):
    
    K = cam.newcameramtx  
    extrinsic_matrix = cam.extrinsic_matrix  
    
    # get the top 3x4 part (first 3 rows and 4 columns)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]  
    
    # return projection matrix P = K * [R | t]    
    return np.dot(K, extrinsic_matrix_3x4)

proj_matrix = [get_projection_matrix(c) for c in cam]

def get_image_resolution_for_frame(cam_num):
    image_path = os.path.join(PATH_FRAME_DISTORTED, f"cam_{cam_num}.png")
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    height, width, _ = image.shape
    return width, height

def scale_detection_to_original(point2d, img_width, img_height):
    # Assume the resized frame is always 800x800
    resized_size = 800
    
    # Calculate scaling factors from the current image resolution
    scale_x = img_width / resized_size
    scale_y = img_height / resized_size
    
    # Scale the point2d from resized frame to original resolution
    return np.array([point2d[0] * scale_x, point2d[1] * scale_y])

def triangulate(cam1, cam2, point2d1, point2d2):
    # Get image resolution for the current frame and camera
    img_width1, img_height1 = get_image_resolution_for_frame(cam1.camera_number)
    img_width2, img_height2 = get_image_resolution_for_frame(cam2.camera_number)

    # Scale 2D points back to original resolution
    point2d1_scaled = scale_detection_to_original(point2d1, img_width1, img_height1)
    point2d2_scaled = scale_detection_to_original(point2d2, img_width2, img_height2)

    proj1 = get_projection_matrix(cam1)
    proj2 = get_projection_matrix(cam2)

    point2d1_scaled = np.array([point2d1_scaled], dtype=np.float32)
    point2d2_scaled = np.array([point2d2_scaled], dtype=np.float32)

    point4d = cv2.triangulatePoints(proj1, proj2, point2d1_scaled.T, point2d2_scaled.T)
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
        field_corners = np.array(data["field_corners"]) # * 1000
    return positions, field_corners

def main():
    det_3D = {}  # store valid 3D detections for each frame

    # Setup the plot once
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
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

    for i in range(1, MAX_FRAME):
        det_frame_3D = []  # 3D detections for the current frame
        detections_frame = {cam: detections.get((cam, i), []) for cam in VALID_CAMERA_NUMBERS}
        count_valid = 0  # Initialize counter for valid points
        count_not_valid = 0  # Initialize counter for invalid points
        
        # iterate over pairs of cameras
        for idx1, cam1 in enumerate(VALID_CAMERA_NUMBERS):
            for idx2 in range(idx1 + 1, len(VALID_CAMERA_NUMBERS)):
                cam2 = VALID_CAMERA_NUMBERS[idx2]
                
                detections_cam1 = detections_frame[cam1]
                detections_cam2 = detections_frame[cam2]

                if not detections_cam1 or not detections_cam2:
                    continue

                for point2d1 in detections_cam1:
                    for point2d2 in detections_cam2:
                        point3d = triangulate(cam[idx1], cam[idx2], point2d1, point2d2)
                        
                        if is_valid_point(point3d):
                            det_frame_3D.append(point3d)
                            x_coords.append(point3d[0])
                            y_coords.append(point3d[1])
                            z_coords.append(point3d[2])
                            count_valid += 1  # Increment valid count
                        else:
                            count_not_valid += 1  # Increment not valid count

        print(f"\n\nFrame {i} - valid points: {count_valid}, invalid points: {count_not_valid}\n\n")
        
        det_3D[i] = det_frame_3D  # Store valid 3D detections for the frame

        # Update the scatter plot for each frame
        ax.scatter(x_coords, y_coords, z_coords, c='blue', s=50, marker='o')
        
        # Update the plot with the new points
        plt.draw()
        plt.pause(0.001)

        # Ask for user input to proceed to next frame
        input("Press 'n' to continue to the next frame...")
    
    # Save the final data to a file
    save_pickle(det_3D, os.path.join(PATH_DETECTIONS, 'detections_3D.pkl'))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.set_xlim([-15, 15])
    ax.set_ylim([-10, 10])
    ax.set_zlim([np.min(z_coords) - 1, np.max(z_coords) + 1])

    ax.set_title('3D Tracked Points and Real Corners (with Path)')
    ax.legend()

    set_axes_equal(ax)
    plt.show()

if __name__ == "__main__":
    main()
