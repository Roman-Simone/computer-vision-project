import numpy as np
# import torch
import cv2 
from utils import *

class CameraInfo:
    def __init__(self, camera_number):
        self.camera_number = camera_number  # Camera number
        self.chessboard_size = None     # Chessboard size
        self.objpoints = []     
        self.imgpoints = []     
        self.mtx = None
        self.newcameramtx = None
        self.dist = None
        self.roi = None
        self.extrinsic_matrix = None

    # def get_projection_matrix(self):
    #     """
    #     Get the projection matrix of the camera

    #     Returns:
    #     - np.array: Projection matrix
    #     """

    #     rot_mtx, _ = cv2.Rodrigues(self.rvecs)

    #     return np.dot(self.newcameramtx, np.hstack((rot_mtx, self.tvecs)))

    def get_projection_matrix(self):
        """
        Gets the projection matrix for a camera.

        Args:
            camera_number (int): Camera number.
            cameraInfos (dict): Dictionary containing the camera information.
        
        Returns:
            np.array: Projection matrix for the camera.
        """

        K = self.newcameramtx  # 3x3 
        
        # print("K: ", K)
        
        # (4x4)
        extrinsic_matrix = self.extrinsic_matrix  
        
        # print("Extrinsic matrix: ", extrinsic_matrix)
        
        # get the top 3x4 part (first 3 rows and 4 columns)
        extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]  
        
        # print("K: ", K)
        # print("Extrinsic matrix (3x4): ", extrinsic_matrix_3x4)
        
        # return projection matrix P = K * [R | t]    
        return np.dot(K, extrinsic_matrix_3x4)

    def triangulate(self, cam2, point2d1, point2d2):
            proj1 = self.get_projection_matrix()
            proj2 = cam2.get_projection_matrix()

            point2d1 = np.array([point2d1], dtype=np.float32)
            point2d2 = np.array([point2d2], dtype=np.float32)

            point4d = cv2.triangulatePoints(proj1, proj2, point2d1.T, point2d2.T)
            point3d = cv2.convertPointsFromHomogeneous(point4d.T)[0][0]
            return point3d

    def detections_to_point(all_dets, camera_infos, prev_est=None):
        # Ensure there are at least two camera detections to triangulate
        if len(all_dets.keys()) < 2:
            return None

        triangulated_points = []  # Store triangulated 3D points
        checked = {}  # To avoid re-checking pairs of cameras

        # Iterate over all pairs of camera detections
        for cam_idx_1, det1 in all_dets.items():
            for cam_idx_2, det2 in all_dets.items():
                if cam_idx_1 == cam_idx_2:
                    continue  # Skip comparison with the same camera
                if (cam_idx_2, cam_idx_1) in checked:
                    continue  # Skip if this camera pair has already been checked
                
                # Mark this camera pair as checked
                checked[(cam_idx_1, cam_idx_2)] = True

                # Get the camera information for both cameras
                cam1, _ = take_info_camera(cam_idx_1, camera_infos)
                cam2, _ = take_info_camera(cam_idx_2, camera_infos)

                # Triangulate the 3D point using detections from both cameras
                point3d = cam1.triangulate(cam2, det1, det2)
                triangulated_points.append(point3d)

        # Convert the list of triangulated points to a NumPy array
        triangulated_points_np = np.array(triangulated_points)

        # If there is no previous estimate, compute the mean of valid triangulated points
        if prev_est is None:
            if len(triangulated_points_np) == 0:
                return None  # Return None if there are no valid triangulated points

            # Step 1: Compute pairwise distances between triangulated points
            vec1 = torch.tensor(triangulated_points_np).unsqueeze(0)  # Shape (1, N, 3)
            vec2 = torch.tensor(triangulated_points_np).unsqueeze(1)  # Shape (N, 1, 3)
            distances = torch.norm(vec1 - vec2, dim=2)  # Compute pairwise distances

            # Step 2: Filter out points that are too far from each other
            good_points = []
            for i in range(distances.shape[0]):
                for j in range(i + 1, distances.shape[1]):  # Compare distinct pairs
                    if distances[i][j] < 1000:  # Threshold distance (e.g., 1000 units)
                        if i not in good_points:
                            good_points.append(i)
                        if j not in good_points:
                            good_points.append(j)

            # Step 3: Check if any points survived the filtering
            if len(good_points) == 0:
                return None  # Return None if no good points were found

            # Step 4: Compute the mean of the filtered points
            good_points = np.array([triangulated_points_np[i] for i in good_points])
            final_point = np.mean(good_points, axis=0)  # Compute the average of good points

        else:
            # If a previous estimate exists, choose the triangulated point closest to it
            dst = [np.linalg.norm(vec - prev_est) for vec in triangulated_points_np]  # Compute distances to previous estimate
            closest = np.array(dst).argmin()  # Find the closest point index
            final_point = triangulated_points_np[closest]  # Select the closest point

            # If the closest point is too far from the previous estimate, discard it
            if dst[closest] > 1000:  # Threshold distance from previous estimate
                final_point = None

        return final_point  # Return the final selected or averaged point


    def __str__(self):
        return f"Camera number: {self.camera_number}"


class HomographyInfo:
    def __init__(self, camera_number_1, camera_number_2):
        self.camera_number_1 = camera_number_1
        self.camera_number_2 = camera_number_2
        self.homography = None
    
    def __str__(self):
        return f"Camera 1: {self.camera_number_1}, Camera 2: {self.camera_number_2}"