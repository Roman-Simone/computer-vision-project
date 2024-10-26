import re
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# Add the parent directory to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(parent_path)

from utils.utils import *
from utils.config import *

@dataclass
class CameraInfo:
    camera_number: int
    chessboard_size: Tuple[int, int] = (6, 9)
    mtx: Optional[np.ndarray] = None  # Camera matrix
    dist: Optional[np.ndarray] = None  # Distortion coefficients
    newcameramtx: Optional[np.ndarray] = None  # Optimized camera matrix
    roi: Optional[Tuple] = None  # Region of interest
    rvecs: Optional[List] = None  # Rotation vectors
    tvecs: Optional[List] = None  # Translation vectors
    objpoints: List = None  # 3D points in real world space
    imgpoints: List = None  # 2D points in image plane
    
    def __post_init__(self):
        self.objpoints = []
        self.imgpoints = []

class CameraCalibrator:
    def __init__(self):
        # Configuration for all cameras' chessboard sizes
        self.all_chessboard_sizes = {
            1: (5, 7), 2: (5, 7), 3: (5, 7), 4: (5, 7),
            5: (6, 9), 6: (6, 9), 7: (5, 7), 8: (6, 9),
            12: (5, 7), 13: (5, 7)
        }
        self.SKIP_FRAME = 5
        
    def find_chessboard_points(self, video_path: str, camera_info: CameraInfo, debug: bool = True) -> Tuple[List, List, np.ndarray]:
        """Find chessboard corners in video frames."""
        chess_width, chess_height = camera_info.chessboard_size
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = np.zeros((chess_width * chess_height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_height, 0:chess_width].T.reshape(-1, 2)
        
        # Initialize lists
        objpoints, imgpoints = [], []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup debug directory
        if debug:
            output_dir = os.path.join(os.path.dirname(__file__), f"samples/Camera{camera_info.camera_number}")
            os.makedirs(output_dir, exist_ok=True)
            [os.remove(os.path.join(output_dir, f)) for f in os.listdir(output_dir)]
            print(f"Saving debug frames to {output_dir}")
        
        # Process frames
        frame_count = successful_detections = processed_frames = 0
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                frame_count += 1
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.SKIP_FRAME != 0:
                    pbar.update(1)
                    continue
                
                processed_frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (chess_height, chess_width), None)
                
                if ret:
                    successful_detections += 1
                    objpoints.append(objp)
                    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(refined_corners)
                    
                    if debug:
                        debug_frame = frame.copy()
                        cv2.drawChessboardCorners(debug_frame, (chess_height, chess_width), refined_corners, ret)
                        cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", debug_frame)
                
                pbar.update(self.SKIP_FRAME)
        
        # Print statistics
        print(f"\nCalibration Statistics:")
        print(f"Total frames: {total_frames}")
        print(f"Processed frames: {processed_frames}")
        print(f"Successful detections: {successful_detections}")
        print(f"Detection rate: {(successful_detections/processed_frames)*100:.2f}%")
        
        if successful_detections < 10:
            print("\nWARNING: Insufficient successful detections (<10).")
            print("Consider recording new calibration video with:")
            print("- Better lighting conditions")
            print("- More varied chessboard positions")
            print("- Slower movement")
        
        cap.release()
        return objpoints, imgpoints, gray
    
    def calculate_reprojection_error(self, camera_info: CameraInfo) -> Tuple[float, float, List[float]]:
        """Calculate reprojection error metrics."""
        errors = []
        total_error = 0
        max_error = 0
        
        for i in range(len(camera_info.objpoints)):
            projected_points, _ = cv2.projectPoints(
                camera_info.objpoints[i], 
                camera_info.rvecs[i], 
                camera_info.tvecs[i], 
                camera_info.mtx, 
                camera_info.dist
            )
            error = cv2.norm(camera_info.imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
            errors.append(error)
            total_error += error
            max_error = max(max_error, error)
        
        mean_error = total_error / len(camera_info.objpoints)
        return mean_error, max_error, errors
    
    def plot_reprojection_errors(self, errors: List[float], camera_number: int):
        """Plot histogram of reprojection errors."""
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=20, edgecolor='black')
        plt.title(f'Reprojection Error Distribution - Camera {camera_number}')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'reprojection_error_cam{camera_number}.png')
        plt.close()
    
    def calibrate_single_camera(self, camera_number: int) -> CameraInfo:
        """Calibrate a single camera."""
        # Initialize camera info
        camera_info = CameraInfo(
            camera_number=camera_number,
            chessboard_size=self.all_chessboard_sizes[camera_number]
        )
        
        # Find calibration video
        videos = find_files(PATH_VIDEOS_CALIBRATION)
        video_path = None
        for video in videos:
            if str(camera_number) in video:
                video_path = os.path.join(PATH_VIDEOS_CALIBRATION, video)
                break
        
        if not video_path:
            raise FileNotFoundError(f"No calibration video found for camera {camera_number}")
        
        print(f"\nCalibrating Camera {camera_number}")
        print(f"Using video: {video_path}")
        print(f"Chessboard size: {camera_info.chessboard_size}")
        
        # Find chessboard points
        camera_info.objpoints, camera_info.imgpoints, gray = self.find_chessboard_points(
            video_path, camera_info, debug=True
        )
        
        if len(camera_info.objpoints) < 10:
            print(f"ERROR: Insufficient points for camera {camera_number}")
            return camera_info
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            camera_info.objpoints,
            camera_info.imgpoints,
            gray.shape[::-1],
            None,
            None
        )
        
        if not ret:
            print(f"ERROR: Calibration failed for camera {camera_number}")
            return camera_info
        
        # Store calibration results
        camera_info.mtx = mtx
        camera_info.dist = dist
        camera_info.rvecs = rvecs
        camera_info.tvecs = tvecs
        
        # Calculate optimal camera matrix
        h, w = gray.shape[:2]
        camera_info.newcameramtx, camera_info.roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h)
        )
        
        # Calculate and display reprojection error
        mean_error, max_error, errors = self.calculate_reprojection_error(camera_info)
        print("\nCalibration Quality:")
        print(f"Mean reprojection error: {mean_error:.6f} pixels")
        print(f"Max reprojection error: {max_error:.6f} pixels")
        
        # Plot error distribution
        self.plot_reprojection_errors(errors, camera_number)
        
        if mean_error > 1.0:
            print("\nWARNING: High reprojection error detected!")
            print("Consider recalibrating with better conditions.")
        
        return camera_info
    
    def calibrate_all_cameras(self):
        """Calibrate all cameras and save results."""
        cameras_info = []
        for camera_number in self.all_chessboard_sizes.keys():
            try:
                camera_info = self.calibrate_single_camera(camera_number)
                cameras_info.append(camera_info)
            except Exception as e:
                print(f"Error calibrating camera {camera_number}: {str(e)}")
        
        save_pickle(cameras_info, PATH_CALIBRATION_MATRIX)
        print(f"\nCalibration data saved to {PATH_CALIBRATION_MATRIX}")
    
    def test_calibration(self):
        """Test calibration results on videos."""
        cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)
        videos = find_files(PATH_VIDEOS)
        videos.sort()
        
        for video in videos:
            camera_number = int(re.findall(r'\d+', video.replace(".mp4", ""))[0])
            if camera_number not in VALID_CAMERA_NUMBERS:
                continue
            
            camera_info = next((cam for cam in cameras_info if cam.camera_number == camera_number), None)
            if not camera_info:
                continue
            
            cap = cv2.VideoCapture(os.path.join(PATH_VIDEOS, video))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                undistorted = cv2.undistort(
                    frame, 
                    camera_info.mtx, 
                    camera_info.dist, 
                    None,
                    camera_info.newcameramtx
                )
                
                # Show comparison
                undistorted = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))
                comparison = np.hstack((frame, undistorted))
                comparison = cv2.resize(comparison, (comparison.shape[1]//2, comparison.shape[0]//2))
                
                cv2.imshow('Original (Left) vs Undistorted (Right)', comparison)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
        
        cv2.destroyAllWindows()

def main():
    calibrator = CameraCalibrator()
    
    # Uncomment the desired operation:
    
    # Calibrate all cameras
    # calibrator.calibrate_all_cameras()
    
    # Calibrate single camera
    camera_number = 6
    camera_info = calibrator.calibrate_single_camera(camera_number)
    cameras_info = load_pickle(PATH_CALIBRATION_MATRIX)
    
    # Update or append the new calibration
    updated = False
    for i, cam in enumerate(cameras_info):
        if cam.camera_number == camera_number:
            cameras_info[i] = camera_info
            updated = True
            break
    if not updated:
        cameras_info.append(camera_info)
    
    save_pickle(cameras_info, PATH_CALIBRATION_MATRIX)
    
    # Test calibration
    calibrator.test_calibration()

if __name__ == '__main__':
    main()