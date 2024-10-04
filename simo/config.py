import os

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)

PATH_VIDEOS_CALIBRATION = os.path.join(parent_path, 'data/videos/calibration')
PATH_VIDEOS = os.path.join(parent_path, 'data/videos/video')
PATH_CALIBRATION_MATRIX = os.path.join(parent_path, 'data/calibrationMatrix/calibration.pkl')
PATH_HOMOGRAPHY_MATRIX = os.path.join(parent_path, 'data/homographyMatrix/homography.pkl')
PATH_JSON_DISTORTED = os.path.join(parent_path, 'data/json/world_points_all_cameras_distorted.json')
PATH_JSON_UNDISTORTED = os.path.join(parent_path, 'data/json/world_points_all_cameras_undistorted.json')
PATH_FRAME_DISTORTED = os.path.join(parent_path, 'data/images/distorted')
PATH_FRAME_UNDISTORTED = os.path.join(parent_path, 'data/images/undistorted')
PATH_COURT = os.path.join(parent_path, 'data/images/courts.jpg')
PATH_DATASET = os.path.join(parent_path, 'data/dataset')

VALID_CAMERA_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]