import os

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.join(current_path, os.pardir)
parent_path = os.path.abspath(parent_path)

path_videos_calibration = os.path.join(parent_path, 'data/dataset/calibration')
path_videos = os.path.join(parent_path, 'data/dataset/video')
path_calibration_matrix = os.path.join(parent_path, 'data/calibrationMatrix/calibration.pkl')
path_json = os.path.join(parent_path, 'data/world_points_all_cameras.json')
path_frames = os.path.join(parent_path, 'data/dataset/singleFrame')
path_court = os.path.join(parent_path, 'data/images/courts.jpg')