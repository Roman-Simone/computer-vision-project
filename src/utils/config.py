import os

current_path = os.path.dirname(os.path.abspath(__file__))
grandparent_path = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

PATH_FRAME_SAMPLES_CALIBRATION = os.path.join(grandparent_path, 'data/images/samples_calibration')
PATH_VIDEOS_CALIBRATION = os.path.join(grandparent_path, 'data/videos/calibration')
PATH_VIDEOS = os.path.join(grandparent_path, 'data/videos/video')
PATH_CALIBRATION_MATRIX = os.path.join(grandparent_path, 'data/calibrationMatrix/calibration.pkl')
PATH_HOMOGRAPHY_MATRIX = os.path.join(grandparent_path, 'data/homographyMatrix/homography.pkl')
PATH_JSON_DISTORTED = os.path.join(grandparent_path, 'data/json/world_points_all_cameras_distorted.json')
PATH_JSON_UNDISTORTED = os.path.join(grandparent_path, 'data/json/world_points_all_cameras_undistorted.json')
PATH_FRAME_DISTORTED = os.path.join(grandparent_path, 'data/images/distorted')
PATH_FRAME_UNDISTORTED = os.path.join(grandparent_path, 'data/images/undistorted')
PATH_COURT = os.path.join(grandparent_path, 'data/images/courts.jpg')
PATH_DATASET = os.path.join(grandparent_path, 'data/dataset')
PATH_STATIC = os.path.join(grandparent_path, 'src/app/static')
PATH_CSS = os.path.join(grandparent_path, 'src/app/static/css')
PATH_DETECTIONS = os.path.join(grandparent_path, 'data/detections')
PATH_DETECTIONS_WINDOW_04 = os.path.join(grandparent_path, 'data/detections/detections[windows]')
PATH_DETECTIONS_04 = os.path.join(grandparent_path, 'data/detections/detections')
PATH_3D_DETECTIONS_04 = os.path.join(grandparent_path, 'data/detections/detections/3D_detections')
PATH_WEIGHT = os.path.join(grandparent_path, 'data/weights')
PATH_CAMERA_POS = os.path.join(grandparent_path, 'data/camera_positions.json')

VALID_CAMERA_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

YOLO_INPUT_SIZE = 800  # size for YOLO model input

ACTIONS = {
    1: (48, 230),
    2: (1050, 1230),
    3: (1850, 2060),
    4: (2620, 2790),
    5: (3770, 3990)
}



