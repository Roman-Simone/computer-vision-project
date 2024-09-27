class CameraInfo:
    def __init__(self, camera_number):
        self.camera_number = camera_number
        self.chessboard_size = None
        self.objpoints = []
        self.imgpoints = []
        self.mtx = None
        self.newcameramtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.roi = None
        self.extrinsic_matrix = None
        self.inverse_rotation_matrix = None
        self.inverse_translation_vector = None

class InterCameraInfo:
    def __init__(self, cameras_number):
        self.cameras_number = cameras_number
        self.fundamental_matrix = None
        self.corners_cam1 = []  #order: top_left, top_right, bottom_left, bottom_right
        self.corners_cam2 = []  #order: top_left, top_right, bottom_left, bottom_right
