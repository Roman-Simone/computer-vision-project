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


class InterCameraInfo:
    def __init__(self, camera_number_1, camera_number_2):
        self.camera_number_1 = camera_number_1
        self.camera_number_2 = camera_number_2
        self.homography = None
        self.h1 = None
        self.h2 = None