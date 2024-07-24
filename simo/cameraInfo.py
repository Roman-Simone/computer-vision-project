class CameraInfo:
    def __init__(self, camera_number):
        self.camera_number = camera_number
        self.chessboard_size = None
        self.objpoints = []
        self.imgpoints = []
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None