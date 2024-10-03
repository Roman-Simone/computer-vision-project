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

    def __str__(self):
        return f"Camera number: {self.camera_number}"


class InterCameraInfo:
    def __init__(self, camera_number_1, camera_number_2):
        self.camera_number_1 = camera_number_1
        self.camera_number_2 = camera_number_2
        self.homography = None
    
    def __str__(self):
        return f"Camera 1: {self.camera_number_1}, Camera 2: {self.camera_number_2}"