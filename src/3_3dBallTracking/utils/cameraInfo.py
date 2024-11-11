class CameraInfo:
    def __init__(self, camera_number):
        # Initialize the CameraInfo object with the given camera number
        self.camera_number = camera_number 
        self.chessboard_size = None     # Size of the chessboard used for calibration
        self.objpoints = []             # 3D points in real world space
        self.imgpoints = []             # 2D points in image plane
        self.mtx = None                 # Camera matrix
        self.newcameramtx = None        # New camera matrix after undistortion
        self.dist = None                # Distortion coefficients
        self.roi = None                 # Region of interest
        self.extrinsic_matrix = None    # Extrinsic parameters

    def __str__(self):
        # Return a string representation of the CameraInfo object
        return f"Camera number: {self.camera_number}"


class HomographyInfo:
    def __init__(self, camera_number_1, camera_number_2):
        # Initialize the HomographyInfo object with the given camera numbers
        self.camera_number_1 = camera_number_1
        self.camera_number_2 = camera_number_2
        self.homography = None          # Homography matrix between the two cameras
    
    def __str__(self):
        # Return a string representation of the HomographyInfo object
        return f"Camera 1: {self.camera_number_1}, Camera 2: {self.camera_number_2}"