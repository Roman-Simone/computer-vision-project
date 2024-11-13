class CameraInfo:
    """
    A class to represent the information and calibration data of a camera.
    """
    
    def __init__(self, camera_number):
        self.camera_number = camera_number  
        self.chessboard_size = None         # Size of the chessboard used for calibration
        self.objpoints = []                 # 3D points in real world space
        self.imgpoints = []                 # 2D points in image plane
        self.mtx = None                     # Camera matrix
        self.newcameramtx = None            # New camera matrix after undistortion
        self.dist = None                    # Distortion coefficients
        self.roi = None                     
        self.extrinsic_matrix = None        # Extrinsic parameters (rotation and translation vectors)
        
    def __str__(self):
        return f"Camera number: {self.camera_number}"


class HomographyInfo:
    """
    A class to represent the homography information between two cameras.
    """
    
    def __init__(self, camera_number_1, camera_number_2):
        self.camera_number_1 = camera_number_1  
        self.camera_number_2 = camera_number_2  
        self.homography = None                  # Homography matrix between the two cameras
    
    def __str__(self):
        return f"Camera 1: {self.camera_number_1}, Camera 2: {self.camera_number_2}"