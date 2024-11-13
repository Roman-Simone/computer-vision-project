class CameraInfo:
    """
    A class to represent the information and calibration data of a camera.

    Attributes
    ----------
    camera_number : int
        The identifier number of the camera.
    chessboard_size : tuple or None
        The size of the chessboard used for calibration (rows, columns).
    objpoints : list
        The list of 3D points in real world space.
    imgpoints : list
        The list of 2D points in image plane.
    mtx : numpy.ndarray or None
        The camera matrix.
    newcameramtx : numpy.ndarray or None
        The new camera matrix after undistortion.
    dist : numpy.ndarray or None
        The distortion coefficients.
    roi : tuple or None
        The region of interest.
    extrinsic_matrix : numpy.ndarray or None
        The extrinsic parameters.

    Methods
    -------
    __str__():
        Returns a string representation of the CameraInfo object.
    """
    def __init__(self, camera_number):
        self.camera_number = camera_number 
        self.chessboard_size = None     
        self.objpoints = []             
        self.imgpoints = []             
        self.mtx = None                 
        self.newcameramtx = None        
        self.dist = None
        self.roi = None                 
        self.extrinsic_matrix = None    

    def __str__(self):
        return f"Camera number: {self.camera_number}"


class HomographyInfo:
    """
    A class to represent the homography information between two cameras.
    Attributes
    ----------
    camera_number_1 : int
        The number of the first camera.
    camera_number_2 : int
        The number of the second camera.
    homography : numpy.ndarray or None
        The homography matrix between the two cameras. Initialized to None.
    Methods
    -------
    __str__():
        Returns a string representation of the HomographyInfo object.
    """
    
    def __init__(self, camera_number_1, camera_number_2):
        self.camerqa_number_1 = camera_number_1
        self.camera_number_2 = camera_number_2
        self.homography = None
    
    def __str__(self):
        return f"Camera 1: {self.camera_number_1}, Camera 2: {self.camera_number_2}"