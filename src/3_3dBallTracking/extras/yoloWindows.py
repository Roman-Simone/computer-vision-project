import cv2
import torch
from random import randint
from ultralytics import YOLO

class yoloWindows:
    """
    A class for managing sliding windows over an image, detecting objects within each window using a YOLO model,
    and visualizing the results.
    
    Parameters:
        pathWeight (str): path to the YOLO model weights.
        windowSize (tuple)p: size of each sliding window as (width, height).
        overlap (tuple): overlap percentage between consecutive windows along x and y axes.
    """

    def __init__(self, pathWeight="", windowSize=(640, 640), overlap=(0,0)):
        self.windowSize = windowSize
        self.overlap = overlap
        self.model = YOLO(pathWeight)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

    def draWindows(self, frame):
        """
        Draws the sliding windows as rectangles on the given image frame.
        
        Parameters:
            frame (np.ndarray): image frame on which to draw the windows.

        Returns:
            np.ndarray: frame with windows drawn as rectangles.
        """
        imgSize = frame.shape[:2][::-1]  
        origins = self.createWindow(imgSize)  

        for origin in origins:
            x, y = origin
            cv2.rectangle(
                frame,
                (x, y),
                (x + self.windowSize[0], y + self.windowSize[1]),
                (randint(0, 100), randint(0, 100), 0),
                15,
            )
        return frame

    def is_in_any_region(self, x_center, y_center, regions):
        """
        Checks if a given point (center of a bounding box) lies within any defined regions to ignore.
        
        Parameters:
            x_center (float): x-coordinate of the point.
            y_center (float): y-coordinate of the point.
            regions (list): list of regions defined by tuples (x, y, width, height).
            
        Returns:
            bool: True if the point is in any region; otherwise, False.
        """
        for region in regions:
            x, y, w, h = region
            if x <= x_center <= x + w and y <= y_center <= y + h:
                return True
        return False

    def createWindow(self, imgSize):
        """
        Calculates the origins of sliding windows across the given image size, based on the specified window size and overlap.

        Parameters:
            imgSize (tuple): size of the image as (width, height).

        Returns:
            list: list of tuples representing the top-left origin (x, y) of each window.
        """
        xOverlay = int(self.windowSize[0] * self.overlap[0])
        yOverlay = int(self.windowSize[1] * self.overlap[1])

        nX = (imgSize[0] - self.windowSize[0]) // (self.windowSize[0] - xOverlay) + 1
        nY = (imgSize[1] - self.windowSize[1]) // (self.windowSize[1] - yOverlay) + 1

        padX = (imgSize[0] - self.windowSize[0]) % (self.windowSize[0] - xOverlay)
        padY = (imgSize[1] - self.windowSize[1]) % (self.windowSize[1] - yOverlay)

        windowsOrigins = []
        for i in range(nX):
            for j in range(nY):
                x = i * (self.windowSize[0] - xOverlay) + padX // 2
                y = j * (self.windowSize[1] - yOverlay) + padY // 2
                windowsOrigins.append((x, y))

        return windowsOrigins

    def isRectangleOverlap(self, x1_c, y1_c, w1, h1, x2_c, y2_c, w2, h2):
        """
        Checks if two rectangles overlap based on their center coordinates and dimensions.

        Parameters:
            x1_c, y1_c (float): center coordinates of the first rectangle.
            w1, h1 (float): width and height of the first rectangle.
            x2_c, y2_c (float): center coordinates of the second rectangle.
            w2, h2 (float): width and height of the second rectangle.

        Returns:
            bool: True if the rectangles overlap, False otherwise.
        """
        l1, r1 = x1_c - w1 / 2, x1_c + w1 / 2
        t1, b1 = y1_c - h1 / 2, y1_c + h1 / 2
        l2, r2 = x2_c - w2 / 2, x2_c + w2 / 2
        t2, b2 = y2_c - h2 / 2, y2_c + h2 / 2

        if l1 >= r2 or l2 >= r1:
            return False
        if t1 >= b2 or t2 >= b1:
            return False
        return True

    def detect(self, frame, visualizeBBox=False, visualizeWindows=False, thresholdConfidence=0, regions=[]):
        """
        Detects objects within each sliding window using the YOLO model and returns detections above the confidence threshold.
        
        Parameters:
            frame (np.ndarray):  image frame to process.
            visualizeBBox (bool): if True, draw bounding boxes around detections. Defaults to False.
            visualizeWindows (bool): if True, draw the sliding windows on the frame. Defaults to False.
            thresholdConfidence (float): minimum confidence score to consider a detection.
            regions (list): regions to ignore, defined as [(x, y, w, h), ...].

        Returns:
            tuple: list of detected bounding boxes [(x, y, w, h, confidence), ...] and the modified frame.
        """
        original_size = frame.shape[:2][::-1]  # (width, height)
        imgSize = frame.shape[:2][::-1]

        windowsOrigins = self.createWindow(imgSize)
        windows = []

        for origin in windowsOrigins:
            x, y = origin
            window = {
                "img": frame[
                    y: y + self.windowSize[1],
                    x: x + self.windowSize[0],
                ],
                "coordinates": (x, y, self.windowSize[0], self.windowSize[1]),
            }
            windows.append(window)
        
        detections = []
        batch = [window["img"] for window in windows]

        # Run YOLO model on batch of window images
        results = self.model.predict(batch, verbose=False, device=self.device)

        # Process YOLO detections for each window
        for win, result in zip(windows, results):
            boxes = result.boxes.xywh.cpu().numpy()  # Bounding boxes in [x_center, y_center, width, height] format
            confs = result.boxes.conf.cpu().numpy()  # Confidence scores
            real_x, real_y, _, _ = win["coordinates"]

            for box, conf in zip(boxes, confs):
                if conf >= thresholdConfidence:
                    x_center, y_center, w, h = box
                    x_center += real_x
                    y_center += real_y
                    detection = (x_center, y_center, w, h, conf)

                    # Check for overlaps and keep the detection with higher confidence
                    overlap_found = False
                    for i, existing_det in enumerate(detections):
                        if self.isRectangleOverlap(
                            x_center, y_center, w, h,
                            existing_det[0], existing_det[1], existing_det[2], existing_det[3]
                        ):
                            if conf > existing_det[4]:  # Update detection if confidence is higher
                                detections[i] = detection
                            overlap_found = True
                            break
                    if not overlap_found:
                        detections.append(detection)

        if regions:
            detections = [
                detection for detection in detections 
                if not self.is_in_any_region(detection[0], detection[1], regions)
            ]
        
        if visualizeBBox:
            for detection in detections:
                x_center, y_center, w, h, conf = detection
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        if visualizeWindows:
            frame = self.draWindows(frame)

        frame = cv2.resize(frame, original_size)
        scale_x = original_size[0] / imgSize[0]
        scale_y = original_size[1] / imgSize[1]

        resized_detections = [
            (x_center * scale_x, y_center * scale_y, w * scale_x, h * scale_y, conf)
            for x_center, y_center, w, h, conf in detections
        ]

        if not resized_detections:
            return None, frame

        return resized_detections, frame
