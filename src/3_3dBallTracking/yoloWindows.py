import cv2
import torch
from random import randint
from ultralytics import YOLO

class yoloWindow:
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
        """_summary_

        Args:
            frame: image Frame

        Returns:
            img: the modified frame with the windows drawn
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

    def createWindow(self, imgSize):
        """_summary_

        Args:
            imgSize (tupla): (width, height of the image)

        Returns:
            list: a list of tuples with the origin of the windows
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
        Checks if two rectangles overlap based on their center coordinates and sizes.

        Args:
            x1_c, y1_c: Center coordinates of the first rectangle.
            w1, h1: Width and height of the first rectangle.
            x2_c, y2_c: Center coordinates of the second rectangle.
            w2, h2: Width and height of the second rectangle.

        Returns:
            bool: True if rectangles overlap, False otherwise.
        """
        l1 = x1_c - w1 / 2
        r1 = x1_c + w1 / 2
        t1 = y1_c - h1 / 2
        b1 = y1_c + h1 / 2

        l2 = x2_c - w2 / 2
        r2 = x2_c + w2 / 2
        t2 = y2_c - h2 / 2
        b2 = y2_c + h2 / 2

        if l1 >= r2 or l2 >= r1:
            return False
        if t1 >= b2 or t2 >= b1:
            return False
        return True

    def detect(self, frame, visualizeBBox=False, visualizeWindows=False, thresholdConfidence=0):
        """_summary_

        Args:
            frame : image frame
            visualizeBBox (bool, optional): flag to draw the bounding boxes. Defaults to False.
            visualizeWindows (bool, optional): flag to draw the windows. Defaults to False.
            thresholdConfidence (int, optional): threshold to consider a detection. Defaults to 0.

        Returns:
            detections, frame: a list of detections and the modified frame
        """
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
        batch = []

        for window in windows:
            img = window["img"]
            batch.append(img)

        results = self.model.predict(batch, verbose=False, device=self.device)

        for win, result in zip(windows, results):
            boxes = result.boxes.xywh.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
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
                            if conf > existing_det[4]:
                                detections[i] = detection
                            overlap_found = True
                            break
                    if not overlap_found:
                        detections.append(detection)
                

        
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
                    (0, 0, 255),
                    2,
                )

        if visualizeWindows:
            frame = self.draWindows(frame)

        if len(detections) == 0:
            return None, frame


        return detections, frame


