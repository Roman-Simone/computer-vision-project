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
            boxes = result.boxes.xywh.cpu().tolist()
            real_x, real_y, _, _ = win["coordinates"]
            for idx, box in enumerate(boxes):
                conf = result.boxes.conf.tolist()[idx]
                x, y, w, h = map(int, box)

                x, y = x + real_x, y + real_y,

                

                if visualizeBBox:
                    if conf > thresholdConfidence:
                        detections.append((x, y, w, h, conf))
                        frame = cv2.rectangle(
                            frame,
                            (x - w // 2, y - h // 2),
                            (x + w // 2, y + h // 2),
                            (0, 255, 0),
                            4,
                        )

        if visualizeWindows:
            frame = self.draWindows(frame)

        if len(detections) == 0:
            return None, frame


        return detections, frame


