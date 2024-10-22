import cv2 as cv
import torch

from random import randint
from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path="", wsize=(640, 640), overlap=(0, 0)):

        self.model_path = model_path
        self.wsize = wsize
        self.overlap = overlap
        self.model = YOLO(model_path)

    def print_windows(self, frame):
        imsize = frame.shape[:2][::-1]
        origins = self.get_windows(imsize)
        for ori in origins:
            x, y = ori
            cv.rectangle(
                frame,
                (x, y),
                (x + self.wsize[0], y + self.wsize[1]),
                (randint(100, 200), randint(100, 200), 0),
                15,
            )
        return frame

    def get_windows(self, imsize):
        window_size = self.wsize

        ox_size = int(window_size[0] * self.overlap[0])
        oy_size = int(window_size[1] * self.overlap[1])

        nx = (imsize[0] - window_size[0]) // (window_size[0] - ox_size) + 1
        ny = (imsize[1] - window_size[1]) // (window_size[1] - oy_size) + 1

        xpad = (imsize[0] - window_size[0]) % (window_size[0] - ox_size)
        ypad = (imsize[1] - window_size[1]) % (window_size[1] - oy_size)

        origins = []
        for i in range(nx):
            for j in range(ny):
                x = i * (window_size[0] - ox_size) + xpad // 2
                y = j * (window_size[1] - oy_size) + ypad // 2
                origins.append((x, y))

        # breakpoint()
        return origins

    def predict(self, frame, scale=1.0, viz=False, winz=False):

        if scale != 1.0:
            frame = cv.resize(
                frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            )

        imsize = frame.shape[:2][::-1]
        origins = self.get_windows(imsize)

        windows = []
        for origin in origins:
            x, y = origin
            window = {
                "image": frame[
                    y : y + self.wsize[1],
                    x : x + self.wsize[0],
                ],
                "coo": (x, y, self.wsize[0], self.wsize[1]),
            }
            windows.append(window)

        detections = []

        batch = []

        # print(f"Inferencing on {len(windows)} windows")

        for win in windows:
            img = win["image"]
            batch.append(img)

        results = self.model.predict(batch, verbose=False)

        for win, result in zip(windows, results):
            boxes = result.boxes.xywh.cpu().tolist()
            real_x, real_y, _, _ = win["coo"]
            for idx, box in enumerate(boxes):
                conf = result.boxes.conf.tolist()[idx]
                x, y, w, h = map(int, box)

                if scale != 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)

                detections.append((x + real_x, y + real_y, w, h, conf))

        # TODO: merge adjacent detections
        # vec1 = torch.tensor(detections).T[:2].T.unsqueeze(0)
        # vec2 = torch.tensor(detections).T[:2].T.unsqueeze(1)
        # distances = torch.norm(vec1 - vec2, dim=2)

        if winz:
            frame = self.print_windows(frame)

        if len(detections) == 0:
            return None, None, frame
        else:

            out = detections[torch.argmax(torch.tensor(detections).T[-1]).item()]

            if viz:
                x, y, w, h, c = out
                frame = cv.rectangle(
                    frame,
                    (x - w // 2, y - h // 2),
                    (x + w // 2, y + h // 2),
                    (0, 255, 0),
                    8,
                )
            return out, detections, frame
