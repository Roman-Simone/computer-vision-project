import numpy as np
import cv2
import torch
from ultralytics import YOLO
from config import *
from utils import *
import os

pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')
cameraInfos = load_pickle(PATH_CALIBRATION_MATRIX)

model = YOLO(pathWeight)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')

size = 800
MIN_DISTANCE = 50  # Distanza minima tra le detection in pixel
MAX_BALLS = 3  # Numero massimo di palle da tracciare
CONFIDENCE_THRESHOLD = 0.5
AGE_THRESHOLD = 10  # Numero di frame dopo i quali un tracker viene rimosso se non aggiornato

class Ball:
    def __init__(self, position):
        self.position = np.array(position)
        self.velocity = np.zeros(2)
        self.age = 0
        self.history = [position]

    def predict(self):
        self.position += self.velocity
        self.age += 1
        self.position = np.clip(self.position, 0, size - 1)  # Mantieni la posizione all'interno del frame

    def update(self, new_position):
        new_position = np.array(new_position)
        self.velocity = new_position - self.position
        self.position = new_position
        self.age = 0
        self.history.append(tuple(new_position))
        if len(self.history) > 20:  # Mantieni solo gli ultimi 20 punti della traiettoria
            self.history.pop(0)

class MultiballTracker:
    def __init__(self):
        self.balls = []

    def update(self, detections):
        # Predizione
        for ball in self.balls:
            ball.predict()

        # Associazione
        if detections:
            cost_matrix = np.array([[np.linalg.norm(ball.position - det) for det in detections] for ball in self.balls])
            assigned_balls, assigned_detections = self._hungarian_assignment(cost_matrix)

            # Aggiornamento dei ball assegnati
            for ball_idx, det_idx in zip(assigned_balls, assigned_detections):
                if cost_matrix[ball_idx, det_idx] < MIN_DISTANCE:
                    self.balls[ball_idx].update(detections[det_idx])
                else:
                    assigned_balls = np.delete(assigned_balls, np.where(assigned_balls == ball_idx))
                    assigned_detections = np.delete(assigned_detections, np.where(assigned_detections == det_idx))

            # Creazione di nuovi ball per le detection non assegnate
            unassigned_detections = set(range(len(detections))) - set(assigned_detections)
            for det_idx in unassigned_detections:
                if len(self.balls) < MAX_BALLS:
                    self.balls.append(Ball(detections[det_idx]))

        # Rimozione dei ball non aggiornati
        self.balls = [ball for ball in self.balls if ball.age < AGE_THRESHOLD]

    def _hungarian_assignment(self, cost_matrix):
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost_matrix)
        return rows, cols

    def draw(self, frame):
        for ball in self.balls:
            # Disegna la posizione stimata
            cv2.circle(frame, tuple(ball.position.astype(int)), 5, (0, 0, 255), -1)
            
            # Disegna la traiettoria
            if len(ball.history) > 1:
                for i in range(1, len(ball.history)):
                    cv2.line(frame, ball.history[i-1], ball.history[i], (0, 255, 0), 2)

def applyModel(frame, model, tracker):
    results = model.track(frame, save=True, verbose=False, device=device)
    
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]

        if class_id == 0 and confidence > CONFIDENCE_THRESHOLD:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            center = (int(x_center), int(y_center))
            detections.append(center)
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

    tracker.update(detections)
    tracker.draw(frame)

    return frame, detections

def testModel(num_cam):
    pathVideo = os.path.join(PATH_VIDEOS, f'out{num_cam}.mp4')
    cameraInfo, _ = take_info_camera(num_cam, cameraInfos)

    videoCapture = cv2.VideoCapture(pathVideo)
    tracker = MultiballTracker()

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break

        frameUndistorted = undistorted(frame, cameraInfo)
        frameUndistorted = cv2.resize(frameUndistorted, (size, size))
        frameWithBbox, detections = applyModel(frameUndistorted, model, tracker)
        cv2.imshow('Frame', frameWithBbox)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    testModel(6)