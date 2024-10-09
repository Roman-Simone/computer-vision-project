import threading
import torch
import cv2
import os
from config import *
from ultralytics import YOLO

# Define model names and video sources
pathWeight = os.path.join(PATH_WEIGHT, 'best_v11_800.pt')

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
    print('MPS is available')
else:
    device = 'cpu'


size = 800
model = YOLO(pathWeight)  
SOURCES = ["/Users/simoneroman/Desktop/CV/Computer_Vision_project/data/videos/video/out1.mp4", "/Users/simoneroman/Desktop/CV/Computer_Vision_project/data/videos/video/out2.mp4"]  # local video, 0 for webcam


def run_tracker_in_thread(filename):
    """
    Run YOLO tracker in its own thread for concurrent processing.

    Args:
        model_name (str): The YOLO11 model object.
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
    """
    results = model.track(filename, save=True, stream=True, device=device)
    for r in results:
        pass


# Create and start tracker threads using a for loop
tracker_threads = []
for video_file in SOURCES:
    print("Starting tracker thread for video file:", video_file)
    thread = threading.Thread(target=run_tracker_in_thread, args=(video_file, ), daemon=True)
    tracker_threads.append(thread)
    thread.start()

# Wait for all tracker threads to finish
for thread in tracker_threads:
    thread.join()

# Clean up and close windows
cv2.destroyAllWindows()