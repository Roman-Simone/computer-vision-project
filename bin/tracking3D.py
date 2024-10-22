import sys
import torch
import cv2 as cv
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from config import *
from yolo_model import YoloModel 
from utils import *  
from cameraInfo import CameraInfo  
from tracker import Tracker  

START = 2750                # starting frame
END = 2950                  # ending frame 
FRAME_SKIP = 30  
MODE = 2

pathWeight = os.path.join(PATH_WEIGHT, 'best_v8_800.pt')

if MODE == 1:
    print("FULL RESOLUTION")
    model = YoloModel(model_path=pathWeight, wsize=(3840, 2160), overlap=(0, 0))
elif MODE == 2:
    print("MAX SPEED")
    model = YoloModel(model_path=pathWeight, wsize=(1920, 1130), overlap=(0.1, 0.1))
elif MODE == 3:
    print("BALANCED")
    model = YoloModel(model_path=pathWeight, wsize=(1300, 1130), overlap=(0.05, 0.1))
elif MODE == 4:
    print("STANDARD")
    model = YoloModel(model_path=pathWeight, wsize=(640, 640), overlap=(0.1, 0.1))

video_paths = [os.path.join(PATH_VIDEOS, f'out{cam_idx}.mp4') for cam_idx in VALID_CAMERA_NUMBERS]
cams = []  # Initialize camera controllers

camera_infos = load_pickle(PATH_CALIBRATION_MATRIX)
    
caps = [cv.VideoCapture(video_paths[idx]) for idx in range(len(video_paths))]  # Initialize video captures
trackers = [Tracker(idx) for idx in VALID_CAMERA_NUMBERS]  # Initialize trackers for each camera

cv.namedWindow("frame", cv.WINDOW_NORMAL)

# Lists to store tracked points and detections
tracked_points = []
every_det = []

if START > 0:
    for cap in caps:
        cap.set(cv.CAP_PROP_POS_FRAMES, START)

# Initialize frame index and final tracking point
frame_idx = START
final_point = None

# Dictionary to hold detections for each camera
detecs = {}
for idx in VALID_CAMERA_NUMBERS:
    detecs[idx] = []

# Initialize lists to store frames and detections
all_dets = {}  # Dictionary to hold detections from all cameras

# Main loop for processing frames
while True:
    all_frames = []  # List to hold frames from all cameras

    print(f"FRAME {frame_idx}-------------------------------")

    ret = True  # Flag to check if frames are read successfully

    for curr_cam_idx in range(len(VALID_CAMERA_NUMBERS)):
        cap = caps[curr_cam_idx]  # Get the current camera capture
        cam, _ = take_info_camera(curr_cam_idx, camera_infos) # Get the current camera controller

        ret, frame = cap.read()  # Read the next frame
        if not ret:
            print("[TRACK] Frame corrupt, exiting...")
            exit()  # Exit if the frame is corrupt

        # Save the undistorted frame for visualization later
        uframe = frame

        # Perform object detection on the frame using the Sliced YOLO model
        out, det, uframe = model.predict(uframe, viz=True)

        # Initialize tracking coordinates
        track_x = -1
        track_y = -1
        if out is not None:
            # Get detection output
            x, y, w, h, c = out
            # Store detections in all_dets
            all_dets[curr_cam_idx] = [x, y]
            # Append current detection details to the corresponding camera index
            detecs[VALID_CAMERA_NUMBERS[curr_cam_idx]].append([x, y, w, h])

        # Resize the undistorted frame for display to common size
        all_frames.append(uframe)  # Append the resized frame

        # Update frame index with the current position
        frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)

    #  Dictionary to hold filtered detections after processing
    filt_dets = {}
    for idx, tracker in enumerate(trackers):
        # Update tracker with the current detections or None if no detection
        if VALID_CAMERA_NUMBERS[idx] not in all_dets:
            point = tracker.update(None)
        else:
            point = tracker.update(all_dets[VALID_CAMERA_NUMBERS[idx]])
        if point is not None:
            # Store the filtered detection point if available
            filt_dets[VALID_CAMERA_NUMBERS[idx]] = point

    # Update all detections with filtered detections
    all_dets = filt_dets

    # Convert detections to a single point using camera controllers
    final_point = CameraInfo.detections_to_point(all_dets, camera_infos, final_point)

    if final_point is not None:
        tracked_points.append(final_point)  # Append the final tracked point


    # for i, frame in enumerate(all_frames):
        # print(f"Frame {i+1} shape:", frame.shape)

    # Resize all frames to the same common size before stacking

    # Now proceed with stacking
    if len(all_frames) > 0:  # Only stack if there are frames
        try:
            frame = np.vstack([np.hstack(all_frames[:4]), 
                            np.hstack(all_frames[4:8]), 
                            np.hstack(all_frames[8:])])
        except ValueError as e:
            print(f"Error during frame stacking: {e}")

    if final_point is not None:
        # Draw a green circle if a point was tracked
        cv.circle(frame, (50, 50), 30, (0, 255, 0), -1)
        cv.putText(
            frame,
            f"{len(all_dets)}",  # Display the number of detections
            (40, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )
    else:
        # Draw a red circle if no point was tracked
        cv.circle(frame, (50, 50), 30, (0, 0, 255), -1)
    
    # Show the combined frame in the window
    cv.imshow("frame", frame)

    # Check for keyboard input
    k = cv.waitKey(1)  # Wait for a key event
    if k == ord("d"):
        print(f"SKIPPING {FRAME_SKIP} FRAMES")
        # Skip the specified number of frames for each camera
        for cap in caps:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx + FRAME_SKIP)
    if k == ord("q"):
        print("EXITING")
        break  # Exit the loop if 'q' is pressed

# Save detection results to a text file
with open("detcs.txt", "w") as f:
    for idx in VALID_CAMERA_NUMBERS:
        for det in detecs[idx]:
            f.write(f"{idx} {' '.join([str(x) for x in det])}\n")  # Write camera index and detection data

###################### PLOTS #######################
# Convert tracked points to a NumPy array for plotting
tracked_points_np = np.array(tracked_points)
plot_points = np.array(tracked_points_np).T  # Transpose for easier access

print("Plot points shape:", plot_points.shape)
print("Plot points:", plot_points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")  # Create a 3D plot
ax.set_box_aspect([1, 1, 1])  # Set aspect ratio for the 3D plot

# Get real corner positions and field corners for comparison
positions, field_corners = get_positions()

# Plot real corners on the 3D plot
ax.scatter(
    field_corners[:, 0],
    field_corners[:, 1],
    field_corners[:, 2],
    c="red",  # Color for real corners
    label="Real Corners",
)

x_coords = plot_points[0]  
y_coords = plot_points[1]  
z_coords = plot_points[2]  

ax.scatter(
    x_coords,  
    y_coords,  
    z_coords,  
    c='blue',  
    label="Tracked Points",
    s=50, 
    marker='o'
)

ax.plot(
    x_coords,  
    y_coords,  
    z_coords,  
    color='blue',  
    label="Tracked Path",
)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ax.set_xlim([np.min(x_coords) - 1, np.max(x_coords) + 1])  
ax.set_ylim([np.min(y_coords) - 1, np.max(y_coords) + 1])  
ax.set_zlim([np.min(z_coords) - 1, np.max(z_coords) + 1])  

ax.set_title('3D Tracked Points and Real Corners (with Path)')

ax.legend()

set_axes_equal(ax)

plt.show()
