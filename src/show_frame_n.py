import cv2 
import os
from src.utils.config import *

# FINO A FRAME 5100 OK

# 1) 0 - 230
# 2) 580 - 770
# 3) 1360 - 1590
# 4) 2100 - 2350
# 5) 2800 - 2940
# 6) 3350 - 3615
# 7) 4050 - 4235
# 8) 4800 - 5000

START = 0

cv2.namedWindow("Video con numero del frame", cv2.WINDOW_NORMAL)

def show_frame_number(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Errore nell'apertura del video")
        return
    
    # Set the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    
    frame_idx = START
    paused = False  # This flag controls pause/play state

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            text = f"Frame: {frame_idx}"
            position = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            color = (0, 255, 0)
            thickness = 4
            
            frame = cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
            
            frame_idx += 1

        # Show the frame (or last frame when paused)
        cv2.imshow("Video con numero del frame", frame)
        
        key = cv2.waitKey(25) & 0xFF  # Wait for 25 ms for a key press
        
        if key == ord('q'):  # Quit on 'q' key
            break
        elif key == ord(' '):  # Toggle pause/play on spacebar press
            paused = not paused  # Toggle the paused flag

    cap.release()
    cv2.destroyAllWindows()

video_path = os.path.join(PATH_VIDEOS, 'out6.mp4')
show_frame_number(video_path)