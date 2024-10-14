import cv2 
import os
from config import *

# FINO A FRAME 5100 OK

cv2.namedWindow("Video con numero del frame", cv2.WINDOW_NORMAL)

def show_frame_number(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Errore nell'apertura del video")
        return
    
    frame_idx = 0

    while cap.isOpened():
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
        
        cv2.imshow("Video con numero del frame", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()

video_path = os.path.join(PATH_VIDEOS, 'out6.mp4')
show_frame_number(video_path)



