import cv2

def extract_frame(video_path, frame_number, output_image_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if ret:
        # Save the frame as a PNG image
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} saved as {output_image_path}")
    else:
        print(f"Error: Could not read frame {frame_number}")
    
    # Release the video capture object
    cap.release()

# Example usage
video_path = '/Users/stefanobonetto/Documents/GitHub/computer-vision-project/data/videos/calibration/out6F.mp4'
frame_number = 1600  # Change this to the frame number you want to extract
output_image_path = '/Users/stefanobonetto/Documents/GitHub/computer-vision-project/src/frame_100.png'

extract_frame(video_path, frame_number, output_image_path)



