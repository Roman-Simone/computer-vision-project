from moviepy.video.io.VideoFileClip import VideoFileClip
import os

# Load the video file
video_path = "out11.mp4"  # Change this to your video file path
video = VideoFileClip(video_path)

# Define the segment duration (30 seconds)
segment_duration = 2

# Create the subclip for the first 30 seconds
subclip = video.subclip(0, segment_duration)

# Define the output filename
output_filename = "out11_first2.mp4"

# Write the subclip to a file
subclip.write_videofile(output_filename, codec="libx264")

print(f"First 30-second segment saved as {output_filename}")