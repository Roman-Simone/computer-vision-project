import cv2
import argparse
import os
import glob
import numpy as np
from tqdm import tqdm

from stitcher import Stitcher

def frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # return number of frames in the video

def slice_video(video_path):
    out_folder = os.path.join("videos", "cropped")
    os.makedirs(out_folder, exist_ok=True)
    
    # Camera 10 has height 1800 and width 4096, the four sections are all equally wide 1024
    # height = 1800
    # width = 1024
    # offset_x = 1024

    # Camera 9 and 11 have height 1792 and width 4096, the four sections are wide 896, 1152, 1152, 896 respectively
    height = 1792
    width = 1152
    offset_x = 896

    left_out = os.path.join(out_folder, "left.mp4")
    right_out = os.path.join(out_folder, "right.mp4")

    os.system(f'ffmpeg -i {video_path} -filter:v "crop={width}:{height}:{offset_x}:0" {left_out}')
    os.system(f'ffmpeg -i {video_path} -filter:v "crop={width}:{height}:{offset_x + width}:0" {right_out}')

def load_csv(file_name):
    return np.loadtxt(file_name, delimiter=",")         # used to read calibration data from csv files

def rectify_images():
    distort_folder = os.path.join("frames", "distorted")
    rectify_folder = os.path.join("frames", "rectified")
    os.makedirs(rectify_folder, exist_ok=True)

    if not os.path.exists(distort_folder):
        print("No distorted images found.")
        return

    img_files = [os.path.basename(x) for x in glob.glob(os.path.join(distort_folder, "*.png"))] # all distorted images

    for img_file in tqdm(img_files):
        img = cv2.imread(os.path.join(distort_folder, img_file))

        if "left" in img_file:
            mapX = load_csv(os.path.join("calibration", "out11_left_map_x.csv"))
            mapY = load_csv(os.path.join("calibration", "out11_left_map_y.csv"))
        elif "right" in img_file:
            mapX = load_csv(os.path.join("calibration", "out11_right_map_x.csv"))
            mapY = load_csv(os.path.join("calibration", "out11_right_map_y.csv"))

        mapX = np.float32(mapX)
        mapY = np.float32(mapY)

        rectified_img = cv2.remap(img, mapX, mapY, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(rectify_folder, img_file), rectified_img)

def extract_frames(total_frames):
    
    crop_folder = os.path.join("videos", "cropped")
    frames_folder = os.path.join("frames", "distorted")
    os.makedirs(frames_folder, exist_ok=True)

    videos = glob.glob(os.path.join(crop_folder, "*.mp4"))
    with tqdm(total=total_frames * len(videos)) as pbar:                # extract each distorted frame
        for vid in videos:
            vid_name = os.path.splitext(os.path.basename(vid))[0]
            cap = cv2.VideoCapture(vid)
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_name = f"frame{count}_{vid_name}.png"
                cv2.imwrite(os.path.join(frames_folder, frame_name), frame)
                count += 1
                pbar.update(1)

def img_size(img_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    return (height, width)       # return size of frame 

def stitch_images(stitcher, num_frames):
    output_dir = "frames/stitched"
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(num_frames)):
        path = os.path.join(output_dir, f"frame_{idx}.png")
        panorama = stitcher.stitch(idx, computeHomography=(idx == 0))
        cv2.imwrite(path, panorama)

def recreate_video(out_path, fps, total_frames):
    
    # ricompone il video utilizzando i frame rectified
    
    size = img_size(os.path.join("frames", "stitched", "frame_0.png"))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    frames_path = os.path.join("frames", "stitched")

    for i in tqdm(range(total_frames)):
        frame = cv2.imread(os.path.join(frames_path, f"frame_{i}.png"))
        out.write(frame)

    out.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-video", required=True)
    parser.add_argument("-o", "--output-video", required=True)
    args = parser.parse_args()
    
    total_frames = frame_count(args.input_video)

    # slice_video(args.input_video)
    # print("\n\nSliced video into left and right sections.\n\n")
    # extract_frames(total_frames)
    # print("\n\nExtracted frames from the video.\n\n")
    # rectify_images()
    # print("\n\nRectified the distorted images.\n\n")
    
    stitcher = Stitcher()
    stitch_images(stitcher, total_frames)
    
    fps = 30.0
    recreate_video(args.output_video, fps, total_frames)
    print(f"\n\nStitched video saved as {args.output_video}\n\n")