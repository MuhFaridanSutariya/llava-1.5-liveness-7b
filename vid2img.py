import cv2
import os
from tqdm.auto import tqdm

def video_to_frames(video_path, output_path, max_frames):
    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read each frame and save it as an image
    while success and count < max_frames:
        cv2.imwrite(f"{output_path}/frame{count}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

    print(f'{count} frames extracted from {video_path}')

def process_videos_in_folder(folder_path, output_folder, max_frames):
    if not os.path.exists(folder_path):
        print(f"Input folder '{folder_path}' does not exist.")
        return

    # List all files in the folder
    video_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Filter out non-video files based on extensions (assuming mp4, avi, mkv, etc.)
    video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mkv'))]

    if not video_files:
        print(f"No video files found in folder '{folder_path}'.")
        return

    for video_file in tqdm(video_files, desc='Convert Videos to Images', total=len(video_files)):
        video_path = os.path.join(folder_path, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_path = os.path.join(output_folder, video_name)
        
        print(f"Processing {video_file}...")
        video_to_frames(video_path, video_output_path, max_frames)
        print(f"Finished processing {video_file}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert MP4 videos to JPG images per frame")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing video files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where images will be saved")
    parser.add_argument("--max_frames", type=int, default=250, help="Maximum number of frames to extract per video")
    args = parser.parse_args()

    input_folder_path = args.input_folder
    output_folder_path = args.output_folder
    max_frames = args.max_frames

    process_videos_in_folder(input_folder_path, output_folder_path, max_frames)
