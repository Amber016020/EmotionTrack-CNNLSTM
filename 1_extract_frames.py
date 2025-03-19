import cv2
import os
import re

# Directories for input videos and output frames
VIDEO_DIR = "data/videos"
FRAME_DIR = "data/frames"
FRAME_RATE = 5  # Extract 5 frames per second

# Regular expression to parse file name
FILENAME_PATTERN = r"recording-(\d+)-([a-zA-Z\-]+)-"

# Ensure the output directory exists
if not os.path.exists(FRAME_DIR):
    os.makedirs(FRAME_DIR)

def extract_frames(video_path, user_id, condition, output_folder):
    """
    Extract frames from video and save them in user_id/condition folder.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(fps // FRAME_RATE, 1)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"{output_folder}/frame_{count}.jpg"
            cv2.imwrite(frame_name, frame)
        count += 1

    cap.release()

# Process all videos in the directory
for video_file  in os.listdir(VIDEO_DIR):
    match = re.match(FILENAME_PATTERN, video_file)
    if match:
        user_id = match.group(1)
        condition = match.group(2)

        # Create output folder: data/frames/user_id/condition/
        output_subfolder = os.path.join(FRAME_DIR, user_id, condition)
        os.makedirs(output_subfolder, exist_ok=True)

        print(f"📁 Processing {video_file} → User: {user_id}, Condition: {condition}")

        # Extract frames
        video_path = os.path.join(VIDEO_DIR, video_file)
        extract_frames(video_path, user_id, condition, output_subfolder)


print("✅ Frame extraction completed!")
