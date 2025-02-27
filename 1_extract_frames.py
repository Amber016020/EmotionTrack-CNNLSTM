import cv2
import os

# Directories for input videos and output frames
VIDEO_DIR = "data/videos"
FRAME_DIR = "data/frames"
FRAME_RATE = 5  # Extract 5 frames per second

# Ensure the output directory exists
if not os.path.exists(FRAME_DIR):
    os.makedirs(FRAME_DIR)

def extract_frames(video_path, output_folder):
    """
    Extracts frames from a video file at a specified frame rate.
    
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // FRAME_RATE

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
for video in os.listdir(VIDEO_DIR):
    extract_frames(os.path.join(VIDEO_DIR, video), FRAME_DIR)

print("✅ Frame extraction completed!")
