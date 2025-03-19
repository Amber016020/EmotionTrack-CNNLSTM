import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy.interpolate import make_interp_spline
from scipy.signal import argrelextrema
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Parameters
FRAME_DIR = "data/frames"
LABELS_PATH = "data/labeled_frames/labels.npy"
IMAGE_SIZE = (48, 48)
label_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load labeled emotion data
labels_dict = np.load(LABELS_PATH, allow_pickle=True).item()

# Traverse user_id and condition folders
for user_id in labels_dict:
    for condition in labels_dict[user_id]:
        print(f"📊 Plotting: User {user_id}, Condition {condition}")

        frames_info = labels_dict[user_id][condition]
        sorted_frame_names = sorted(frames_info.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0]))

        timestamps = []
        emotions = []
        for idx, frame_name in enumerate(sorted_frame_names):
            timestamps.append(idx)  # 以 frame index 當時間軸
            emotion_label = frames_info[frame_name]
            emotions.append(label_map.index(emotion_label))

        if len(emotions) < 6:
            continue  # Skip if too few frames to smooth

        # Smooth curve
        x_smooth = np.linspace(min(timestamps), max(timestamps), 300)
        spline = make_interp_spline(timestamps, emotions, k=3)
        y_smooth = spline(x_smooth)

        # Find peaks and valleys
        peaks = argrelextrema(y_smooth, np.greater)[0]
        valleys = argrelextrema(y_smooth, np.less)[0]

        # Plot
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(x_smooth, y_smooth, linestyle='-', linewidth=2, alpha=0.7, color="orange")
        ax.set_yticks(range(len(label_map)))
        ax.set_yticklabels(label_map)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Predicted Emotion")
        ax.set_title(f"Emotion Trend - User: {user_id}, Condition: {condition}")
        ax.grid()

        # Annotate peaks and valleys
        frame_folder = os.path.join(FRAME_DIR, user_id, condition)
        for extreme in np.concatenate((peaks, valleys)):
            extreme_time = x_smooth[extreme]
            extreme_emotion = y_smooth[extreme]

            # Find closest frame
            closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - extreme_time))
            frame_name = sorted_frame_names[closest_idx]
            img_path = os.path.join(frame_folder, frame_name)

            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (100, 100))
                imagebox = OffsetImage(img, cmap="gray", zoom=0.5)
                ab = AnnotationBbox(imagebox, (extreme_time, extreme_emotion), frameon=False, pad=0.05)
                ax.add_artist(ab)

        # Save or Show
        output_path = f"results/emotion_trend_{user_id}_{condition}.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
