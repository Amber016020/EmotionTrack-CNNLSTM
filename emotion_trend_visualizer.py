import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy.interpolate import make_interp_spline
from scipy.signal import argrelextrema
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Define the directory containing extracted frames
FRAME_DIR = "data/frames"
IMAGE_SIZE = (48, 48)

# Emotion labels
label_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Generate mock timestamps and emotion predictions (replace with real data)
timestamps = np.linspace(0, 60, 50)  # Simulated timestamps (50 data points over 60 seconds)
emotions = np.random.randint(0, 7, size=50)  # Randomly generated emotions (replace with model output)

# **Smooth the curve using interpolation**
x_smooth = np.linspace(min(timestamps), max(timestamps), 300)  # Generate denser time points
spline = make_interp_spline(timestamps, emotions, k=3)  # Apply cubic spline interpolation
y_smooth = spline(x_smooth)

# **Find peaks (local maxima) and valleys (local minima)**
peaks = argrelextrema(y_smooth, np.greater)[0]  # Identify peaks
valleys = argrelextrema(y_smooth, np.less)[0]  # Identify valleys

# **Plot the emotion trend over time**
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(x_smooth, y_smooth, linestyle='-', linewidth=2, alpha=0.7, color="orange")

# Set Y-axis labels to emotion categories
ax.set_yticks(range(len(label_map)))
ax.set_yticklabels(label_map)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Predicted Emotion")
ax.set_title("Emotion Changes Over Time (10-minute Video)")
ax.grid()

# **Annotate peaks and valleys with corresponding images**
if os.path.exists(FRAME_DIR):
    sorted_frames = sorted(os.listdir(FRAME_DIR))  # Ensure images are sorted correctly

    # Combine peaks and valleys for annotation
    for extreme in np.concatenate((peaks, valleys)):  
        extreme_time = x_smooth[extreme]  # Timestamp of the peak/valley
        extreme_emotion = y_smooth[extreme]  # Corresponding emotion label index

        # **Find the closest matching frame**
        closest_frame_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - extreme_time))
        closest_frame_name = f"frame_{closest_frame_index * 6}.jpg"  # Adjust frame selection interval

        # **Ensure the frame exists before annotation**
        img_path = os.path.join(FRAME_DIR, closest_frame_name)
        if os.path.exists(img_path):
            # **Load and resize the frame**
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))  # Adjust image size

            # **Convert image to a format compatible with matplotlib**
            imagebox = OffsetImage(img, cmap="gray", zoom=0.5)
            ab = AnnotationBbox(imagebox, (extreme_time, extreme_emotion), frameon=False, pad=0.05)
            ax.add_artist(ab)

# **Display the final plot**
plt.show()
