import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import glob
from collections import defaultdict

# ==== 參數設定 ====
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
FRAME_DIR = "data/frame/experiment"
CNN_MODEL_PATH = "models/facialemotionmodel.h5"
LSTM_MODEL_PATH = "models/lstm_emotion_model.h5"
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
FEATURE_DIM = 512
MAX_LEN = 32
STEP = 10

# ==== 載入模型 ====
cnn_model = load_model(CNN_MODEL_PATH, compile=False)
feature_model = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("flatten_3").output)
lstm_model = load_model(LSTM_MODEL_PATH)

# ==== 預測 ====
results_grouped = defaultdict(lambda: {"timestamps": [], "emotions": []})

for user_id in os.listdir(FRAME_DIR):
    user_path = os.path.join(FRAME_DIR, user_id)
    if not os.path.isdir(user_path):
        continue

    for condition in os.listdir(user_path):
        folder_path = os.path.join(user_path, condition)
        if not os.path.isdir(folder_path):
            continue

        user_condition = f"{user_id}_{condition}"
        print(f"Processing: {user_condition}")

        image_files = sorted(glob.glob(os.path.join(folder_path, "face_*.jpg")))
        if not image_files:
            print(f"  [SKIP] No images found in {folder_path}")
            continue

        # === Step 1: 提取特徵 ===
        frames = []
        for img_path in image_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            feature = feature_model.predict(img, verbose=0)[0]
            frames.append(feature)

        if len(frames) == 0:
            continue

        # === Step 2: LSTM 推論 ===
        features = np.array(frames)
        offset = len(results_grouped[user_condition]["timestamps"]) * STEP
        for i in range(0, len(features), STEP):
            segment = features[i:i+MAX_LEN]
            if segment.shape[0] < MAX_LEN:
                pad = np.zeros((MAX_LEN - segment.shape[0], FEATURE_DIM))
                segment = np.vstack([segment, pad])
            elif segment.shape[0] > MAX_LEN:
                segment = segment[:MAX_LEN]
            input_seq = np.expand_dims(segment, axis=0)
            pred = lstm_model.predict(input_seq, verbose=0)
            pred_label = np.argmax(pred)
            results_grouped[user_condition]["emotions"].append(pred_label.item())
            results_grouped[user_condition]["timestamps"].append(i + offset)

# === Step 3: 畫圖 ===
for uc, data in results_grouped.items():
    if not data["timestamps"]:
        continue
    plt.figure(figsize=(10, 3))
    plt.plot(data["timestamps"], data["emotions"], marker='o')
    plt.yticks(ticks=range(len(EMOTION_LABELS)), labels=EMOTION_LABELS)
    plt.title(f"Emotion Trend: {uc}")
    plt.xlabel("Time (frames)")
    plt.ylabel("Predicted Emotion")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(FRAME_DIR, f"{uc}_trend.png")
    plt.savefig(plot_path)
    plt.close()

# === Step 4: 存所有結果 ===
with open(os.path.join(FRAME_DIR, "all_predictions_from_jpg.json"), "w") as f:
    json.dump(results_grouped, f, indent=2)
