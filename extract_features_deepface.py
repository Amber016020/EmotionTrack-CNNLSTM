import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from tqdm import tqdm
from deepface import DeepFace

# ==== 路徑參數 ====
FRAME_DIR = "data/frames"
FEATURE_DIR = "data/features"
MODEL_NAME = "Facenet"  # 可換成 'ArcFace', 'VGG-Face', 'OpenFace', 'DeepFace'
FEATURE_PATH = os.path.join(FEATURE_DIR, f"features_{MODEL_NAME.lower()}.npy")

os.makedirs(FEATURE_DIR, exist_ok=True)

# ==== 讀取已存在的特徵檔（如果有）====
if os.path.exists(FEATURE_PATH):
    features_dict = np.load(FEATURE_PATH, allow_pickle=True).item()
    print(f"🗂️ Loaded {len(features_dict)} existing features from {FEATURE_PATH}")
else:
    features_dict = {}

# ==== 特徵提取函數 ====
def extract_feature_vector(image_path):
    try:
        embedding_obj = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME, enforce_detection=False)
        return np.array(embedding_obj[0]["embedding"])
    except Exception as e:
        print(f"⚠️ Error extracting {image_path}: {e}")
        return None

# ==== 批次處理所有影格 ====
for user_id in tqdm(os.listdir(FRAME_DIR)):
    user_path = os.path.join(FRAME_DIR, user_id)
    if not os.path.isdir(user_path):
        continue

    for condition in os.listdir(user_path):
        condition_path = os.path.join(user_path, condition)
        if not os.path.isdir(condition_path):
            continue

        print(f"📁 Extracting features: User {user_id}, Condition {condition}")
        for img_name in os.listdir(condition_path):
            img_path = os.path.join(condition_path, img_name)
            key = f"{user_id}/{condition}/{img_name}"
            if key in features_dict:
                continue  # ✅ 已經有了就跳過
            feature_vector = extract_feature_vector(img_path)
            if feature_vector is not None:
                features_dict[key] = feature_vector

# ==== 儲存特徵 ====
np.save(FEATURE_PATH, features_dict)
print(f"✅ Feature extraction completed using {MODEL_NAME} and saved to {FEATURE_PATH}")
