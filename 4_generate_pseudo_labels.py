import os
import numpy as np
import json
import cv2
import tempfile
from keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm
import torch
import timm
from torchvision import transforms
from PIL import Image
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 載入 PyTorch 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('legacy_xception', pretrained=False, num_classes=8, in_chans=3)
model.load_state_dict(torch.load("xception_model_best.pth", map_location=device))
model.eval().to(device)

# 取中間層（例如 avgpool 或 flatten 前的輸出）作為特徵
# 替代 Keras 的 feature_model
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])  # 取出全連接層前的部分
    def forward(self, x):
        with torch.no_grad():
            return self.features(x).squeeze()

feature_extractor = FeatureExtractor(model).to(device)

# ==== 安全儲存函式（.npy / .json） ====
def safe_save_npy(path, data):
    dir_path = os.path.dirname(path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_path)
    with os.fdopen(tmp_fd, 'wb') as tmp_file:
        np.save(tmp_file, data)
    os.replace(tmp_path, path)

def safe_save_json(path, data):
    dir_path = os.path.dirname(path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_path)
    with os.fdopen(tmp_fd, 'w', encoding='utf-8') as tmp_file:
        json.dump(data, tmp_file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

# ==== 設定參數 ====
FRAME_DIR = "data/frame/train"
FEATURE_DIR = "data/feature"
CNN_MODEL_PATH = "xception_affectnet_pretrained.h5"
SEQUENCE_SAVE_PATH = os.path.join(FEATURE_DIR, "sequences.npy")
LABEL_PATH = os.path.join(FEATURE_DIR, "pseudo_labels.json")

EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fear
    "07": 6,  # disgust
    "08": 7,  # surprise
}

os.makedirs(FEATURE_DIR, exist_ok=True)

# ==== 載入模型（flatten 層） ====
cnn_model = load_model(CNN_MODEL_PATH, compile=False)
feature_model = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("global_average_pooling2d").output)

# ==== 載入已存在資料（續跑用）====
features_dict = {}
label_dict = {}

if os.path.exists(SEQUENCE_SAVE_PATH):
    features_dict = np.load(SEQUENCE_SAVE_PATH, allow_pickle=True).item()
    print(f"🔁 Loaded existing features: {len(features_dict)}")

if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_dict = json.load(f)
    print(f"🔁 Loaded existing labels: {len(label_dict)}")

# ==== 特徵抽取函式 ====
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def extract_feature_vector(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    features = feature_extractor(img)  # shape: (2048,)
    return features.cpu().numpy()

# ==== 主流程：每部影片處理 ====
processed_since_last_save = 0

for video_id in tqdm(os.listdir(FRAME_DIR)):
    if video_id in features_dict:
        continue  # 已處理過

    video_path = os.path.join(FRAME_DIR, video_id)
    if not os.path.isdir(video_path):
        continue

    print(f"📁 Processing: {video_id}")

    # 擷取 label
    parts = video_id.split("-")
    if len(parts) >= 3:
        emotion_code = parts[2]
        if emotion_code in EMOTION_MAP:
            label_dict[video_id] = EMOTION_MAP[emotion_code]
        else:
            print(f"⚠️ Unknown emotion code in: {video_id}")
            continue
    else:
        print(f"⚠️ Unexpected video ID format: {video_id}")
        continue

    # 抽特徵序列
    image_files = sorted(os.listdir(video_path), key=lambda x: int(x.split("_")[1].split(".")[0]))
    sequence = []
    for img_name in image_files:
        img_path = os.path.join(video_path, img_name)
        try:
            feature_vector = extract_feature_vector(img_path)
            sequence.append(feature_vector)
        except Exception as e:
            print(f"⚠️ Error processing {img_path} → {e}")

    if sequence:
        features_dict[video_id] = np.array(sequence)
        processed_since_last_save += 1

        # 每處理10部影片就儲存一次
        if processed_since_last_save >= 10:
            safe_save_npy(SEQUENCE_SAVE_PATH, features_dict)
            safe_save_json(LABEL_PATH, label_dict)
            print(f"💾 Auto-saved after processing 10 videos.")
            processed_since_last_save = 0

# 最後一次儲存（處理數不是10的倍數時）
if processed_since_last_save > 0:
    safe_save_npy(SEQUENCE_SAVE_PATH, features_dict)
    safe_save_json(LABEL_PATH, label_dict)
    print(f"💾 Final auto-save of remaining videos.")

# ==== 完成訊息 ====
print("✅ Feature extraction completed.")
print(f"📦 Total videos processed: {len(features_dict)}")
print(f"✅ Labels saved to {LABEL_PATH}")