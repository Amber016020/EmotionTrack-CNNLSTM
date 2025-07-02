import os
import cv2
from pathlib import Path
from tqdm import tqdm

# 設定輸入與輸出路徑
RAW_DATASETS = ['AffectNet', 'CK+ dataset', 'JAFFE']
INPUT_DIR = './data/train'
OUTPUT_DIR = './data/train/test'
IMG_SIZE = (64, 64)

# 載入 Haar Cascade 做人臉偵測
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        return None

    # 擷取第一個偵測到的臉
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, IMG_SIZE)
    return face

def process_dataset(dataset_name):
    input_path = os.path.join(INPUT_DIR, dataset_name)
    for emotion in os.listdir(input_path):
        emotion_path = os.path.join(input_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        output_emotion_path = os.path.join(OUTPUT_DIR, dataset_name, emotion)
        os.makedirs(output_emotion_path, exist_ok=True)

        for img_name in tqdm(os.listdir(emotion_path), desc=f"{dataset_name}/{emotion}"):
            img_path = os.path.join(emotion_path, img_name)
            processed = process_image(img_path)
            if processed is not None:
                save_path = os.path.join(output_emotion_path, img_name)
                cv2.imwrite(save_path, processed)

def main():
    for dataset in RAW_DATASETS:
        process_dataset(dataset)

if __name__ == "__main__":
    main()
