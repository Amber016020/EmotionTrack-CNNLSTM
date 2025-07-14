import cv2
import os

# === 設定 ===
FRAME_DIR = "data/frame/train"
FACE_SIZE = (112, 112)
ROT_ANGLES = [-15, -10, -5, 5, 10, 15]

def augment_sequence_folder(folder_path):
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    seq_name = os.path.basename(folder_path)

    # ➤ 水平翻轉
    flip_dir = os.path.join(FRAME_DIR, f"{seq_name}_flip")
    os.makedirs(flip_dir, exist_ok=True)

    for f in frame_files:
        img = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_COLOR)  # 彩色讀圖
        flipped = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(flip_dir, f), flipped)

    # ➤ 旋轉增強
    for angle in ROT_ANGLES:
        rot_dir = os.path.join(FRAME_DIR, f"{seq_name}_rot{angle}")
        os.makedirs(rot_dir, exist_ok=True)

        for f in frame_files:
            img = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_COLOR)  # 彩色讀圖
            M = cv2.getRotationMatrix2D((FACE_SIZE[0] // 2, FACE_SIZE[1] // 2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, FACE_SIZE, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(os.path.join(rot_dir, f), rotated)

# === 主流程 ===
for folder in os.listdir(FRAME_DIR):
    full_path = os.path.join(FRAME_DIR, folder)
    if os.path.isdir(full_path) and not any(c in folder for c in ['_flip', '_rot']):
        print(f"📁 Augmenting {folder}")
        augment_sequence_folder(full_path)

print("✅ 彩色資料的翻轉與旋轉增強已完成！")
