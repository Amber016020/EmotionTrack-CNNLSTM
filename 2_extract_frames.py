import cv2
import os

# ==== 設定參數 ====
VIDEO_DIR = "data/video/train"
FRAME_DIR = "data/frame/train"
FRAME_RATE = 5
FACE_SIZE = (112, 112)

# 載入 OpenCV 的人臉偵測模型
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
os.makedirs(FRAME_DIR, exist_ok=True)

def extract_face_frames(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(fps // FRAME_RATE, 1)

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, FACE_SIZE)

                # 每支影片一個資料夾，命名為 video_id（去掉副檔名）
                subdir = os.path.join(FRAME_DIR, video_id)
                os.makedirs(subdir, exist_ok=True)

                frame_name = f"face_{saved}.jpg"
                frame_path = os.path.join(subdir, frame_name)
                cv2.imwrite(frame_path, face_resized)
                saved += 1
            else:
                print(f"⚠️ No face detected at frame {count} in {video_path}")

        count += 1

    cap.release()

# ==== 主處理流程 ====
for root, dirs, files in os.walk(VIDEO_DIR):
    for video_file in files:
        if video_file.endswith(".mp4"):
            video_id = os.path.splitext(video_file)[0]  # 去掉 .mp4

            subdir = os.path.join(FRAME_DIR, video_id)
            if os.path.exists(subdir) and len(os.listdir(subdir)) > 0:
                print(f"⏩ Skipping {video_file}: already processed.")
                continue

            print(f"🎞️ Processing {video_file} → Folder: {video_id}")
            video_path = os.path.join(root, video_file)
            extract_face_frames(video_path, video_id)

print("✅ All face frames extracted, gray-scaled, resized, and saved (one folder per video)!")
