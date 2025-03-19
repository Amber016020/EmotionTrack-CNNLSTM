# 📄 README.md

# EmotionTrack-CNNLSTM

A deep learning-based facial emotion recognition system designed to analyze emotional transitions in video sequences. This project integrates **CNN** and **LSTM** architectures to capture both spatial and temporal patterns in facial expressions, enabling fine-grained emotion trend analysis over time.

---

## 🧠 Features
- **CNN-based Emotion Recognition:** Trained on the Kaggle FER2013 dataset to classify 7 basic emotions.
- **Sequential Modeling with LSTM:** Captures temporal dependencies across video frames to improve emotion prediction accuracy.
- **Automated Video Processing:**
  - Extracts frames from user-recorded videos.
  - Organizes frames into directories based on user ID and experimental condition.
- **Emotion Labeling:**
  - Uses pre-trained CNN model to label each frame's emotion.
  - Saves labeled results in a structured `.npy` format.
- **Emotion Trend Visualization:**
  - Smooths emotion predictions over time.
  - Annotates peaks & valleys with corresponding facial frames.
  - Generates clear visual charts per user & condition.
- **Model Evaluation:** Calculates accuracy, F1-score, confusion matrix to assess model performance.
  
---

## 🗂️ Folder Structure

```
EmotionTrack-CNNLSTM
├── data
│   ├── videos               # Raw video files (one per minute)
│   ├── frames               # Extracted frames organized by user_id/condition
│   └── labeled_frames       # Emotion labels saved in .npy
├── models
│   └── facialemotionmodel.h5  # Pre-trained CNN model
│   └── lstm_model.h5          # Trained CNN+LSTM model
├── results
│   └── emotion_trend_X_X.png  # Visualized emotion trend plots
├── 1_extract_frames.py      # Extract frames & organize folders
├── 2_label_frames.py        # CNN model labels emotion on frames
├── 3_train_cnn_lstm.py      # Train CNN + LSTM sequence model
├── 4_evaluate_model.py      # Evaluate model accuracy & F1-score
├── 5_visualize_emotion.py   # Plot emotion transitions + annotate frames
└── README.md
```

---

## 🚀 Workflow Overview

1. **Frame Extraction:**
   - `1_extract_frames.py`  
   - Extracts 5 FPS frames from videos, categorizes them into `user_id/condition` folders.

2. **Emotion Labeling:**
   - `2_label_frames.py`  
   - Uses pre-trained CNN to label each frame's emotion → saves structured `.npy`.

3. **Model Training:**
   - `3_train_cnn_lstm.py`  
   - Constructs sequences of frames (length = 5), feeds into CNN+LSTM for training.
   - Saves trained model `lstm_model.h5`.

4. **Model Evaluation:**
   - `4_evaluate_model.py`  
   - Calculates accuracy, F1-score, confusion matrix.

5. **Emotion Trend Visualization:**
   - `5_visualize_emotion.py`  
   - Smooths predictions over time.
   - Annotates peaks & valleys with actual frame images.
   - Saves charts for each user-condition.

---

## 📊 Emotion Categories

| Index | Emotion  |
|------|---------|
| 0    | Angry   |
| 1    | Disgust |
| 2    | Fear    |
| 3    | Happy   |
| 4    | Neutral |
| 5    | Sad     |
| 6    | Surprise|

---

## 📦 Dependencies

- Python 3.x
- Keras
- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- tqdm
- scipy

Install:

```bash
pip install -r requirements.txt
```

---

## 💡 Possible Applications
- Affective computing in human-computer interaction (HCI).
- User behavior analysis during real-time communication.
- Psychological studies on emotion dynamics.

---

## 📑 Future Work Ideas
- Integrate real-time streaming emotion feedback.
- Fine-tune on in-domain datasets (real user video data).
- Expand to multi-modal analysis (text, audio + facial).
