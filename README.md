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
- **Real-Time Emotion Recognition:** Uses webcam input to perform live facial emotion classification and display.
- **Standalone CNN Training Notebook:** Provides an independent Jupyter notebook to train CNN classifiers without LSTM.

---
## Folder Structure

EmotionTrack-CNNLSTM/
├── data/
│   ├── video/               # Original user-recorded videos
│   ├── frame/               # Extracted face frames
│   └── feature/             # CNN feature sequences & pseudo labels
├── models/
│   ├── cnn_model.h5         # Trained CNN
│   └── lstm_model.h5        # Trained LSTM
├── results/
│   └── predictions/         # Output of final emotion predictions
├── 1_train_cnn.py
├── 2_extract_frames.py
├── 3_augment_face_sequences.py
├── 4_generate_pseudo_labels.py
├── 5_train_lstm.py
└── 6_predict_labels.py

  
---

## Pipeline Overview
1. Train CNN on Facial Emotion Dataset
📄 1_train_cnn.py
Train a CNN classifier (e.g., on FER2013) to recognize 7 facial emotions from single images.

2. Extract Faces from User Videos
📄 2_extract_frames.py
Extract face frames from recorded videos at 5 FPS. Organize them into folders by user ID and condition.

3. Augment Face Image Sequences
📄 3_augment_face_sequences.py
Apply data augmentation (horizontal flip, rotation) to expand the training set diversity.

4. Generate Pseudo Labels for CNN Features
📄 4_generate_pseudo_labels.py
Use the trained CNN model to label each frame. Generate and save CNN feature sequences and pseudo labels in .npy and .json formats.

5. Train LSTM on Feature Sequences
📄 5_train_lstm.py
Feed the padded CNN feature sequences into an LSTM model to learn temporal patterns. Save the trained LSTM model for evaluation.

6. Predict Emotion Sequences Using LSTM
📄 6_predict_labels.py
Apply the trained LSTM model to unseen video sequences and generate emotion predictions for each user-condition session.

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
