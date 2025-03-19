import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tqdm import tqdm

# Define directories for input frames and labeled output
FRAME_DIR = "data/frames"
LABELED_DIR = "data/labeled_frames"
CNN_MODEL_PATH = "models/facialemotionmodel.h5"

# Ensure the labeled output directory exists
if not os.path.exists(LABELED_DIR):
    os.makedirs(LABELED_DIR)

# Load the pre-trained CNN model for emotion recognition
cnn_model = load_model(CNN_MODEL_PATH)

# Load the pre-trained CNN model for emotion recognition
label_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(image_path):
    """
    Predicts the emotion from a given image using the trained CNN model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The predicted emotion label.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    img = cv2.resize(img, (48, 48))  # Resize to model's input size
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    pred = cnn_model.predict(img)
    return label_map[np.argmax(pred)]

# Predict emotions for all images and store results
predictions = {}
for user_id in tqdm(os.listdir(FRAME_DIR)):
    user_path = os.path.join(FRAME_DIR, user_id)
    if not os.path.isdir(user_path):
        continue  # Skip non-folder files
    
    predictions[user_id] = {}
    
    for condition in os.listdir(user_path):
        condition_path = os.path.join(user_path, condition)
        if not os.path.isdir(condition_path):
            continue
        
        print(f"📁 Processing: User {user_id}, Condition {condition}")
        predictions[user_id][condition] = {}
        
        for img_name in os.listdir(condition_path):
            img_path = os.path.join(condition_path, img_name)
            label = predict_emotion(img_path)
            predictions[user_id][condition][img_name] = label

# Save labeled results
np.save(os.path.join(LABELED_DIR, "labels.npy"), predictions)
print("✅ Emotion labeling completed and saved!")