import cv2
import numpy as np
from keras.models import model_from_json

# Load CNN model from json and weights
def load_model():
    with open("models/facialemotionmodel.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights("models/facialemotionmodel.h5")
    return model

# Extract features
def extract_features(image):
    image = np.array(image).reshape(1, 48, 48, 1)
    return image / 255.0

# Main loop
def run_realtime():
    model = load_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
    
    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            input_data = extract_features(roi)
            prediction = model.predict(input_data)
            emotion = labels[prediction.argmax()]
            print(f"Detected: {emotion}")
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow('Real-time Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
