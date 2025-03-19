import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Parameters
LABELS_PATH = "data/labeled_frames/labels.npy"
IMAGE_FOLDER = "data/frames"
SEQUENCE_LENGTH = 5
IMAGE_SIZE = (48, 48)

# Load labels
labels_dict = np.load(LABELS_PATH, allow_pickle=True).item()

# Flatten labels into { full_frame_path : label }
flatten_labels = {}
for user_id in labels_dict:
    for condition in labels_dict[user_id]:
        for frame_name, label in labels_dict[user_id][condition].items():
            frame_path = os.path.join(user_id, condition, frame_name)
            flatten_labels[frame_path] = label

# Group frames by user_id and condition
grouped_frames = {}
for frame_path in flatten_labels:
    parts = frame_path.split(os.sep)
    user_id, condition, frame_name = parts[0], parts[1], parts[2]
    key = (user_id, condition)
    if key not in grouped_frames:
        grouped_frames[key] = []
    grouped_frames[key].append((frame_path, flatten_labels[frame_path]))

# Prepare sequences
sequences = []
labels_list = []

for key in grouped_frames:
    frames = sorted(grouped_frames[key], key=lambda x: int(x[0].split('_')[-1].split('.')[0]))  # sort by frame index
    for i in range(len(frames) - SEQUENCE_LENGTH):
        sequence = []
        for j in range(i, i + SEQUENCE_LENGTH):
            img_path = os.path.join(IMAGE_FOLDER, frames[j][0])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0
            sequence.append(img)
        label = frames[i + SEQUENCE_LENGTH][1]  # use label of last frame
        sequences.append(np.array(sequence))
        labels_list.append(label)

print(f"✅ Total sequences: {len(sequences)}")

# Convert to arrays
x_data = np.array(sequences).reshape(len(sequences), SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
y_data = np.array(labels_list)

# Encode labels
label_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_encoder = LabelEncoder()
label_encoder.fit(label_map)
y_data = label_encoder.transform(y_data)
y_data = to_categorical(y_data, num_classes=7)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
print("✅ x_train shape:", x_train.shape)
print("✅ y_train shape:", y_train.shape)

# Save data (optional)
np.save("data/x_train.npy", x_train)
np.save("data/y_train.npy", y_train)
np.save("data/x_test.npy", x_test)
np.save("data/y_test.npy", y_test)

# Build model
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu'), input_shape=(SEQUENCE_LENGTH, 48, 48, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test))

model.save("models/lstm_model.h5")
print("✅ CNN+LSTM training completed!")
