import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Paths and parameters
LABELS_PATH = "data/labeled_frames/labels.npy"
IMAGE_FOLDER = "data/frames"  # Directory containing extracted frames
SEQUENCE_LENGTH = 5  # Number of frames per sequence
IMAGE_SIZE = (48, 48)  # Image resolution

# Load emotion labels
labels_dict = np.load(LABELS_PATH, allow_pickle=True).item()

# Ensure labels are sorted by time order
sorted_files = sorted(labels_dict.keys())  

# Construct LSTM training dataset
sequences = []
labels_list = []

for i in range(len(sorted_files) - SEQUENCE_LENGTH):
    sequence = []
    for j in range(i, i + SEQUENCE_LENGTH):
        img_path = os.path.join(IMAGE_FOLDER, sorted_files[j])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        img = cv2.resize(img, IMAGE_SIZE)  # Resize image
        img = img / 255.0  # Normalize pixel values
        sequence.append(img)

    label = labels_dict[sorted_files[i + SEQUENCE_LENGTH]]  # 取最後一張影像的標記
    sequences.append(np.array(sequence))
    labels_list.append(label)

# Convert to NumPy arrays and reshape
x_data = np.array(sequences).reshape(len(sequences), SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
y_data = np.array(labels_list)

# Convert string labels to integer labels
label_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_encoder = LabelEncoder()
label_encoder.fit(label_map)
y_data = label_encoder.transform(y_data)  # Encode labels as integers

# Convert labels to one-hot encoding
y_data = to_categorical(y_data, num_classes=7)

# Split dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Verify dataset shapes
print("✅ x_train shape:", x_train.shape)  # (batch_size, SEQUENCE_LENGTH, 48, 48, 1)
print("✅ y_train shape:", y_train.shape)  # (batch_size, 7) (one-hot encoded)

if not os.path.exists("data"):
    os.makedirs("data")
    
# Save test data
np.save("data/x_test.npy", x_test)
np.save("data/y_test.npy", y_test)

# Save training data
np.save("data/x_train.npy", x_train)
np.save("data/y_train.npy", y_train)

print("✅ Test data successfully saved!")

# CNN + LSTM Model
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu'), input_shape=(SEQUENCE_LENGTH, 48, 48, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))  # 7 emotion classes

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test))

# Save the trained model
model.save("models/lstm_model.h5")
print("✅ CNN+LSTM training completed!")
