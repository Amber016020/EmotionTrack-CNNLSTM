from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from keras.models import load_model

# Load the trained LSTM model and test dataset
model = load_model("models/lstm_model.h5")
x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")

# Make predictions on the test dataset
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"✅ LSTM Model Accuracy: {accuracy:.2f}")

# Display classification metrics (precision, recall, F1-score)
print(classification_report(y_true_classes, y_pred_classes))
