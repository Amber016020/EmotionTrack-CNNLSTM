import os
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import pickle
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==== 路徑與參數 ====
SEQUENCE_PATH = "data/feature/sequences.npy"
LABEL_PATH = "data/feature/pseudo_labels.json"
MODEL_PATH = "models/lstm_emotion_model.h5"
HISTORY_PATH = "models/lstm_training_history.pkl"
XTEST_PATH = "data/feature/X_test.npy"
YTEST_PATH = "data/feature/y_test.npy"
NUM_CLASSES = 8

# ==== 載入資料 ====
sequence_data = np.load(SEQUENCE_PATH, allow_pickle=True).item()
with open(LABEL_PATH, "r") as f:
    label_dict = json.load(f)

# ==== 對齊資料 ====
common_keys = set(sequence_data.keys()) & set(label_dict.keys())
if not common_keys:
    raise ValueError("❌ No matching keys between features and labels!")

X = []
y = []

max_len = max(sequence_data[k].shape[0] for k in common_keys)
feature_dim = next(iter(sequence_data.values())).shape[1]
print(f"📏 Max sequence length: {max_len}, Feature dimension: {feature_dim}")

for key in common_keys:
    seq = sequence_data[key]
    label = label_dict[key]

    # Padding
    pad_len = max_len - seq.shape[0]
    pad = np.zeros((pad_len, feature_dim))
    padded_seq = np.vstack([seq, pad])

    X.append(padded_seq)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"✅ Loaded {len(X)} samples. Shape: {X.shape}")

# ==== 三切資料：Train / Val / Test ====
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42
)
# 0.1111 ≈ 1/9，使得驗證集佔總資料的 10%，與測試集一樣

print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ==== Class Weights ====
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("🧾 Class weights:", class_weights)

# ==== 建立 LSTM 模型 ====
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(max_len, feature_dim)))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==== 訓練 ====
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ==== 顯示訓練與驗證準確率 ====
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"📈 Final Training Accuracy: {train_acc:.4f}")
print(f"📉 Final Validation Accuracy: {val_acc:.4f}")

# ==== 測試集評估 ====
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🧪 Test Accuracy: {test_acc:.4f}")

# ==== 儲存模型與紀錄 ====
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history.history, f)
np.save(XTEST_PATH, X_test)
np.save(YTEST_PATH, y_test)

print(f"✅ LSTM model saved to: {MODEL_PATH}")
print(f"📊 Training history saved to: {HISTORY_PATH}")
print(f"🧪 Test data saved to: {XTEST_PATH} / {YTEST_PATH}")
