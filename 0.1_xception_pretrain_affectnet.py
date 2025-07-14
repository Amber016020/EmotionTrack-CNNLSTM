# 1️⃣ Xception 預訓練階段（用 AffectNet）
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 關閉 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- 參數設定 ---
img_size = (96, 96)
batch_size = 16
epochs = 30
affectnet_path = r'D:\emotionRecognition\Face\EmotionTrack-CNNLSTM\data\train\AffectNet'
num_classes = len(os.listdir(affectnet_path))

# --- 資料處理 ---
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_gen = datagen.flow_from_directory(
    affectnet_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    affectnet_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='validation'
)

# --- 模型架構 ---
base_model = Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# --- 預訓練 ---
print("📌 Start pretraining on AffectNet...")
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stop])

# --- 儲存模型 ---
model.save('xception_affectnet_pretrained.h5')

# --- 畫圖 ---
plt.figure(figsize=(6, 4))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orangered', linewidth=1)
plt.title('Validation Accuracy on AffectNet')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig("affectnet_val_accuracy.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history['val_loss'], label='Validation Loss', color='orangered', linewidth=1)
plt.title('Validation Loss on AffectNet')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig("affectnet_val_loss.png", dpi=300)
plt.show()
