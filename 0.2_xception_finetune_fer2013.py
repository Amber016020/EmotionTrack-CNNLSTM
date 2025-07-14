import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- 參數 ---
img_size = (96, 96)
batch_size = 32
epochs = 30
fer_path = r'D:\emotionRecognition\Face\EmotionTrack-CNNLSTM\data\train\FER-2013\train'
num_classes = len(os.listdir(fer_path))

# --- 資料增強設定 ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)

train_gen = datagen.flow_from_directory(
    fer_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    fer_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='validation'
)

# --- 模型建立與載入預訓練權重 ---
base_model_new = Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
x = GlobalAveragePooling2D()(base_model_new.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # ← Dropout 加在 dense 後，防止過擬合
output = Dense(num_classes, activation='softmax')(x)
model_new = Model(inputs=base_model_new.input, outputs=output)

# --- 載入卷積層的預訓練權重 ---
pretrained_model = load_model('xception_affectnet_pretrained.h5')
for i in range(len(base_model_new.layers)):
    try:
        model_new.layers[i].set_weights(pretrained_model.layers[i].get_weights())
    except:
        print(f"⚠️ Skip incompatible layer {i}")

# --- 編譯與訓練 ---
model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("📌 Start fine-tuning on FER2013 with data augmentation and dropout...")
history = model_new.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stop])

# --- 模型儲存 ---
model_new.save('xception_finetuned_fer2013_dropout_aug.h5')

# --- 訓練曲線圖 ---
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='royalblue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orangered')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='royalblue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orangered')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("lstm_training_plot_dropout_aug.png", dpi=300)
plt.show()
