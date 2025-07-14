import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- 基本參數設定 ---
img_size = (96, 96)
batch_size = 16
epochs_pretrain = 30
epochs_finetune = 30

# --- 資料路徑設定 ---
affectnet_path = r'D:\emotionRecognition\Face\EmotionTrack-CNNLSTM\data\train\AffectNet'
fer_path = r'D:\emotionRecognition\Face\EmotionTrack-CNNLSTM\data\train\FER-2013\train'

# 自動偵測類別數
num_classes = len(os.listdir(affectnet_path))

# --- AffectNet 預處理器與載入器 ---
affectnet_gen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

affectnet_train = affectnet_gen.flow_from_directory(
    affectnet_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

affectnet_val = affectnet_gen.flow_from_directory(
    affectnet_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --- EarlyStopping 設定 ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# --- 建立 Xception 模型（使用 ImageNet 預訓練權重）---
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("📌 開始在 AffectNet 預訓練 Xception ...")
history_pretrain = model.fit(
    affectnet_train,
    validation_data=affectnet_val,
    epochs=epochs_pretrain,
    callbacks=[early_stop]
)
model.save('xception_affectnet_pretrained.h5')

# --- 重新載入 FER2013 並更新類別數 ---
num_classes_fer = len(os.listdir(fer_path))

# 重新建立模型（輸出層為 7 類）也保留 ImageNet 預訓練
base_model_finetune = Xception(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
x = GlobalAveragePooling2D()(base_model_finetune.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes_fer, activation='softmax')(x)
model_finetune = Model(inputs=base_model_finetune.input, outputs=predictions)

# 載入 AffectNet 預訓練好的卷積層權重（前面幾層）
for i in range(len(base_model_finetune.layers)):
    try:
        model_finetune.layers[i].set_weights(model.layers[i].get_weights())
    except:
        print(f"⚠️ 跳過第 {i} 層權重（不相容）")

model_finetune.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- FER2013 載入器 ---
fer_gen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

fer_train = fer_gen.flow_from_directory(
    fer_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    subset='training'
)

fer_val = fer_gen.flow_from_directory(
    fer_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation'
)

print("📌 開始在 FER-2013 微調 Xception ...")
history_finetune = model_finetune.fit(
    fer_train,
    validation_data=fer_val,
    epochs=epochs_finetune,
    callbacks=[early_stop]
)
model_finetune.save('xception_finetuned_fer2013.h5')

# --- 視覺化訓練結果（分開畫）---
plt.figure(figsize=(10, 8))

# 第一張圖：AffectNet
plt.subplot(2, 1, 1)
plt.plot(history_pretrain.history['accuracy'], label='AffectNet Train Acc')
plt.plot(history_pretrain.history['val_accuracy'], label='AffectNet Val Acc')
plt.plot(history_pretrain.history['loss'], label='AffectNet Train Loss')
plt.plot(history_pretrain.history['val_loss'], label='AffectNet Val Loss')
plt.title('AffectNet Training History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.grid(True)

# 第二張圖：FER2013
plt.subplot(2, 1, 2)
plt.plot(history_finetune.history['accuracy'], label='FER2013 Train Acc')
plt.plot(history_finetune.history['val_accuracy'], label='FER2013 Val Acc')
plt.plot(history_finetune.history['loss'], label='FER2013 Train Loss')
plt.plot(history_finetune.history['val_loss'], label='FER2013 Val Loss')
plt.title('FER2013 Fine-Tuning History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("CNN_training_plot_split.png", dpi=300)
plt.show()
