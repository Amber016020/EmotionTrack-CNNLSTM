import pickle
import matplotlib.pyplot as plt

# ==== 載入訓練歷史 ====
with open("models/lstm_training_history.pkl", "rb") as f:
    history = pickle.load(f)

# ==== 畫圖 ====
plt.figure(figsize=(10, 4))

# Accuracy 曲線
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss 曲線
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("models/lstm_training_plot.png")
plt.show()
