import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import timm
import copy

# ---------- EarlyStopping Class ----------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'Ὥ8 EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

# ---------- Dataset ----------
class MultiDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.class_to_idx = [], {}
        idx = 0
        for emotion in os.listdir(root):
            e_path = os.path.join(root, emotion)
            if not os.path.isdir(e_path): continue
            if emotion not in self.class_to_idx:
                self.class_to_idx[emotion] = idx
                idx += 1
            for img in os.listdir(e_path):
                self.samples.append((os.path.join(e_path, img), self.class_to_idx[emotion]))
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert('L').convert('RGB')
        except Exception as e:
            print(f"[Error] Cannot load image {path}: {e}")
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform: image = self.transform(image)
        return image, label


def main():
    # ---------- Transform ----------
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ---------- Data ----------
    print("🚀 Initializing dataset and dataloaders...")
    dataset = MultiDataset('./data/train/AffectNet', transform)
    print(f"✅ Dataset loaded: {len(dataset)} images, {len(dataset.class_to_idx)} classes: {list(dataset.class_to_idx.keys())}")

    targets = dataset.targets
    num_classes = len(dataset.class_to_idx)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=targets, random_state=42)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, num_workers=2, pin_memory=True)
    print("✅ Dataloaders ready.")

    # ---------- Loss ----------
    class_counts = np.bincount([targets[i] for i in train_idx])
    class_weights = 1. / class_counts
    weights = torch.tensor(class_weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🧠 Initializing Xception model...")
    model = timm.create_model('legacy_xception', pretrained=True, num_classes=num_classes, in_chans=3)
    model = model.to(device)
    print("✅ Model ready on device:", device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # ---------- Training ----------
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    num_epochs = 300
    best_val_loss, best_epoch = float('inf'), 0

    print("🎯 Starting training loop...")
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  🏋️ Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"✅ Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += loss_fn(out, y).item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_loss)
        print(f"📊 Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "xception_model_best.pth")
            print(f"📀 Best model saved (Val Loss: {val_loss:.4f})")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered")
            model.load_state_dict(early_stopping.best_model_wts)
            break

    print(f"\n🎉 Training complete! Best model from Epoch {best_epoch} with Val Acc: {val_accs[best_epoch-1]:.4f}")
    torch.save(model.state_dict(), "xception_model_final.pth")

    # ---------- Save History ----------
    with open("train_history.json", "w") as f:
        json.dump({
            "train_losses": train_losses, "val_losses": val_losses,
            "train_accs": train_accs, "val_accs": val_accs
        }, f)

    # ---------- Plot ----------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    x = range(len(train_losses))
    plt.plot(x, train_losses, label="Train Loss", color='tab:blue')
    plt.plot(x, val_losses, label="Val Loss", color='tab:orange')
    plt.fill_between(x, train_losses, alpha=0.2, color='tab:blue')
    plt.fill_between(x, val_losses, alpha=0.2, color='tab:orange')
    plt.title("a. Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, train_accs, label="Train Acc", color='tab:green')
    plt.plot(x, val_accs, label="Val Acc", color='tab:red')
    plt.fill_between(x, train_accs, alpha=0.2, color='tab:green')
    plt.fill_between(x, val_accs, alpha=0.2, color='tab:red')
    plt.title("b. Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    plt.show()

    # ---------- Evaluation ----------
    print("\n🔍 Evaluating best model...")
    model.load_state_dict(torch.load("xception_model_best.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            out = model(X)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    print("📓 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n📄 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=dataset.class_to_idx.keys()))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
