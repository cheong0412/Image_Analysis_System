import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# -----------------------------
# [1] 커스텀 데이터셋 정의
# -----------------------------
class MaskDataset(Dataset):
    def __init__(self, mask_dir, nomask_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for file in os.listdir(mask_dir):
            self.data.append(os.path.join(mask_dir, file))
            self.labels.append(1)

        for file in os.listdir(nomask_dir):
            self.data.append(os.path.join(nomask_dir, file))
            self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# [2] 이미지 전처리 및 데이터로더
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

mask_dir = './data/with_mask/'
nomask_dir = './data/without_mask/'
dataset = MaskDataset(mask_dir, nomask_dir, transform=transform)

# 훈련/검증 데이터 분리 (9:1)
train_len = int(len(dataset) * 0.9)
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train size: {train_len}, Val size: {val_len}")

# -----------------------------
# [3] CNN 모델 정의
# -----------------------------
class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # 두 클래스
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# -----------------------------
# [4] 학습 준비
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MaskCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# [5] 학습 루프
# -----------------------------
best_acc = 0.0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(10):
    # 학습
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # 검증
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    # 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), f"./models/mask_detector_epoch10_{val_acc:.3f}.pt")
        print(">> 새로운 최고 성능 모델 저장됨!")


# val_acc = correct / total
# val_losses.append(val_loss / len(val_loader))
# val_accuracies.append(val_acc)

# print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

#     # 모델 저장
# if val_acc > best_acc:
#         best_acc = val_acc
#         os.makedirs("./models", exist_ok=True)
#         torch.save(model.state_dict(), f"./models/mask_detector_epoch10_{val_acc:.3f}.pt")
#         print(">> 새로운 최고 성능 모델 저장됨!")

# -----------------------------
# [6] 시각화
# -----------------------------
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()
