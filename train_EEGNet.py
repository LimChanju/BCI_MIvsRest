import mne
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --------------------------
# 1. dataset 불러오기 (Training 세션만)
# --------------------------
files = sorted(glob.glob("./dataset/*T.gdf"))
print("Using files:", files)

X_all, y_all = [], []

for f in files:
    print(f"\n=== Loading {f} ===")
    raw = mne.io.read_raw_gdf(f, preload=True)
    events, event_dict = mne.events_from_annotations(raw)
    raw.pick_channels(["EEG:C3"])
    raw.filter(8., 30., fir_design="firwin")

    # MI 이벤트 찾기
    if "769" in event_dict:
        left = event_dict["769"]
        right = event_dict["770"]
    elif 769 in event_dict:
        left = event_dict[769]
        right = event_dict[770]
    else:
        print(f"⚠️ {f}에서 MI 이벤트 없음 → 건너뜀")
        continue

    # 공통 라벨 매핑
    events_fixed = events.copy()
    events_fixed[events_fixed[:, -1] == left, -1] = 1
    events_fixed[events_fixed[:, -1] == right, -1] = 2
    event_id = {"left": 1, "right": 2}

    # === Rest epochs (−2.0s ~ 0.0s) ===
    rest_epochs = mne.Epochs(raw, events_fixed, event_id=event_id,
                             tmin=-2.0, tmax=0.0,
                             baseline=None, preload=True)
    X_rest = rest_epochs.get_data()   # (n_trials, 1, times)
    y_rest = np.zeros(len(X_rest))    # label=0

    # === MI epochs (0.5s ~ 2.5s) ===
    mi_epochs = mne.Epochs(raw, events_fixed, event_id=event_id,
                           tmin=0.5, tmax=2.5,
                           baseline=None, preload=True)
    X_mi = mi_epochs.get_data()
    y_mi = np.ones(len(X_mi))         # label=1

    # 합치기
    X_all.append(np.concatenate([X_rest, X_mi], axis=0))
    y_all.append(np.concatenate([y_rest, y_mi], axis=0))

# --------------------------
# 2. 데이터 합치기
# --------------------------
X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
print("\nFinal data shape:", X.shape)
print("Final labels:", np.unique(y, return_counts=True))

# Torch tensor 변환
X = torch.tensor(X, dtype=torch.float32)  # (trials, 1, times)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# --------------------------
# 3. EEGNet 모델 정의
# --------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels=1, n_times=500, n_classes=2, F1=8, D=2, F2=16, kernel_length=64, dropout=0.25):
        super(EEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # 1. Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # 2. Depthwise Convolution (spatial filter)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()

        # Pooling + Dropout
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # 3. Separable Convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # 최종 FC
        out_size = self._get_out_size()
        self.classifier = nn.Linear(out_size, n_classes)

    def _get_out_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.depthwise(x)
            x = self.bn2(x)
            x = self.elu(x)
            x = self.pool1(x)
            x = self.drop1(x)
            x = self.separable_conv(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        # x: (batch, 1, times) → (batch, 1, channels, times)
        x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --------------------------
# 4. 모델 초기화
# --------------------------
model = EEGNet(n_channels=1, n_times=X.shape[2], n_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 5. 학습 루프
# --------------------------
train_losses, train_accs, test_accs = [], [], []

for epoch in range(30):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # --- Test Accuracy ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)
    test_acc = correct / total
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}, Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")

# --------------------------
# 6. 결과 시각화 + 저장
# --------------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy (Train vs Test)")
plt.legend()

plt.tight_layout()
plt.savefig("./CNN_result/EEGNet_C3.png", dpi=300)
plt.show()
