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
# 3. Light-weight 1D-CNN 모델 정의 (논문 방식)
# --------------------------
class Light1DCNN(nn.Module):
    def __init__(self, n_times, n_classes=2):
        super(Light1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, bias=False)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, bias=False)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        conv_out = (n_times - 4 - 4) // 2
        self.fc = nn.Linear(32 * conv_out, n_classes, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

model = Light1DCNN(n_times=X.shape[2], n_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 5. 학습 루프 (로그 저장)
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
plt.savefig("/.CNN_result/Light1DCNN_C3.png", dpi=300)
plt.show()