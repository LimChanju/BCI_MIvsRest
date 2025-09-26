import mne
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# --------------------------
# 1. dataset 폴더 안 training GDF 파일 불러오기
# --------------------------
files = sorted(glob.glob("./dataset/*T.gdf"))
print("Using training files:", files)

all_epochs = []

for f in files:
    print(f"\n=== Loading {f} ===")
    raw = mne.io.read_raw_gdf(f, preload=True)
    events, event_dict = mne.events_from_annotations(raw)
    print("Event dict:", event_dict)

    # EEG 채널만 선택
    raw.pick_channels(["EEG:C3", "EEG:Cz", "EEG:C4"])
    raw.filter(8., 30., fir_design="firwin")

    # 매핑 확인 (파일마다 다름)
    if "769" in event_dict:  # 문자열 키
        left = event_dict["769"]
        right = event_dict["770"]
    elif 769 in event_dict:  # 정수 키
        left = event_dict[769]
        right = event_dict[770]
    else:
        print(f"⚠️ {f}에서 769/770 이벤트 없음 → 건너뜀")
        continue

    # 공통 라벨로 통일 (left=1, right=2)
    mapping = {left: 1, right: 2}
    events_fixed = events.copy()
    for old, new in mapping.items():
        events_fixed[events_fixed[:, -1] == old, -1] = new

    event_id = {"left": 1, "right": 2}

    # Epoching (cue 이후 0.5~2.5s)
    tmin, tmax = 0.5, 2.5
    epochs = mne.Epochs(raw, events_fixed, event_id, tmin, tmax,
                        baseline=None, preload=True)
    all_epochs.append(epochs)

# --------------------------
# 2. 모든 세션 합치기
# --------------------------
if len(all_epochs) == 0:
    raise RuntimeError("⚠️ 유효한 세션이 없음. event_dict를 다시 확인하세요.")

epochs = mne.concatenate_epochs(all_epochs)
X = epochs.get_data()   # (n_trials, 3, n_times)
y = epochs.events[:, -1]

print("Epoch events unique:", np.unique(epochs.events[:, -1], return_counts=True))
y = np.where(y == 1, 0, 1)  # left=0, right=1

print("Final labels unique:", np.unique(y, return_counts=True))

print("\nFinal EEG data:", X.shape, "Labels:", np.unique(y, return_counts=True))

# --------------------------
# 3. Torch Dataset 준비
# --------------------------
X = torch.tensor(X, dtype=torch.float32)  # (trial, 3, time)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# --------------------------
# 4. 논문 기반 Light-weight 1D-CNN 정의
# --------------------------
class Light1DCNN(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2):
        super(Light1DCNN, self).__init__()

        # Conv1D-I: 16 filters, kernel=5, stride=1, He init
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=5, stride=1, bias=True)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')

        # Conv1D-II: 32 filters, kernel=5, stride=1, He init
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, bias=True)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

        # MaxPooling + Dropout(0.25)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        # Flatten 이후 크기 계산
        conv_out = (n_times - 4 - 4) // 2  # kernel=5 두 번 거친 후 pool=2
        self.fc = nn.Linear(32 * conv_out, n_classes, bias=False)  # bias 제거
        nn.init.xavier_uniform_(self.fc.weight)  # Glorot(Xavier) init

    def forward(self, x):
        # Conv1 + ReLU
        x = torch.relu(self.conv1(x))
        # Conv2 + ReLU
        x = torch.relu(self.conv2(x))
        # MaxPool
        x = self.pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Dropout
        x = self.dropout(x)
        # Fully Connected
        x = self.fc(x)
        return x

model = Light1DCNN(n_channels=X.shape[1], n_times=X.shape[2], n_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 5. 학습 루프
# --------------------------
for epoch in range(50):
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

    train_acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}, Train Acc: {train_acc:.3f}")

# --------------------------
# 6. 최종 Test Accuracy
# --------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        pred_labels = preds.argmax(dim=1)
        correct += (pred_labels == yb).sum().item()
        total += yb.size(0)

print(f"\nFinal Test Accuracy: {correct/total:.3f}")
