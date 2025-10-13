import mne
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# ======================================================
# 1. 데이터 불러오기 (파일명으로 라벨 결정)
# ======================================================
files = sorted(glob.glob("./dataset_MA/*.edf"))
print("Using files:", files)

X_all, y_all, subj_ids = [], [], []
segment_length = 10.0  # 초 단위

for f in files:
    print(f"\n=== Loading {f} ===")
    raw = mne.io.read_raw_edf(f, preload=True)

    # --- 필터 적용 (논문 설정) ---
    raw.notch_filter(50.0, fir_design="firwin")      # 50 Hz 노치 필터
    raw.filter(0.5, 50., fir_design="firwin")        # 0.5–50 Hz band-pass

    # Cz 채널 찾기
    cz_candidates = [ch for ch in raw.ch_names if "Cz" in ch]
    if len(cz_candidates) == 0:
        raise ValueError("⚠️ Cz 채널을 찾을 수 없음")
    cz_idx = raw.ch_names.index(cz_candidates[0])

    total_time = raw.times[-1]

    # 파일명으로 라벨 결정
    if f.endswith("_1.edf"):
        label = 0  # Rest
    elif f.endswith("_2.edf"):
        label = 1  # MA
    else:
        raise ValueError("⚠️ 파일 이름이 예상과 다름 (SubjectXX_1.edf / SubjectXX_2.edf)")

    # 10초 단위 segment 반복
    start = 0.0
    while start + segment_length <= total_time:
        epoch = raw.copy().crop(tmin=start, tmax=start+segment_length).get_data()
        data = epoch[cz_idx, :][np.newaxis, :]
        X_all.append(data)
        y_all.append(label)
        subj_ids.append(int(f.split("Subject")[1].split("_")[0]))
        start += segment_length

X = np.stack(X_all, axis=0)   # (trials, 1, times)
y = np.array(y_all)
subj_ids = np.array(subj_ids)
print("\nFinal data shape:", X.shape)
print("Final labels:", np.unique(y, return_counts=True))

# ======================================================
# 2. Torch Tensor 변환 + Global z-score normalization
# ======================================================
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# ---- 전체 데이터 기준 정규화 ----
mean = X.mean()
std = X.std()
X = (X - mean) / std
print("✅ Global z-score normalization 완료 (논문 설정)")

# ======================================================
# 3. 모델 정의 (Light-weight 1D CNN + BatchNorm)
# ======================================================
class Light1DCNN(nn.Module):
    def __init__(self, n_times, n_classes=2):
        super(Light1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        conv_out = (n_times - 4 - 4) // 2
        self.fc = nn.Linear(32 * conv_out, n_classes, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# ======================================================
# 4. LOSO Cross-validation
# ======================================================
all_test_acc = []
unique_subjects = np.unique(subj_ids)

for test_subj in unique_subjects:
    print(f"\n=== LOSO Fold: Test Subject {test_subj} ===")
    train_mask = subj_ids != test_subj
    test_mask = subj_ids == test_subj

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 모델 초기화
    model = Light1DCNN(n_times=X.shape[2], n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    for epoch in range(30):  # 논문도 약 30 epoch
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # 평가
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)
    acc = correct / total if total > 0 else 0
    all_test_acc.append(acc)
    print(f"Test Acc for Subject {test_subj}: {acc:.3f}")

# ======================================================
# 5. 최종 결과 시각화
# ======================================================
print("\nLOSO Test Accuracies:", all_test_acc)
print("Mean Accuracy: %.3f" % np.mean(all_test_acc))

os.makedirs("./CNN_result", exist_ok=True)
plt.figure(figsize=(12,5))
plt.bar(range(len(all_test_acc)), all_test_acc)
plt.axhline(np.mean(all_test_acc), color='red', linestyle='--', label="Mean")
plt.xlabel("Subject")
plt.ylabel("Accuracy")
plt.title("LOSO Accuracy per Subject (Global z-score, Notch+Bandpass)")
plt.legend()
plt.tight_layout()
plt.savefig("./CNN_result/Light1DCNN_datasetMA_LOSO_globalnorm.png", dpi=300)
plt.show()
