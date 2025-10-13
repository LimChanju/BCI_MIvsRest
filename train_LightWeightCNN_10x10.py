import mne
import numpy as np
import glob, re, os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --------------------------
# 1. Dataset 불러오기 (Training 세션만, 단채널 C3)
# --------------------------
files = sorted(glob.glob("./dataset/*T.gdf"))
print("Using files:", files)

X_all, y_all, subj_ids = [], [], []

for f in files:
    subj_id = int(re.findall(r"B0(\d)", f)[0])
    print(f"\n=== Loading {f} (Subject {subj_id}) ===")

    raw = mne.io.read_raw_gdf(f, preload=True)
    events, event_dict = mne.events_from_annotations(raw)

    # --- C3 단일 채널 + FIR 8–30 Hz 필터
    raw.pick_channels(["EEG:C3"])
    raw.filter(8., 30., fir_design="firwin")

    # --- MI 이벤트 정의
    if "769" in event_dict:
        left, right = event_dict["769"], event_dict["770"]
    elif 769 in event_dict:
        left, right = event_dict[769], event_dict[770]
    else:
        print(f"⚠️ {f}에서 MI 이벤트 없음 → 건너뜀")
        continue

    # --- 이벤트 공통화
    events_fixed = events.copy()
    events_fixed[events_fixed[:, -1] == left, -1] = 1
    events_fixed[events_fixed[:, -1] == right, -1] = 2
    event_id = {"left": 1, "right": 2}

    # === Rest (−4~0 s)
    rest_epochs = mne.Epochs(raw, events_fixed, event_id=event_id,
                             tmin=-4.0, tmax=0.0,
                             baseline=None, preload=True)
    X_rest = rest_epochs.get_data()
    y_rest = np.zeros(len(X_rest))

    # === MI (0~4 s)
    mi_epochs = mne.Epochs(raw, events_fixed, event_id=event_id,
                           tmin=0.0, tmax=4.0,
                           baseline=None, preload=True)
    X_mi = mi_epochs.get_data()
    y_mi = np.ones(len(X_mi))

    X_all.append(np.concatenate([X_rest, X_mi], axis=0))
    y_all.append(np.concatenate([y_rest, y_mi], axis=0))
    subj_ids.extend([subj_id] * (len(y_rest) + len(y_mi)))

X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
subj_ids = np.array(subj_ids)

print("\nFinal data shape:", X.shape)
print("Label distribution:", np.unique(y, return_counts=True))
print("Subjects:", np.unique(subj_ids))

# --------------------------
# 2. Light1D-CNN 모델 정의
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

        conv_out = ((n_times - 4 - 4) // 2)
        self.fc = nn.Linear(32 * conv_out, n_classes, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # 입력 형태: (B, 1, T)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --------------------------
# 3. 10×10 subject-dependent CV
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unique_subjects = np.unique(subj_ids)
save_dir = "./Light1DCNN_10x10"
os.makedirs(save_dir, exist_ok=True)

subject_results = []

for subj in unique_subjects:
    print(f"\n=== Subject {subj}: 10×10 CV ===")
    subj_mask = subj_ids == subj
    X_subj, y_subj = X[subj_mask], y[subj_mask]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_subj, y_subj), 1):
        X_train = torch.tensor(X_subj[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_subj[train_idx], dtype=torch.long)
        X_test = torch.tensor(X_subj[test_idx], dtype=torch.float32)
        y_test = torch.tensor(y_subj[test_idx], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

        model = Light1DCNN(n_times=X.shape[2], n_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_acc = 0
        for epoch in range(50):
            model.train()
            correct, total = 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                correct += (out.argmax(1) == yb).sum().item()
                total += yb.size(0)

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    correct += (preds.argmax(1) == yb).sum().item()
                    total += yb.size(0)
            test_acc = correct / total
            best_acc = max(best_acc, test_acc)

        fold_accs.append(best_acc)
        print(f"  Fold {fold:2d} | Best Acc: {best_acc:.3f}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    subject_results.append(mean_acc)
    print(f" → Subject {subj} mean = {mean_acc:.3f} ± {std_acc:.3f}")

    # --- Fold 시각화 ---
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 11), fold_accs, marker='o', color='blue', label='Fold Accuracies')
    plt.axhline(mean_acc, color='red', linestyle='--', label=f'Mean = {mean_acc:.3f}')
    plt.fill_between(range(1, 11),
                     mean_acc - std_acc, mean_acc + std_acc,
                     color='red', alpha=0.2, label=f'±1 SD ({std_acc:.3f})')
    plt.title(f"Subject {subj} - 10×10 CV Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Subj{subj}_Light1DCNN.png"), dpi=300)
    plt.close()

# --------------------------
# 4. 전체 평균 + 막대그래프 시각화
# --------------------------
overall_mean = np.mean(subject_results)
overall_std = np.std(subject_results)

print("\n✅ Subject-wise mean accuracies:", np.round(subject_results, 3))
print(f"Overall mean accuracy = {overall_mean:.3f} ± {overall_std:.3f}")
print(f"📁 Results saved to: {save_dir}/")

# 전체 subject 비교 막대그래프
subjects = [f"Subj{s}" for s in unique_subjects]
plt.figure(figsize=(8, 5))
plt.bar(subjects, subject_results, color='lightgreen', edgecolor='k')
plt.axhline(overall_mean, color='red', linestyle='--', label=f'Mean = {overall_mean:.3f}')
plt.fill_between(range(len(subjects)),
                 overall_mean - overall_std, overall_mean + overall_std,
                 color='red', alpha=0.2, label=f'±1 SD ({overall_std:.3f})')
plt.title("Subject-wise Mean Accuracy (Light1D-CNN, 10×10 CV, C3)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "AllSubjects_Accuracy_Bar.png"), dpi=300)
plt.show()
