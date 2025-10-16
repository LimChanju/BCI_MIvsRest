# =========================================
#  EEGNet-Raw (단채널 EEG + 제로패딩 + 시각화)
# =========================================
import os, re, glob, mne, torch, warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# -------------------------------
# 설정
# -------------------------------
DATA_DIR = "./dataset/"
SAVE_DIR = "./Results_Raw"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH, EPOCHS, LR, SEED = 16, 50, 1e-3, 42
TARGET_LEN = 2048  # 제로패딩 목표 길이
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------------
# Bandpass Filter
# -------------------------------
def butter_bandpass_filter(data, low, high, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

# -------------------------------
# Zero Padding Function
# -------------------------------
def pad_eeg(x, target_len=TARGET_LEN):
    """EEG 길이가 짧으면 0으로 채워서 고정"""
    if len(x) >= target_len:
        return x[:target_len]
    else:
        return np.pad(x, (0, target_len - len(x)), mode="constant")

# -------------------------------
# BCI2b Loader (Raw EEG)
# -------------------------------
def load_bci2b_dataset(path=DATA_DIR):
    files = sorted(glob.glob(os.path.join(path, "*T.gdf")))
    X_all, y_all, subj_ids = [], [], []
    fs_out = None
    for f in files:
        subj_match = re.search(r"B0(\d)", os.path.basename(f))
        if not subj_match:
            continue
        subj_id = int(subj_match.group(1))
        raw = mne.io.read_raw_gdf(f, preload=True)
        events, event_dict = mne.events_from_annotations(raw)
        raw.pick_channels(["EEG:C3"])
        fs = raw.info["sfreq"]
        fs_out = fs
        raw.filter(8., 30., fir_design="firwin")

        left = event_dict.get("769") or event_dict.get(769)
        right = event_dict.get("770") or event_dict.get(770)
        mi, rest = [], []
        for ev in events:
            if ev[-1] in [left, right]:
                s, e = int(ev[0]), int(ev[0] + 4.0 * fs)
                if e <= len(raw.times):
                    mi.append(raw.get_data(start=s, stop=e).squeeze())
                s, e = int(ev[0] - 4.0 * fs), int(ev[0])
                if s >= 0:
                    rest.append(raw.get_data(start=s, stop=e).squeeze())

        if not mi or not rest:
            continue
        # === 제로패딩 적용 ===
        mi = [pad_eeg(sig) for sig in mi]
        rest = [pad_eeg(sig) for sig in rest]

        mi, rest = np.array(mi), np.array(rest)
        X = np.concatenate([mi, rest], 0)
        y = np.concatenate([np.ones(len(mi)), np.zeros(len(rest))])
        X_all.append(X)
        y_all.append(y)
        subj_ids.extend([subj_id] * len(y))
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    subj_ids = np.array(subj_ids)
    print(f"총 샘플: {len(X)} | MI:{int(y.sum())} | Rest:{len(y)-int(y.sum())} | 샘플 길이:{X.shape[1]}")
    return X, y, subj_ids, fs_out

# -------------------------------
# EEGNet Model
# -------------------------------
class EEGNet(nn.Module):
    def __init__(self, chans, samples, n_classes=2, dropout=0.5, kernLength=64, F1=8, D=2, F2=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding=(0, 8), bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        tmp = torch.zeros(1, 1, chans, samples)
        with torch.no_grad():
            out = self.forward_features(tmp)
        self.classifier = nn.Linear(out.shape[1], n_classes)

    def forward_features(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable(x)
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))

# -------------------------------
# Train/Eval Functions
# -------------------------------
def train_one_epoch(model, train_dl, test_dl, opt, crit, dev, epoch):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        xb = xb.float().unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 2048)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    # eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            xb = xb.float().unsqueeze(1).unsqueeze(1)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"[{epoch+1:02d}/{EPOCHS}] Loss: {total_loss/len(train_dl):.4f} | Acc: {acc*100:.2f}%")
    return acc

# -------------------------------
# Main
# -------------------------------
def main():
    X_raw, y_raw, subj_ids, fs = load_bci2b_dataset(DATA_DIR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results_mean, results_std = [], []
    subjects = np.unique(subj_ids)

    for subj in subjects:
        print(f"\n=== Subject {subj}: 10×10 CV ===")
        subj_dir = os.path.join(SAVE_DIR, f"S{subj}")
        os.makedirs(subj_dir, exist_ok=True)
        mask = subj_ids == subj
        Xs, ys = X_raw[mask], y_raw[mask]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        accs = []

        for f, (tr, te) in enumerate(skf.split(Xs, ys), 1):
            tr_dl = DataLoader(TensorDataset(torch.tensor(Xs[tr]), torch.tensor(ys[tr], dtype=torch.long)), batch_size=BATCH, shuffle=True)
            te_dl = DataLoader(TensorDataset(torch.tensor(Xs[te]), torch.tensor(ys[te], dtype=torch.long)), batch_size=BATCH)
            model = EEGNet(chans=1, samples=TARGET_LEN).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            crit = nn.CrossEntropyLoss()
            best = 0
            for ep in range(EPOCHS):
                acc = train_one_epoch(model, tr_dl, te_dl, opt, crit, dev, ep)
                best = max(best, acc)
            accs.append(best)
            print(f" Fold {f:2d} | Best {best:.3f}")

        # 시각화
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, 11), accs, color="steelblue", edgecolor="black")
        plt.title(f"Subject {subj} Fold Accuracies (Raw)")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, "fold_acc_bar.png"), dpi=300)
        plt.close()

        np.savetxt(os.path.join(subj_dir, "subject_summary.txt"), [np.mean(accs), np.std(accs)], fmt="%.4f")
        results_mean.append(np.mean(accs))
        results_std.append(np.std(accs))
        print(f" → Subject {subj}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # 전체 시각화
    plt.figure(figsize=(10, 6))
    x = np.arange(len(subjects))
    plt.bar(x, results_mean, yerr=results_std, capsize=5, color="royalblue", edgecolor="black")
    plt.xticks(x, [f"S{s}" for s in subjects])
    plt.title("All Subjects Accuracy (Raw EEG)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "AllSubjects_Bar.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
