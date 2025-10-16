# =========================================
#  EEGNet-Mel (32×128 Spectrogram)
# =========================================
import librosa, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from scipy.signal import butter, filtfilt
import mne, os, glob, re

# -------------------------
# 설정
# -------------------------
DATA_DIR = "./dataset/"
N_MELS, N_FFT, HOP = 32, 128, 64
BATCH, EPOCHS, LR, SEED = 16, 50, 1e-3, 42
torch.manual_seed(SEED); np.random.seed(SEED)

# -------------------------
# Band-pass 필터
# -------------------------
def butter_bandpass_filter(data, low, high, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data)

# -------------------------
# EEG → Mel-spectrogram 변환
# -------------------------
def eeg_to_mel(eeg, fs):
    S = np.abs(librosa.stft(eeg, n_fft=N_FFT, hop_length=HOP, window='hann'))**2
    mel_fb = librosa.filters.mel(sr=fs, n_fft=N_FFT, n_mels=N_MELS, fmin=8, fmax=30)
    mel = np.dot(mel_fb, S)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    img = cv2.resize(mel_db, (128, 32))  # (W, H)
    return img.astype(np.float32)

# -------------------------
# BCI IV 2b 데이터 로드 (C3)
# -------------------------
def load_bci2b_mel(path=DATA_DIR):
    files = sorted(glob.glob(os.path.join(path, "*T.gdf")))
    X_all, y_all, subj_ids = [], [], []
    for f in files:
        subj = int(re.findall(r"B0(\d)", f)[0])
        raw = mne.io.read_raw_gdf(f, preload=True)
        events, ev_dict = mne.events_from_annotations(raw)
        raw.pick_channels(["EEG:C3"])
        fs = int(raw.info["sfreq"])
        raw.filter(8., 30.)

        left, right = ev_dict.get("769"), ev_dict.get("770")
        mi, rest = [], []
        for e in events:
            if e[-1] in [left, right]:
                s, e_ = int(e[0]), int(e[0] + 4 * fs)
                if e_ <= len(raw.times):
                    mi.append(raw.get_data(start=s, stop=e_).squeeze())
                s, e_ = int(e[0] - 4 * fs), int(e[0])
                if s >= 0:
                    rest.append(raw.get_data(start=s, stop=e_).squeeze())
        if mi and rest:
            mi, rest = np.array(mi), np.array(rest)
            L = min(mi.shape[1], rest.shape[1])
            X = np.concatenate([mi[:, :L], rest[:, :L]], 0)
            y = np.concatenate([np.ones(len(mi)), np.zeros(len(rest))])
            X_all.append(X); y_all.append(y)
            subj_ids.extend([subj] * len(y))

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    subj_ids = np.array(subj_ids)
    print(f"총 샘플: {len(X)} | MI: {int(y.sum())} | Rest: {len(y)-int(y.sum())}")
    return X, y, subj_ids, fs

# -------------------------
# EEGNet (Lawhern et al., 2018)
# -------------------------
class EEGNet(nn.Module):
    def __init__(self, chans, samples, n_classes=2, dropout=0.5, kernLength=64, F1=8, D=2, F2=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1*D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), groups=F1*D, padding=(0, 8), bias=False),
            nn.Conv2d(F1*D, F2, 1, bias=False),
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

# -------------------------
# 학습 / 평가 루프
# -------------------------
def train_one_epoch(model, dl, opt, crit, dev):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    acc = correct / total
    print(f"Train Loss: {total_loss/len(dl):.4f}, Acc: {acc*100:.2f}%")
    return total_loss / len(dl)

@torch.no_grad()
def eval_acc(model, dl, dev):
    model.eval()
    correct, total = 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(dev), yb.to(dev)
        preds = model(xb).argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total

# -------------------------
# Main (10×10 CV)
# -------------------------
def main_mel():
    X_raw, y_raw, subj_ids, fs = load_bci2b_mel(DATA_DIR)
    X_mel = [eeg_to_mel(butter_bandpass_filter(x, 8, 30, fs), fs) for x in X_raw]
    X = torch.tensor(np.array(X_mel)[:, None, :, :], dtype=torch.float32)  # (B, 1, 32, 128)
    y = torch.tensor(y_raw, dtype=torch.long)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for subj in np.unique(subj_ids):
        print(f"\n=== Subject {subj} ===")
        mask = subj_ids == subj
        Xs, ys = X[mask], y[mask]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        accs = []
        for tr, te in skf.split(Xs, ys):
            tr_dl = DataLoader(TensorDataset(Xs[tr], ys[tr]), batch_size=BATCH, shuffle=True)
            te_dl = DataLoader(TensorDataset(Xs[te], ys[te]), batch_size=BATCH)
            model = EEGNet(chans=32, samples=128).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            crit = nn.CrossEntropyLoss()
            for ep in range(EPOCHS):
                train_one_epoch(model, tr_dl, opt, crit, dev)
            acc = eval_acc(model, te_dl, dev)
            accs.append(acc)
            print(f"Fold Test Acc: {acc*100:.2f}%")
        print(f"→ Subject {subj}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        results.append(np.mean(accs))
    print(f"\n전체 평균 정확도: {np.mean(results):.3f}")

if __name__ == "__main__":
    main_mel()
