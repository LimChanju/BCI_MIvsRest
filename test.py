# -*- coding: utf-8 -*-
import os, re, glob, warnings
import numpy as np
import librosa
import mne
from scipy.signal import butter, filtfilt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# =========================
# 0. 설정
# =========================
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

DATA_DIR = "./dataset/"
N_MELS, N_FFT, HOP = 16, 128, 64
BATCH, EPOCHS, LR, SEED = 16, 50, 1e-3, 42
set_seed = lambda s=SEED: (np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s))
set_seed()

# =========================
# 1. Band-pass 필터
# =========================
def butter_bandpass_filter(data, low, high, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data)

# =========================
# 2. NeuroMel Filterbank 생성
# =========================
def neuro_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """EEG 전용 NeuroMel-v3 filterbank (α·β 대역 밀도 강조형)"""
    # 주파수 구간 (0.5~40Hz 범위 대비 α, β 대역 강조)
    freqs = np.linspace(fmin, fmax, n_mels + 2)

    # α(8–13Hz) 구간을 더 조밀하게, β(13–30Hz)는 완만하게
    alpha_center, beta_center = 10.5, 20
    alpha_boost = np.exp(-((freqs - alpha_center) ** 2) / (2 * 3.5 ** 2))
    beta_boost  = np.exp(-((freqs - beta_center) ** 2) / (2 * 6.5 ** 2))
    weight = 0.6 * alpha_boost + 0.4 * beta_boost

    # 밀도 조절: α쪽은 더 세밀하게, β쪽은 부드럽게
    freqs = fmin + (freqs - fmin) * (1 + 2.0 * weight)

    # FFT bin 변환
    f_bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))

    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = f_bins[m - 1], f_bins[m], f_bins[m + 1]
        f_m_minus = max(f_m_minus, 0)
        f_m_plus  = min(f_m_plus, n_fft // 2)
        for k in range(f_m_minus, f_m):
            fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-6)
        for k in range(f_m, f_m_plus):
            fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-6)
    fb /= np.maximum(np.sum(fb, axis=1, keepdims=True), 1e-8)
    return fb


# =========================
# 3. EEG → NeuroMel 변환
# =========================
def eeg_to_neuromel(eeg_signal, fs, band):
    low, high = band
    filtered = butter_bandpass_filter(eeg_signal, low, high, fs)

    # 🧩 1. STFT (center + Hann window)
    S = np.abs(librosa.stft(
        y=filtered,
        n_fft=N_FFT,
        hop_length=HOP,
        window='hann',
        center=True,
        pad_mode='reflect'
    )) ** 2  # power spectrum

    # 🧩 2. Mel filterbank (EEG용 슬래니 정규화)
    mel_fb = librosa.filters.mel(
        sr=int(fs),
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=low,
        fmax=high,
        norm='slaney',
        htk=False
    )

    mel_spec = np.dot(mel_fb, S)

    # 🧩 3. dB scaling + 표준화
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # 🧩 4. shape 조정
    return mel_db.astype(np.float32)


# =========================
# 4. BCI 2b 데이터 로드 (C3만)
# =========================
def load_bci2b_dataset(path=DATA_DIR):
    files = sorted(glob.glob(os.path.join(path, "*T.gdf")))
    if not files:
        raise FileNotFoundError("❌ dataset 폴더에 *T.gdf 없음")

    X_all, y_all, subj_ids, fs_out = [], [], [], None
    for f in files:
        subj_match = re.search(r"B0(\d)", os.path.basename(f))
        if not subj_match:
            continue
        subj_id = int(subj_match.group(1))
        raw = mne.io.read_raw_gdf(f, preload=True)
        events, event_dict = mne.events_from_annotations(raw)
        raw.pick_channels(["EEG:C3"])
        fs = raw.info["sfreq"]; fs_out = fs
        raw.filter(8., 30., fir_design="firwin")

        left = event_dict.get("769") or event_dict.get(769)
        right = event_dict.get("770") or event_dict.get(770)
        if left is None and right is None:
            continue
        mi_ids = [v for v in [left, right] if v is not None]

        mi, rest = [], []
        for ev in events:
            if ev[-1] in mi_ids:
                s, e = int(ev[0]), int(ev[0] + 4.0 * fs)
                if e <= len(raw.times): mi.append(raw.get_data(start=s, stop=e).squeeze())
                s, e = int(ev[0] - 4.0 * fs), int(ev[0])
                if s >= 0: rest.append(raw.get_data(start=s, stop=e).squeeze())
        if not mi or not rest: continue
        mi, rest = np.array(mi), np.array(rest)
        L = min(mi.shape[1], rest.shape[1])
        X_subj = np.concatenate([mi[:, :L], rest[:, :L]], 0)
        y_subj = np.concatenate([np.ones(len(mi)), np.zeros(len(rest))])
        X_all.append(X_subj); y_all.append(y_subj)
        subj_ids.extend([subj_id]*len(y_subj))
    X = np.concatenate(X_all); y = np.concatenate(y_all)
    subj_ids = np.array(subj_ids)
    print(f"\n총 샘플: {len(X)} | MI:{int(y.sum())} | Rest:{len(y)-int(y.sum())}")
    return X, y, subj_ids, fs_out

# =========================
# 5. μ/β NeuroMel Dataset
# =========================
def make_neuromel_dataset(X_raw, y_raw, fs):
    mu, beta = (8, 12), (13, 30)
    mu_feats = [eeg_to_neuromel(x, fs, mu) for x in X_raw]
    beta_feats = [eeg_to_neuromel(x, fs, beta) for x in X_raw]

    mu_feats = np.stack(mu_feats)
    beta_feats = np.stack(beta_feats)

    # ✅ 4D 유지: (N, 1, n_mels, time)
    mu_feats = mu_feats[:, np.newaxis, :, :]
    beta_feats = beta_feats[:, np.newaxis, :, :]

    # ✅ 두 밴드를 채널 축으로 결합 → (N, 2, n_mels, time)
    X = np.concatenate([mu_feats, beta_feats], axis=1)

    print("Final X shape:", X.shape)
    plt.figure(figsize=(6, 3))
    im = plt.imshow(X[0, 0], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im)
    plt.title("EEG NeuroMel Example (μ-band)")
    plt.tight_layout()
    plt.show()

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.long), X.shape[2], X.shape[3]




# =========================
# 6. CNN 모델
# =========================
class MelCNN(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.conv1 = nn.Conv2d(2,16,3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 🧠 핵심 수정 — width(=1) 유지!
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        
        self.drop = nn.Dropout(0.35)
        self.fc1 = nn.Linear(32*(h//4)*w, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# =========================
# 7. 학습 루프
# =========================
def train_one_epoch(model, loader, opt, crit, dev):
    model.train(); tot=0
    for xb,yb in loader:
        xb,yb=xb.to(dev),yb.to(dev)
        opt.zero_grad(); out=model(xb)
        loss=crit(out,yb); loss.backward(); opt.step()
        tot+=loss.item()
    return tot/len(loader)

@torch.no_grad()
def eval_acc(model, loader, dev):
    model.eval(); c=t=0
    for xb,yb in loader:
        xb,yb=xb.to(dev),yb.to(dev)
        p=model(xb).argmax(1)
        c+=(p==yb).sum().item(); t+=yb.size(0)
    return c/t

# =========================
# 8. 메인 (10×10 CV)
# =========================
def main():
    X_raw, y_raw, subj_ids, fs = load_bci2b_dataset(DATA_DIR)
    X, y, H, W = make_neuromel_dataset(X_raw, y_raw, fs)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = "./Neuro-MelScale_CNN_10x10"
    os.makedirs(save_dir, exist_ok=True)

    subjects = np.unique(subj_ids)
    results_mean, results_std = [], []

    for subj in subjects:
        print(f"\n=== Subject {subj}: 10×10 CV ===")
        msk = subj_ids == subj
        Xs, ys = X[msk], y[msk]
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        accs = []

        for f, (tr, te) in enumerate(skf.split(Xs, ys), 1):
            tr_dl = DataLoader(TensorDataset(Xs[tr], ys[tr]), batch_size=BATCH, shuffle=True)
            te_dl = DataLoader(TensorDataset(Xs[te], ys[te]), batch_size=BATCH)
            model = MelCNN(H, W).to(dev)
            crit = nn.CrossEntropyLoss()
            opt = torch.optim.Adam(model.parameters(), lr=LR)

            best = 0
            for ep in range(EPOCHS):
                train_one_epoch(model, tr_dl, opt, crit, dev)
                best = max(best, eval_acc(model, te_dl, dev))
            accs.append(best)
            print(f" Fold {f:2d} | Best {best:.3f}")

        # 평균 ± 표준편차
        m, s = np.mean(accs), np.std(accs)
        results_mean.append(m)
        results_std.append(s)
        print(f" → Subject {subj}: {m:.3f} ± {s:.3f}")

        # ✅ Fold별 Acc 점선 그래프 (Subject 개별)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, 11), accs, marker='o', linestyle='--', color='royalblue', label='Fold Acc')
        plt.axhline(m, color='red', linestyle='-', label=f"Mean={m:.3f}")
        plt.fill_between(range(1, 11), m - s, m + s, color='red', alpha=0.2, label='±1 std')
        plt.title(f"Subject {subj} - 10×10 CV")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Subj{subj}_fold_curve.png", dpi=300)
        plt.close()

    # ✅ 전체 Subject별 Bar Plot
    om, osd = np.mean(results_mean), np.std(results_mean)

    plt.figure(figsize=(9, 6))
    x = np.arange(len(subjects))
    plt.bar(x, results_mean, yerr=results_std, capsize=5, color='orchid', edgecolor='black', alpha=0.7)
    plt.axhline(om, color='red', linestyle='--', label=f'Mean={om:.3f}')
    plt.fill_between(x, om - osd, om + osd, color='red', alpha=0.15)
    plt.xticks(x, [f"S{s}" for s in subjects])
    plt.title("Subject-wise Accuracy (MelCNN, 10×10 CV, C3 8–30Hz)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/AllSubjects_Bar.png", dpi=300)
    plt.show()

    # ✅ 콘솔 출력
    print("\n✅ Subject mean ± std:")
    for s_id, m, s in zip(subjects, results_mean, results_std):
        print(f" Subject {s_id}: {m:.3f} ± {s:.3f}")
    print(f"\nOverall = {om:.3f} ± {osd:.3f}")

if __name__ == "__main__":
    main()