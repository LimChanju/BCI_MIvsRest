# -*- coding: utf-8 -*-
import os, re, glob, warnings
import numpy as np
import mne, librosa
from scipy.signal import butter, filtfilt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# =========================
# 0. 하이퍼파라미터
# =========================
DATA_DIR = "./dataset/"
N_MELS, N_FFT, HOP = 16, 128, 64
BATCH, EPOCHS, LR, SEED = 16, 50, 1e-3, 42
set_seed = lambda s=SEED: (np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s))
set_seed()

# =========================
# 1. Band-pass + Mel 변환
# =========================
def butter_bandpass_filter(data, low, high, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data)

def eeg_to_mel(eeg_signal, fs, band):
    low, high = band
    filtered = butter_bandpass_filter(eeg_signal, low, high, fs)
    mel = librosa.feature.melspectrogram(
        y=filtered, sr=int(fs), n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=low, fmax=high, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
    return mel_db.astype(np.float32)

# =========================
# 2. BCI 2b 데이터 로드 (C3, −4~0 / 0~4)
# =========================
def load_bci2b_dataset(path=DATA_DIR):
    files = sorted(glob.glob(os.path.join(path, "*T.gdf")))
    if not files: raise FileNotFoundError("❌ dataset 폴더에 *T.gdf 없음")

    X_all, y_all, subj_ids, fs_out = [], [], [], None
    for f in files:
        subj_match = re.search(r"B0(\d)", os.path.basename(f))
        if not subj_match:
            print(f"⚠️ 파일명 인식 불가 → skip: {f}")
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
            print(f"⚠️ {f}: MI 이벤트 없음 → skip"); continue
        mi_ids = [v for v in [left, right] if v is not None]

        mi, rest = [], []
        for ev in events:
            if ev[-1] in mi_ids:
                s, e = int(ev[0] + 0.0 * fs), int(ev[0] + 4.0 * fs)
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
# 3. μ/β Mel 변환 Dataset
# =========================
def make_mel_dataset(X_raw, y_raw, fs):
    mu, beta = (8,12), (13,30)
    mu_feats = [eeg_to_mel(x, fs, mu) for x in X_raw]
    beta_feats = [eeg_to_mel(x, fs, beta) for x in X_raw]
    mu_feats, beta_feats = np.stack(mu_feats), np.stack(beta_feats)
    h, w = min(mu_feats.shape[-2], beta_feats.shape[-2]), min(mu_feats.shape[-1], beta_feats.shape[-1])
    X = np.stack([mu_feats[:, :h, :w], beta_feats[:, :h, :w]], 1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y_raw, dtype=torch.long), h, w

# =========================
# 4. CNN 모델 (μ/β 2채널)
# =========================
class MelCNN(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.conv1 = nn.Conv2d(2,16,3,padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3,padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2); self.drop = nn.Dropout(0.35)
        self.fc1 = nn.Linear(32*(h//4)*(w//4),64); self.fc2 = nn.Linear(64,2)
    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1); x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# =========================
# 5. 학습 루프
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
# 6. 메인 (10×10 CV)
# =========================
def main():
    X_raw,y_raw,subj_ids,fs = load_bci2b_dataset(DATA_DIR)
    X,y,H,W = *make_mel_dataset(X_raw,y_raw,fs), 
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir="./MelCNN_10x10"; os.makedirs(save_dir,exist_ok=True)
    subjects=np.unique(subj_ids); results=[]

    for subj in subjects:
        print(f"\n=== Subject {subj}: 10×10 CV ===")
        msk=subj_ids==subj; Xs,ys=X[msk],y[msk]
        skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=SEED)
        accs=[]
        for f,(tr,te) in enumerate(skf.split(Xs,ys),1):
            tr_dl=DataLoader(TensorDataset(Xs[tr],ys[tr]),batch_size=BATCH,shuffle=True)
            te_dl=DataLoader(TensorDataset(Xs[te],ys[te]),batch_size=BATCH)
            model=MelCNN(H,W).to(dev)
            crit=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=LR)
            best=0
            for ep in range(EPOCHS):
                train_one_epoch(model,tr_dl,opt,crit,dev)
                best=max(best,eval_acc(model,te_dl,dev))
            accs.append(best); print(f" Fold {f:2d} | Best {best:.3f}")
        m,s=np.mean(accs),np.std(accs); results.append(m)
        print(f" → Subject {subj}: {m:.3f} ± {s:.3f}")
        plt.figure(figsize=(6,4))
        plt.plot(accs,marker='o',color='blue')
        plt.axhline(m,color='r',ls='--',label=f'Mean {m:.3f}')
        plt.fill_between(range(10),m-s,m+s,color='r',alpha=0.2)
        plt.title(f"Subject {subj} - 10×10 CV"); plt.xlabel("Fold"); plt.ylabel("Acc")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{save_dir}/Subj{subj}_folds.png",dpi=300); plt.close()

    om,osd=np.mean(results),np.std(results)
    print("\n✅ Subject mean:",np.round(results,3))
    print(f"Overall = {om:.3f} ± {osd:.3f}")
    plt.figure(figsize=(8,5))
    plt.bar([f"S{s}" for s in subjects],results,color='orchid',edgecolor='k')
    plt.axhline(om,color='r',ls='--',label=f"Mean={om:.3f}")
    plt.fill_between(range(len(subjects)),om-osd,om+osd,color='r',alpha=0.2)
    plt.title("Subject-wise Accuracy (MelCNN, 10×10 CV, C3 8–30Hz)")
    plt.ylabel("Accuracy"); plt.ylim(0,1); plt.legend()
    plt.tight_layout(); plt.savefig(f"{save_dir}/AllSubjects_Bar.png",dpi=300)
    plt.show()

if __name__=="__main__":
    main()
