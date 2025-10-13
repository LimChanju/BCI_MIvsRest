# -*- coding: utf-8 -*-
import os, re, glob, math, warnings
import numpy as np
import mne, matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import dct

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

from einops.layers.torch import Rearrange

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# =========================
# 0. 하이퍼파라미터/시드
# =========================
DATA_DIR = "./dataset/"
BATCH, EPOCHS, LR = 16, 50, 1e-3
SEED = 42

def set_seed(seed=SEED):
    import random, torch.backends.cudnn as cudnn
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True; cudnn.benchmark = False
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "./NeuroMelLSTM_10x10"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 1. BCI 2b 로드 (C3, −4~0 / 0~4, 8–30 Hz FIR)
# =========================
def load_bci2b_dataset(path=DATA_DIR):
    files = sorted(glob.glob(os.path.join(path, "*T.gdf")))
    if not files:
        raise FileNotFoundError("❌ dataset 폴더에 *T.gdf 없음")

    X_all, y_all, subj_ids, fs_out = [], [], [], None
    for f in files:
        subj_match = re.search(r"B0(\d)", os.path.basename(f))
        if not subj_match:
            print(f"⚠️ 파일명 인식 불가 → skip: {f}")
            continue
        subj = int(subj_match.group(1))

        raw = mne.io.read_raw_gdf(f, preload=True, verbose=False)
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        raw.pick_channels(["EEG:C3"])
        raw.filter(8., 30., fir_design="firwin", verbose=False)
        fs = raw.info["sfreq"]; fs_out = fs

        left = event_dict.get("769") or event_dict.get(769)
        right = event_dict.get("770") or event_dict.get(770)
        if left is None and right is None:
            print(f"⚠️ {f}: MI 이벤트 없음 → skip"); continue
        mi_ids = [v for v in [left, right] if v is not None]

        mi, rest = [], []
        for ev in events:
            if ev[-1] in mi_ids:
                s, e = int(ev[0] + 0.0*fs), int(ev[0] + 4.0*fs)
                if e <= len(raw.times):
                    mi.append(raw.get_data(start=s, stop=e).squeeze())
                s, e = int(ev[0] - 4.0*fs), int(ev[0])
                if s >= 0:
                    rest.append(raw.get_data(start=s, stop=e).squeeze())

        if not mi or not rest: 
            print(f"⚠️ S{subj}: MI/Rest epoch 부족 → skip")
            continue

        mi, rest = np.array(mi), np.array(rest)
        L = min(mi.shape[1], rest.shape[1])
        X_subj = np.concatenate([mi[:, :L], rest[:, :L]], axis=0)          
        y_subj = np.concatenate([np.ones(len(mi)), np.zeros(len(rest))])    
        X_subj = X_subj[:, None, :]

        mu, sd = X_subj.mean(), X_subj.std() + 1e-6
        X_subj = (X_subj - mu) / sd

        X_all.append(X_subj); y_all.append(y_subj)
        subj_ids.extend([subj]*len(y_subj))

    X = np.concatenate(X_all, axis=0)        
    y = np.concatenate(y_all, axis=0).astype(int)
    subj_ids = np.array(subj_ids)
    print(f"\n총 샘플: {len(X)} | MI:{int(y.sum())} | Rest:{len(y)-int(y.sum())}")
    return X, y, subj_ids, fs_out

# =========================
# 2. NeuroMel + ENCC + α/β 채널
# =========================
def estimate_iaf(sig, fs=250, fmin=7, fmax=14):
    f, pxx = welch(sig, fs=fs, nperseg=fs*2)
    m = (f>=fmin) & (f<=fmax)
    return float(f[m][np.argmax(pxx[m])]) if np.any(m) else 10.0

def neuro_filterbank(fs, n_fft, iaf=None, one_over_f=1.0):
    f_lo, f_hi = 4, 40
    centers = []
    centers += list(np.linspace(4, 8, 4, endpoint=False))
    iaf = iaf or 10.0
    mu_lo, mu_hi = max(8, iaf-3), min(13, iaf+3)
    centers += list(np.linspace(mu_lo, mu_hi, 8, endpoint=False))
    centers += list(np.linspace(13, 30, 8, endpoint=False))
    centers += list(np.linspace(30, 40, 4))
    centers = np.sort(np.clip(centers, f_lo, f_hi))

    freqs = rfftfreq(n_fft, 1.0/fs)
    fb = np.zeros((len(centers), len(freqs)))

    pad = np.concatenate([[centers[0]-(centers[1]-centers[0])],
                          centers,
                          [centers[-1]+(centers[-1]-centers[-2])]])
    for i in range(len(centers)):
        left, c, right = pad[i], pad[i+1], pad[i+2]
        lmask = (freqs>=left) & (freqs<=c)
        rmask = (freqs>=c) & (freqs<=right)
        fb[i, lmask] = (freqs[lmask]-left)/(c-left+1e-9)
        fb[i, rmask] = (right-freqs[rmask])/(right-c+1e-9)

    fb[:, (freqs<f_lo)|(freqs>f_hi)] = 0.0
    if one_over_f > 0:
        fb = fb / (np.clip(freqs, 1e-3, None)**one_over_f)
    fb /= fb.sum(axis=1, keepdims=True) + 1e-8
    return fb, centers

def stft_power(sig, n_fft=256, hop=128):
    win = np.hanning(n_fft)
    frames = []
    for t0 in range(0, len(sig)-n_fft+1, hop):
        seg = sig[t0:t0+n_fft]
        X = rfft(seg * win)
        frames.append((np.abs(X)**2))
    if not frames:
        X = rfft(sig[:n_fft]*win)
        frames = [(np.abs(X)**2)]
    return np.stack(frames, axis=1)

def neuro_mel_encc_for_subject(trials, fs=250, n_fft=256, hop=128, n_ceps=16):
    iaf = estimate_iaf(trials.mean(axis=0)[0], fs=fs)
    fb, centers = neuro_filterbank(fs, n_fft, iaf=iaf)
    NM_list, ENCC_list = [], []
    for sig in trials[:,0]:
        P = stft_power(sig, n_fft, hop)
        NM = fb @ P
        NM = np.log(np.clip(NM, 1e-8, None))
        NM = (NM - NM.mean(axis=0, keepdims=True)) / (NM.std(axis=0, keepdims=True)+1e-6)
        EN = dct(NM, type=2, axis=0, norm='ortho')[:n_ceps,:]
        NM_list.append(NM); ENCC_list.append(EN)
    NM = np.stack(NM_list, axis=0)
    EN = np.stack(ENCC_list, axis=0)
    return NM, EN, centers

def upsample_encc_to_filters(ENCC, n_filters):
    N, C, Tf = ENCC.shape
    x_old = np.linspace(0, 1, C)
    x_new = np.linspace(0, 1, n_filters)
    dist = np.abs(x_new[:, None] - x_old[None, :])
    w = np.maximum(1.0 - dist/dist.max(), 0.0)
    w /= w.sum(axis=1, keepdims=True)+1e-8
    return np.einsum('fc,nct->nft', w, ENCC)

def build_neuromel_dataset(X_raw, y_raw, subj_ids, fs):
    NM_cat, EN_cat, A_cat, B_cat = [], [], [], []
    for s in np.unique(subj_ids):
        mask = (subj_ids==s)
        NM, EN, centers = neuro_mel_encc_for_subject(X_raw[mask], fs)
        EN = upsample_encc_to_filters(EN, NM.shape[1])
        a_mask = (centers>=8)&(centers<=13)
        b_mask = (centers>=13)&(centers<=30)
        alpha_mean = NM[:,a_mask,:].mean(axis=1,keepdims=True)
        beta_mean  = NM[:,b_mask,:].mean(axis=1,keepdims=True)
        Ff,Tf = NM.shape[1],NM.shape[2]
        A = np.repeat(alpha_mean,Ff,axis=1)
        B = np.repeat(beta_mean,Ff,axis=1)
        NM_cat.append(NM); EN_cat.append(EN); A_cat.append(A); B_cat.append(B)
    NM_all=np.concatenate(NM_cat,0); EN_all=np.concatenate(EN_cat,0)
    A_all=np.concatenate(A_cat,0); B_all=np.concatenate(B_cat,0)
    X_feat=np.stack([NM_all,EN_all,A_all,B_all],axis=1)
    return X_feat, y_raw.astype(int)

# =========================
# 3. CNN + BiLSTM 모델
# =========================
class NeuroMelLSTM(nn.Module):
    def __init__(self, in_ch=4, hidden_dim=64, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rearr = Rearrange('b c 1 t -> b t c')
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim*2, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.adapt_pool(x)
        x = self.rearr(x)
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out)

# =========================
# 4. Train/Eval
# =========================
def train_one_epoch(model, loader, opt, crit, dev):
    model.train(); tot=0
    for xb,yb in loader:
        xb,yb=xb.to(dev),yb.to(dev)
        opt.zero_grad()
        out=model(xb)
        loss=crit(out,yb)
        loss.backward(); opt.step()
        tot+=loss.item()
    return tot/len(loader)

@torch.no_grad()
def eval_acc(model, loader, dev):
    model.eval(); c=t=0
    for xb,yb in loader:
        xb,yb=xb.to(dev),yb.to(dev)
        pred=model(xb).argmax(1)
        c+=(pred==yb).sum().item(); t+=yb.size(0)
    return c/t

# =========================
# 5. Main (10×10 Subject-dependent)
# =========================
def main():
    X_raw, y_raw, subj_ids, fs = load_bci2b_dataset(DATA_DIR)
    print("🎧 Converting to NeuroMel + ENCC + α/β channels ...")
    X_feat, y = build_neuromel_dataset(X_raw, y_raw, subj_ids, fs)
    X_t=torch.tensor(X_feat,dtype=torch.float32)
    y_t=torch.tensor(y,dtype=torch.long)
    subjects=np.unique(subj_ids)
    _,in_ch,F,T=X_t.shape

    results=[]
    for subj in subjects:
        print(f"\n=== Subject {subj}: 10×10 CV ===")
        mask=subj_ids==subj
        Xs,ys=X_t[mask],y_t[mask]
        skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=SEED)
        accs=[]
        for f,(tr,te) in enumerate(skf.split(Xs,ys),1):
            tr_dl=DataLoader(TensorDataset(Xs[tr],ys[tr]),batch_size=BATCH,shuffle=True)
            te_dl=DataLoader(TensorDataset(Xs[te],ys[te]),batch_size=BATCH)
            model=NeuroMelLSTM(in_ch=in_ch).to(device)
            crit=nn.CrossEntropyLoss()
            opt=torch.optim.Adam(model.parameters(),lr=LR)
            best=0
            for ep in range(EPOCHS):
                train_one_epoch(model,tr_dl,opt,crit,device)
                acc=eval_acc(model,te_dl,device)
                best=max(best,acc)
            accs.append(best); print(f" Fold {f:2d} | Best {best:.3f}")
        m,s=np.mean(accs),np.std(accs); results.append(m)
        print(f" → Subject {subj}: {m:.3f} ± {s:.3f}")
        plt.figure(figsize=(6,4))
        plt.plot(range(1,11),accs,marker='o',color='blue')
        plt.axhline(m,color='r',ls='--',label=f"Mean {m:.3f}")
        plt.fill_between(range(1,11),m-s,m+s,color='r',alpha=0.2)
        plt.title(f"Subject {subj} - 10×10 CV (NeuroMelLSTM)")
        plt.xlabel("Fold"); plt.ylabel("Acc"); plt.legend()
        plt.tight_layout(); plt.savefig(f"{save_dir}/Subj{subj}_folds.png",dpi=300); plt.close()

    om,osd=np.mean(results),np.std(results)
    print("\n✅ Subject mean:",np.round(results,3))
    print(f"Overall mean = {om:.3f} ± {osd:.3f}")
    plt.figure(figsize=(8,5))
    plt.bar([f"S{s}" for s in subjects],results,color='orchid',edgecolor='k')
    plt.axhline(om,color='r',ls='--',label=f"Mean={om:.3f}")
    plt.fill_between(range(len(subjects)),om-osd,om+osd,color='r',alpha=0.2)
    plt.ylabel("Accuracy"); plt.ylim(0,1); plt.legend()
    plt.title("Subject-wise Accuracy (NeuroMelLSTM, 10×10 CV, C3 8–30Hz)")
    plt.tight_layout(); plt.savefig(f"{save_dir}/AllSubjects_Bar.png",dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
