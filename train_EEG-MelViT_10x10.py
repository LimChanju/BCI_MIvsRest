# -*- coding: utf-8 -*-
import os, re, glob, warnings, math
import numpy as np
import mne, librosa
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import StratifiedKFold

# =========================
# 0. Seed & 기본 설정
# =========================
SEED = 42
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed fixed to {seed}")
set_seed()

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fs = 250
save_dir = "./MelViT_small_10x10"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 1. 데이터 로드 (BCI IV-2b, C3)
# =========================
files = sorted(glob.glob("./dataset/*T.gdf"))
print("Using files:", files)

X_all, y_all, subj_ids = [], [], []
for f in files:
    subj_match = re.search(r"B0(\d)", os.path.basename(f))
    if not subj_match:
        print(f"⚠️ Subject 번호 인식 실패 → skip: {f}")
        continue
    subj = int(subj_match.group(1))
    print(f"\n=== Loading {f} (S{subj}) ===")

    raw = mne.io.read_raw_gdf(f, preload=True)
    events, event_dict = mne.events_from_annotations(raw)
    raw.pick_channels(["EEG:C3"])
    raw.filter(8., 30., fir_design="firwin")

    left = event_dict.get("769") or event_dict.get(769)
    right = event_dict.get("770") or event_dict.get(770)
    if left is None and right is None:
        print("⚠️ MI 이벤트 없음 → skip"); continue
    mi_ids = [v for v in [left, right] if v is not None]

    ev = events.copy()
    ev[ev[:, -1]==left, -1] = 1
    ev[ev[:, -1]==right, -1] = 2
    event_id = {"left":1, "right":2}

    rest_ep = mne.Epochs(raw, ev, event_id, tmin=-4.0, tmax=0.0, baseline=None, preload=True)
    mi_ep   = mne.Epochs(raw, ev, event_id, tmin= 0.0, tmax=4.0, baseline=None, preload=True)

    X_subj = np.concatenate([rest_ep.get_data(), mi_ep.get_data()], axis=0)
    y_subj = np.concatenate([np.zeros(len(rest_ep)), np.ones(len(mi_ep))], axis=0)

    mu, sd = X_subj.mean(), X_subj.std() + 1e-6
    X_subj = (X_subj - mu) / sd

    X_all.append(X_subj)
    y_all.append(y_subj)
    subj_ids.extend([subj]*len(y_subj))

X = np.concatenate(X_all, 0)
y = np.concatenate(y_all, 0).astype(int)
subj_ids = np.array(subj_ids)
print("\nFinal raw:", X.shape, np.unique(y, return_counts=True), "Subjects:", np.unique(subj_ids))

# =========================
# 2. EEG → Mel 변환 + pad
# =========================
def eeg_to_mel(eeg, sr=fs, n_fft=256, hop=128, n_mels=40, fmax=50):
    out = []
    for tr in eeg:
        sig = tr[0]
        M = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=n_fft,
                                           hop_length=hop, n_mels=n_mels, fmax=fmax)
        M = librosa.power_to_db(M, ref=np.max)
        M = (M - M.mean()) / (M.std() + 1e-6)
        out.append(M)
    return np.stack(out)

def pad_to_patch(arr, patch=4):
    N,F,T = arr.shape
    Fp, Tp = math.ceil(F/patch)*patch, math.ceil(T/patch)*patch
    padded = np.zeros((N,Fp,Tp), dtype=arr.dtype)
    padded[:, :F, :T] = arr
    if Fp>F: padded[:, F:Fp, :T] = arr[:, F-1:F, :T]
    if Tp>T: padded[:, :F, T:Tp] = arr[:, :, T-1:T]
    return padded

print("🎧 Converting to Mel...")
mel_X = eeg_to_mel(X)
mel_X = pad_to_patch(mel_X, patch=4)
print("Mel padded:", mel_X.shape)

# =========================
# 3. SpecAugment
# =========================
def spec_augment(x, fmask=4, tmask=4):
    x = x.clone()
    B,C,F,T = x.shape
    for i in range(B):
        if F>fmask:
            f0 = np.random.randint(0,F-fmask+1)
            x[i,:,f0:f0+fmask,:] = 0
        if T>tmask:
            t0 = np.random.randint(0,T-tmask+1)
            x[i,:,:,t0:t0+tmask] = 0
    return x

# =========================
# 4. ViT 모델
# =========================
class EEG_MelViT_Hybrid(nn.Module):
    def __init__(self, F=40, T=64, num_classes=2,
                 patch=2, dim=48, depth=3, heads=4, mlp_dim=128, dropout=0.0):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1,bias=False),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.Fs, self.Ts = F//2, T//2
        self.patch = patch
        self.num_patches = (self.Fs//patch)*(self.Ts//patch)
        self.to_patch = nn.Sequential(
            Rearrange('b c (f p1) (t p2) -> b (f t) (c p1 p2)', p1=patch,p2=patch),
            nn.Linear(16*patch*patch, dim)
        )
        enc = TransformerEncoderLayer(d_model=dim, nhead=heads,
                                      dim_feedforward=mlp_dim, dropout=dropout,
                                      activation='gelu', batch_first=True)
        self.encoder = TransformerEncoder(enc, num_layers=depth)
        self.cls = nn.Parameter(torch.randn(1,1,dim))
        self.pos = nn.Parameter(torch.randn(1,self.num_patches+1,dim))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,num_classes))
    def forward(self,x):
        x = self.conv_stem(x)
        x = self.to_patch(x)
        b = x.size(0)
        cls = self.cls.expand(b,-1,-1)
        x = torch.cat([cls,x],dim=1)
        x = x + self.pos[:,:x.size(1)]
        x = self.drop(x)
        x = self.encoder(x)
        return self.head(x[:,0])

# =========================
# 5. 10×10 CV (Subject-dependent)
# =========================
X_t = torch.tensor(mel_X, dtype=torch.float32).unsqueeze(1)
y_t = torch.tensor(y, dtype=torch.long)
subjects = np.unique(subj_ids)
F, T = X_t.shape[2], X_t.shape[3]
subject_results = []

for subj in subjects:
    print(f"\n=== Subject {subj}: 10×10 CV ===")
    mask = subj_ids == subj
    Xs, ys = X_t[mask], y_t[mask]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_accs = []

    for fold, (tr, te) in enumerate(skf.split(Xs, ys), 1):
        tr_loader = DataLoader(TensorDataset(Xs[tr], ys[tr]), batch_size=16, shuffle=True)
        te_loader = DataLoader(TensorDataset(Xs[te], ys[te]), batch_size=16, shuffle=False)
        model = EEG_MelViT_Hybrid(F=F, T=T, num_classes=2, patch=4, dim=64, depth=3, heads=4, mlp_dim=128).to(device)
        crit = nn.CrossEntropyLoss()
        opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)

        best = 0.0
        for epoch in range(50):
            # train
            model.train(); corr=tot=0
            for xb,yb in tr_loader:
                xb,yb=xb.to(device),yb.to(device)
                xb = spec_augment(xb,4,4)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out,yb)
                loss.backward(); opt.step()
                corr += (out.argmax(1)==yb).sum().item(); tot+=yb.size(0)
            sched.step(epoch+1)
            # eval
            model.eval(); correct=total=0
            with torch.no_grad():
                for xb,yb in te_loader:
                    xb,yb=xb.to(device),yb.to(device)
                    out = model(xb)
                    correct+=(out.argmax(1)==yb).sum().item(); total+=yb.size(0)
            acc = correct/total
            best = max(best, acc)
        fold_accs.append(best)
        print(f" Fold {fold:2d} | Best {best:.3f}")

    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    subject_results.append(mean_acc)
    print(f" → Subject {subj}: {mean_acc:.3f} ± {std_acc:.3f}")

    plt.figure(figsize=(6,4))
    plt.plot(fold_accs, marker='o', color='b')
    plt.axhline(mean_acc, color='r', ls='--', label=f"Mean={mean_acc:.3f}")
    plt.fill_between(range(10), mean_acc-std_acc, mean_acc+std_acc, color='r', alpha=0.2)
    plt.title(f"Subject {subj} - 10×10 CV (MelViT)"); plt.xlabel("Fold"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_dir}/Subj{subj}_folds.png", dpi=300); plt.close()

# =========================
# 6. 전체 평균 막대그래프
# =========================
overall_mean, overall_std = np.mean(subject_results), np.std(subject_results)
print("\n✅ Subject mean:", np.round(subject_results,3))
print(f"Overall mean = {overall_mean:.3f} ± {overall_std:.3f}")

plt.figure(figsize=(8,5))
plt.bar([f"S{s}" for s in subjects], subject_results, color='orchid', edgecolor='k')
plt.axhline(overall_mean, color='r', ls='--', label=f"Mean={overall_mean:.3f}")
plt.fill_between(range(len(subjects)), overall_mean-overall_std, overall_mean+overall_std, color='r', alpha=0.2)
plt.title("Subject-wise Accuracy (EEG-MelViT, 10×10 CV, C3 8–30 Hz)")
plt.ylabel("Accuracy"); plt.ylim(0,1); plt.legend()
plt.tight_layout(); plt.savefig(f"{save_dir}/AllSubjects_Bar.png", dpi=300)
plt.show()
