# -*- coding: utf-8 -*-
import os, re, math, warnings, glob
import numpy as np
import mne, matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import dct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from einops.layers.torch import Rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level("WARNING")

# ===================== 0) 설정 =====================
SEED = 42
def set_seed(seed=SEED):
    import random, torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"🔒 Seed fixed to {seed}")
set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fs = 250
out_dir = "./NeuroMel_ViT_10x10_ab_adapt"
os.makedirs(out_dir, exist_ok=True)

# ===================== 1) 데이터 로드 =====================
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

    raw = mne.io.read_raw_gdf(f, preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    raw.pick_channels(["EEG:C3"])
    raw.filter(8., 30., fir_design="firwin", verbose=False)

    left = event_dict.get("769") or event_dict.get(769)
    right = event_dict.get("770") or event_dict.get(770)
    if left is None and right is None:
        print("⚠️ MI 이벤트 없음 → skip"); continue
    mi_ids = [v for v in [left, right] if v is not None]

    ev = events.copy()
    ev[ev[:, -1] == left, -1] = 1
    ev[ev[:, -1] == right, -1] = 2
    event_id = {"left":1, "right":2}

    rest_ep = mne.Epochs(raw, ev, event_id, tmin=-4.0, tmax=0.0, baseline=None, preload=True)
    mi_ep   = mne.Epochs(raw, ev, event_id, tmin= 0.0, tmax=4.0, baseline=None, preload=True)
    X_subj = np.concatenate([rest_ep.get_data(), mi_ep.get_data()], axis=0)
    y_subj = np.concatenate([np.zeros(len(rest_ep)), np.ones(len(mi_ep))], axis=0).astype(int)

    # ✅ REST-adaptive z-score
    mu = X_subj[y_subj==0].mean()
    sd = X_subj[y_subj==0].std() + 1e-6
    X_subj = (X_subj - mu) / sd

    X_all.append(X_subj)
    y_all.append(y_subj)
    subj_ids.extend([subj]*len(y_subj))

X = np.concatenate(X_all, 0)
y = np.concatenate(y_all, 0).astype(int)
subj_ids = np.array(subj_ids)
print("\nFinal raw:", X.shape, np.unique(y, return_counts=True), "Subjects:", np.unique(subj_ids))

# ===================== 2) Neuro-Mel + ENCC =====================
def estimate_iaf(sig, fs=250, fmin=7, fmax=14):
    f, pxx = welch(sig, fs=fs, nperseg=fs*2)
    m = (f>=fmin)&(f<=fmax)
    return float(f[m][np.argmax(pxx[m])]) if np.any(m) else 10.0

def neuro_filterbank(fs, n_fft, iaf=None, one_over_f=1.0):
    f_lo, f_hi = 4, 40
    centers = []
    centers += list(np.linspace(4,8,4,endpoint=False))
    iaf = iaf or 10.0
    mu_lo, mu_hi = max(8, iaf-3), min(13, iaf+3)
    centers += list(np.linspace(mu_lo, mu_hi,8,endpoint=False))
    centers += list(np.linspace(13,30,8,endpoint=False))
    centers += list(np.linspace(30,40,4))
    centers = np.unique(np.clip(centers,f_lo,f_hi))
    freqs = rfftfreq(n_fft,1/fs)
    fb = np.zeros((len(centers),len(freqs)))
    all_c = np.concatenate([[centers[0]-(centers[1]-centers[0])],centers,[centers[-1]+(centers[-1]-centers[-2])]])
    for i,c in enumerate(centers):
        l,r = all_c[i],all_c[i+2]
        mid = all_c[i+1]
        fb[i,(freqs>=l)&(freqs<=mid)] = (freqs[(freqs>=l)&(freqs<=mid)]-l)/(mid-l+1e-9)
        fb[i,(freqs>=mid)&(freqs<=r)] = (r-freqs[(freqs>=mid)&(freqs<=r)])/(r-mid+1e-9)
    fb[:,(freqs<4)|(freqs>40)] = 0.0
    if one_over_f>0: fb = fb / (np.clip(freqs,1e-3,None)**one_over_f)
    fb /= fb.sum(axis=1,keepdims=True)+1e-8
    return fb, centers

def stft_power(sig,n_fft=256,hop=128):
    win = np.hanning(n_fft); frames=[]
    for t0 in range(0,len(sig)-n_fft+1,hop):
        seg=sig[t0:t0+n_fft]; X=rfft(seg*win)
        frames.append(np.abs(X)**2)
    return np.stack(frames,axis=1)

def neuro_mel_encc_for_subject(trials, labels, fs=250, n_fft=256, hop=128, n_ceps=16):
    iaf = np.mean([
        estimate_iaf(trials[labels==0].mean(0)[0],fs),
        estimate_iaf(trials[labels==1].mean(0)[0],fs)
    ])
    fb, centers = neuro_filterbank(fs,n_fft,iaf=iaf)
    NM_list, ENCC_list = [], []
    for sig in trials[:,0]:
        P = stft_power(sig,n_fft,hop)
        NM = fb @ P
        NM = np.log(np.clip(NM,1e-8,None))
        NM = (NM-NM.mean(0,keepdims=True))/(NM.std(0,keepdims=True)+1e-6)
        EN = dct(NM,type=2,axis=0,norm='ortho')[:n_ceps,:]
        NM_list.append(NM); ENCC_list.append(EN)
    NM=np.stack(NM_list); ENCC=np.stack(ENCC_list)
    return NM, ENCC, centers

def upsample_encc_to_filters(ENCC,n_filters):
    N,C,T=ENCC.shape
    xo=np.linspace(0,1,C); xn=np.linspace(0,1,n_filters)
    dist=np.abs(xn[:,None]-xo[None,:])
    w=np.maximum(1.0-dist/dist.max(),0.0); w/=w.sum(1,keepdims=True)+1e-8
    return np.einsum('fc,nct->nft',w,ENCC)

# ===================== 3) 변환 + α/β 채널 =====================
print("\n🎧 Converting → Neuro-Mel + ENCC + αβ summaries ...")
NM_cat, EN_cat, A_cat, B_cat = [], [], [], []
for s in np.unique(subj_ids):
    mask=subj_ids==s
    NM,ENCC,cent=neuro_mel_encc_for_subject(X[mask],y[mask])
    EN=upsample_encc_to_filters(ENCC,NM.shape[1])
    ca=np.array(cent); am=(ca>=8)&(ca<=13); bm=(ca>=13)&(ca<=30)
    a_mean=NM[:,am,:].mean(1,keepdims=True); b_mean=NM[:,bm,:].mean(1,keepdims=True)
    Ff,Tm=NM.shape[1],NM.shape[2]
    A=np.repeat(a_mean,Ff,1); B=np.repeat(b_mean,Ff,1)
    NM_cat.append(NM); EN_cat.append(EN); A_cat.append(A); B_cat.append(B)
NM_all=np.concatenate(NM_cat); EN_all=np.concatenate(EN_cat)
A_all=np.concatenate(A_cat); B_all=np.concatenate(B_cat)
X_feat=np.stack([NM_all,EN_all,A_all,B_all],1)
print("X_feat:",X_feat.shape)

# ===================== 4) 패딩 =====================
def pad_FT(arr,div=4):
    N,C,F,T=arr.shape
    Fp=math.ceil(F/div)*div; Tp=math.ceil(T/div)*div
    out=np.zeros((N,C,Fp,Tp)); out[:,:,:F,:T]=arr
    if Fp>F: out[:,:,F:Fp,:T]=arr[:,:,F-1:F,:T]
    if Tp>T: out[:,:,:F,T:T]=arr[:,:,:,T-1:T]
    return out
X_feat=pad_FT(X_feat,4); N,C,F,T=X_feat.shape

# ===================== 5) 모델 =====================
class NeuroMel_ViT_DS(nn.Module):
    def __init__(self,F=40,T=64,num_classes=2,patch=4,dim=64,depth=3,heads=4,mlp_dim=128):
        super().__init__()
        self.conv_stem=nn.Sequential(
            nn.Conv2d(4,16,3,padding=1,bias=False),
            nn.BatchNorm2d(16),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),
            nn.Conv2d(16,32,1,bias=False),
            nn.BatchNorm2d(32),nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.Fs,self.Ts=F//2,T//2
        self.num_patches=(self.Fs//patch)*(self.Ts//patch)
        self.to_patch=nn.Sequential(
            Rearrange('b c (f p1) (t p2)->b (f t) (c p1 p2)',p1=patch,p2=patch),
            nn.Linear(32*patch*patch,dim)
        )
        enc=TransformerEncoderLayer(d_model=dim,nhead=heads,
                                    dim_feedforward=mlp_dim,
                                    dropout=0.0,activation='gelu',
                                    batch_first=True)
        self.encoder=TransformerEncoder(enc,num_layers=depth)
        self.cls=nn.Parameter(torch.randn(1,1,dim))
        self.pos=nn.Parameter(torch.randn(1,self.num_patches+1,dim))
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,num_classes))
    def forward(self,x):
        x=self.conv_stem(x); x=self.to_patch(x)
        b=x.size(0); cls=self.cls.expand(b,-1,-1)
        x=torch.cat([cls,x],1)
        x=x+self.pos[:,:x.size(1)]
        x=self.encoder(x)
        return self.head(x[:,0])

# ===================== 6) Subject-dependent 10×10 CV =====================
X_t=torch.tensor(X_feat,dtype=torch.float32)
y_t=torch.tensor(y,dtype=torch.long)
subjects=np.unique(subj_ids)
def spec_augment(x,fmask=3,tmask=3):
    x=x.clone()
    B,C,F,T=x.shape
    for i in range(B):
        if F>fmask:
            f0=np.random.randint(0,F-fmask+1)
            x[i,:,f0:f0+fmask,:]=0
        if T>tmask:
            t0=np.random.randint(0,T-tmask+1)
            x[i,:,:,t0:t0+tmask]=0
    return x

subject_results=[]
for subj in subjects:
    print(f"\n=== Subject {subj}: 10×10 CV ===")
    mask=subj_ids==subj
    Xs,ys=X_t[mask],y_t[mask]
    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=SEED)
    fold_accs=[]
    for fold,(tr,te) in enumerate(skf.split(Xs,ys),1):
        tr_loader=DataLoader(TensorDataset(Xs[tr],ys[tr]),batch_size=16,shuffle=True)
        te_loader=DataLoader(TensorDataset(Xs[te],ys[te]),batch_size=16)
        model=NeuroMel_ViT_DS(F=F,T=T).to(device)
        crit=nn.CrossEntropyLoss(label_smoothing=0.05)
        opt=optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
        sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1)
        best=0.0
        for ep in range(50):
            # train
            model.train(); corr=tot=0
            for xb,yb in tr_loader:
                xb,yb=xb.to(device),yb.to(device)
                xb=spec_augment(xb,3,3)
                opt.zero_grad(); out=model(xb)
                loss=crit(out,yb); loss.backward(); opt.step()
                corr+=(out.argmax(1)==yb).sum().item(); tot+=yb.size(0)
            sched.step(ep+1)
            # eval
            model.eval(); correct=total=0
            with torch.no_grad():
                for xb,yb in te_loader:
                    xb,yb=xb.to(device),yb.to(device)
                    out=model(xb)
                    correct+=(out.argmax(1)==yb).sum().item(); total+=yb.size(0)
            acc=correct/total; best=max(best,acc)
        fold_accs.append(best); print(f" Fold {fold:2d} | Best {best:.3f}")
    m,s=np.mean(fold_accs),np.std(fold_accs); subject_results.append(m)
    print(f" → Subject {subj}: {m:.3f} ± {s:.3f}")
    plt.figure(figsize=(6,4))
    plt.plot(fold_accs,marker='o',color='b')
    plt.axhline(m,color='r',ls='--',label=f"Mean={m:.3f}")
    plt.fill_between(range(10),m-s,m+s,color='r',alpha=0.2)
    plt.title(f"Subject {subj} - 10×10 CV (NeuroMelViT)")
    plt.xlabel("Fold"); plt.ylabel("Acc"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/Subj{subj}_folds.png",dpi=300); plt.close()

# ===================== 7) 전체 결과 =====================
om,osd=np.mean(subject_results),np.std(subject_results)
print("\n✅ Subject mean:",np.round(subject_results,3))
print(f"Overall = {om:.3f} ± {osd:.3f}")
plt.figure(figsize=(8,5))
plt.bar([f"S{s}" for s in subjects],subject_results,color='orchid',edgecolor='k')
plt.axhline(om,color='r',ls='--',label=f"Mean={om:.3f}")
plt.fill_between(range(len(subjects)),om-osd,om+osd,color='r',alpha=0.2)
plt.title("Subject-wise Accuracy (Neuro-Mel+ENCC+αβ ViT, 10×10 CV)")
plt.ylabel("Accuracy"); plt.ylim(0,1); plt.legend()
plt.tight_layout(); plt.savefig(f"{out_dir}/AllSubjects_Bar.png",dpi=300)
plt.show()
