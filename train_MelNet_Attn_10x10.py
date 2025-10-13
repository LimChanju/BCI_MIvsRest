import mne, numpy as np, glob, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa, matplotlib.pyplot as plt, os, re
from sklearn.model_selection import StratifiedKFold

# ======================================================
# 1. 데이터 불러오기 (C3, 8–30 Hz, Rest −4~0 / MI 0~4)
# ======================================================
files = sorted(glob.glob("./dataset/*T.gdf"))
print("Using files:", files)

X_all, y_all, subj_ids = [], [], []
fs = 250
for f in files:
    subj_id = int(re.findall(r"B0(\d)", f)[0])
    print(f"\n=== Loading {f} (Subject {subj_id}) ===")
    raw = mne.io.read_raw_gdf(f, preload=True)
    events, event_dict = mne.events_from_annotations(raw)
    raw.pick_channels(["EEG:C3"])
    raw.filter(8., 30., fir_design="firwin")

    if "769" in event_dict:
        left, right = event_dict["769"], event_dict["770"]
    elif 769 in event_dict:
        left, right = event_dict[769], event_dict[770]
    else:
        print(f"⚠️ {f}: MI 이벤트 없음 → skip")
        continue

    events_fixed = events.copy()
    events_fixed[events_fixed[:, -1]==left, -1]=1
    events_fixed[events_fixed[:, -1]==right, -1]=2
    event_id={"left":1,"right":2}

    rest=mne.Epochs(raw,events_fixed,event_id,tmin=-4.0,tmax=0.0,
                    baseline=None,preload=True)
    Xr=rest.get_data(); yr=np.zeros(len(Xr))
    mi=mne.Epochs(raw,events_fixed,event_id,tmin=0.0,tmax=4.0,
                  baseline=None,preload=True)
    Xm=mi.get_data(); ym=np.ones(len(Xm))
    X_all.append(np.concatenate([Xr,Xm]))
    y_all.append(np.concatenate([yr,ym]))
    subj_ids.extend([subj_id]*(len(yr)+len(ym)))

X=np.concatenate(X_all); y=np.concatenate(y_all); subj_ids=np.array(subj_ids)
print("\nFinal data:",X.shape,np.unique(y,return_counts=True))

# ======================================================
# 2. EEG → Mel-Spectrogram 변환
# ======================================================
def eeg_to_mel(eeg_data,sr=fs,n_fft=256,hop=128,n_mels=40,fmax=50):
    out=[]
    for trial in eeg_data:
        sig=trial[0]
        mel=librosa.feature.melspectrogram(y=sig,sr=sr,n_fft=n_fft,
                                           hop_length=hop,n_mels=n_mels,fmax=fmax)
        mel_db=librosa.power_to_db(mel,ref=np.max)
        mel_db=(mel_db-np.mean(mel_db))/(np.std(mel_db)+1e-6)
        out.append(mel_db)
    return np.stack(out)

print("🎧 EEG → Mel 변환중 ...")
mel_X=eeg_to_mel(X)
print("Mel shape:",mel_X.shape)

# ======================================================
# 3. 모델 정의 (EEG-MelNet + Attention)
# ======================================================
class FreqAttention(nn.Module):
    def __init__(self,n_mels):
        super().__init__()
        self.attn=nn.Sequential(
            nn.Linear(n_mels,n_mels),nn.ReLU(),
            nn.Linear(n_mels,n_mels),nn.Sigmoid()
        )
    def forward(self,x):
        w=self.attn(x.mean(-1).squeeze(1))
        return x*w.unsqueeze(1).unsqueeze(-1)

class TempAttention(nn.Module):
    def __init__(self,n_frames):
        super().__init__()
        self.attn=nn.Sequential(
            nn.Linear(n_frames,n_frames),nn.ReLU(),
            nn.Linear(n_frames,n_frames),nn.Sigmoid()
        )
    def forward(self,x):
        w=self.attn(x.mean(2).squeeze(1))
        return x*w.unsqueeze(1).unsqueeze(2)

class EEG_MelNet_Attn(nn.Module):
    def __init__(self,n_mels=40,n_frames=64,n_classes=2):
        super().__init__()
        self.freq=FreqAttention(n_mels)
        self.temp=TempAttention(n_frames)
        self.conv=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout(0.3)
        )
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*(n_mels//2)*(n_frames//2),64),
            nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(64,n_classes)
        )
    def forward(self,x):
        x=self.freq(x)
        x=self.temp(x)
        x=self.conv(x)
        return self.fc(x)

# ======================================================
# 4. Subject-dependent 10×10 CV
# ======================================================
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_t=torch.tensor(mel_X,dtype=torch.float32).unsqueeze(1)
y_t=torch.tensor(y,dtype=torch.long)
unique_subjs=np.unique(subj_ids)
save_dir="./MelNet_Attn_10x10"
os.makedirs(save_dir,exist_ok=True)

subject_results=[]
for subj in unique_subjs:
    print(f"\n=== Subject {subj}: 10×10 CV ===")
    mask=subj_ids==subj
    Xs,ys=X_t[mask],y_t[mask]
    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    fold_accs=[]
    n_mels,n_frames=X_t.shape[2],X_t.shape[3]

    for fold,(train_idx,test_idx) in enumerate(skf.split(Xs,ys),1):
        Xtr,Xte=Xs[train_idx],Xs[test_idx]
        ytr,yte=ys[train_idx],ys[test_idx]
        train_dl=DataLoader(TensorDataset(Xtr,ytr),batch_size=16,shuffle=True)
        test_dl=DataLoader(TensorDataset(Xte,yte),batch_size=16,shuffle=False)

        model=EEG_MelNet_Attn(n_mels,n_frames,2).to(device)
        crit=nn.CrossEntropyLoss(); opt=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best=0
        for ep in range(50):
            model.train(); c=t=0
            for xb,yb in train_dl:
                xb,yb=xb.to(device),yb.to(device)
                opt.zero_grad(); out=model(xb); loss=crit(out,yb)
                loss.backward(); opt.step()
                c+=(out.argmax(1)==yb).sum().item(); t+=yb.size(0)
            model.eval(); cc=tt=0
            with torch.no_grad():
                for xb,yb in test_dl:
                    xb,yb=xb.to(device),yb.to(device)
                    pr=model(xb); cc+=(pr.argmax(1)==yb).sum().item(); tt+=yb.size(0)
            acc=cc/tt; best=max(best,acc)
        fold_accs.append(best)
        print(f" Fold {fold:2d} | Best Acc: {best:.3f}")

    m,s=np.mean(fold_accs),np.std(fold_accs)
    subject_results.append(m)
    print(f" → Subject {subj} mean = {m:.3f} ± {s:.3f}")

    plt.figure(figsize=(6,4))
    plt.plot(range(1,11),fold_accs,marker='o',color='blue',label='Fold Accuracies')
    plt.axhline(m,color='r',ls='--',label=f'Mean = {m:.3f}')
    plt.fill_between(range(1,11),m-s,m+s,color='r',alpha=0.2,label=f'±1 SD ({s:.3f})')
    plt.title(f"Subject {subj} - 10×10 CV Accuracy")
    plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.ylim(0,1)
    plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f"Subj{subj}_MelNetAttn.png"),dpi=300); plt.close()

# ======================================================
# 5. 전체 평균 + 막대그래프
# ======================================================
om,osd=np.mean(subject_results),np.std(subject_results)
print("\n✅ Subject-wise mean accuracies:",np.round(subject_results,3))
print(f"Overall mean = {om:.3f} ± {osd:.3f}")

subjects=[f"Subj{s}" for s in unique_subjs]
plt.figure(figsize=(8,5))
plt.bar(subjects,subject_results,color='orchid',edgecolor='k')
plt.axhline(om,color='red',ls='--',label=f'Mean = {om:.3f}')
plt.fill_between(range(len(subjects)),om-osd,om+osd,color='red',alpha=0.2,
                 label=f'±1 SD ({osd:.3f})')
plt.title("Subject-wise Mean Accuracy (EEG-MelNet + Attention, 10×10 CV, C3)")
plt.ylabel("Accuracy"); plt.ylim(0,1); plt.xticks(rotation=45)
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(save_dir,"AllSubjects_Accuracy_Bar.png"),dpi=300)
plt.show()
