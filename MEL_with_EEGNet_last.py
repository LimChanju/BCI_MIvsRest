# =========================================
#  EEGNet-Mel (32×128 Spectrogram + μ/β Heatmap Visualization)
# =========================================
import librosa, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from scipy.signal import butter, filtfilt
import mne, os, glob, re, warnings, matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")  # 필터 로그 완전 숨김

DATA_DIR = "./dataset/"
SAVE_DIR = "./Results_Mel"
os.makedirs(SAVE_DIR, exist_ok=True)

N_MELS, N_FFT, HOP = 32, 128, 64
BATCH, EPOCHS, LR, SEED = 16, 50, 1e-3, 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------
# Signal → Mel 변환
# ------------------------------
def butter_bandpass_filter(data, low, high, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data)

def eeg_to_mel(eeg, fs):
    S = np.abs(librosa.stft(eeg, n_fft=N_FFT, hop_length=HOP, window='hann'))**2
    mel_fb = librosa.filters.mel(sr=fs, n_fft=N_FFT, n_mels=N_MELS, fmin=8, fmax=30)
    mel = np.dot(mel_fb, S)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    img = cv2.resize(mel_db, (128, 32))
    return img.astype(np.float32), mel, mel_fb

# ------------------------------
# 데이터 로드
# ------------------------------
def load_bci2b_mel(path=DATA_DIR):
    files = sorted(glob.glob(os.path.join(path, "*T.gdf")))
    X_all, y_all, subj_ids = [], [], []
    for f in files:
        subj = int(re.findall(r"B0(\d)", f)[0])
        raw = mne.io.read_raw_gdf(f, preload=True)
        events, ev_dict = mne.events_from_annotations(raw)
        raw.pick_channels(["EEG:C3"]); fs = int(raw.info["sfreq"])
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
            subj_ids.extend([subj]*len(y))
    X = np.concatenate(X_all); y = np.concatenate(y_all); subj_ids = np.array(subj_ids)
    print(f"총 샘플:{len(X)} | MI:{int(y.sum())} | Rest:{len(y)-int(y.sum())}")
    return X, y, subj_ids, fs

# ------------------------------
# EEGNet
# ------------------------------
class EEGNet(nn.Module):
    def __init__(self,chans,samples,n_classes=2,dropout=0.5,kernLength=64,F1=8,D=2,F2=16):
        super().__init__()
        self.conv1=nn.Conv2d(1,F1,(1,kernLength),padding=(0,kernLength//2),bias=False)
        self.bn1=nn.BatchNorm2d(F1)
        self.depthwise=nn.Conv2d(F1,F1*D,(chans,1),groups=F1,bias=False)
        self.bn2=nn.BatchNorm2d(F1*D)
        self.pool1=nn.AvgPool2d((1,4))
        self.drop1=nn.Dropout(dropout)
        self.separable=nn.Sequential(
            nn.Conv2d(F1*D,F1*D,(1,16),groups=F1*D,padding=(0,8),bias=False),
            nn.Conv2d(F1*D,F2,1,bias=False),
            nn.BatchNorm2d(F2),nn.ELU(),
            nn.AvgPool2d((1,8)),nn.Dropout(dropout)
        )
        tmp=torch.zeros(1,1,chans,samples)
        with torch.no_grad(): out=self.forward_features(tmp)
        self.classifier=nn.Linear(out.shape[1],n_classes)
    def forward_features(self,x):
        x=F.elu(self.bn1(self.conv1(x)))
        x=F.elu(self.bn2(self.depthwise(x)))
        x=self.pool1(x); x=self.drop1(x)
        x=self.separable(x)
        return x.flatten(1)
    def forward(self,x): return self.classifier(self.forward_features(x))

# ------------------------------
# 학습 루프
# ------------------------------
def train_one_epoch(model,train_dl,test_dl,opt,crit,dev,epoch):
    model.train(); loss_total=0
    for xb,yb in train_dl:
        xb,yb=xb.to(dev),yb.to(dev)
        xb=xb.reshape(xb.size(0),1,xb.size(2),xb.size(3))
        opt.zero_grad(); out=model(xb)
        loss=crit(out,yb); loss.backward(); opt.step()
        loss_total+=loss.item()
    model.eval(); correct,total=0,0
    with torch.no_grad():
        for xb,yb in test_dl:
            xb,yb=xb.to(dev),yb.to(dev)
            xb=xb.reshape(xb.size(0),1,xb.size(2),xb.size(3))
            preds=model(xb).argmax(1)
            correct+=(preds==yb).sum().item(); total+=yb.size(0)
    acc=correct/total
    print(f"[{epoch+1:02d}/{EPOCHS}] Loss: {loss_total/len(train_dl):.4f} | Acc: {acc*100:.2f}%")
    return acc

# ------------------------------
# Main
# ------------------------------
def main_mel():
    X_raw,y_raw,subj_ids,fs=load_bci2b_mel(DATA_DIR)
    X_mel,mu_maps,beta_maps,mel_fulls=[],[],[],[]
    for x in X_raw:
        mel_img,mel_full,mel_fb=eeg_to_mel(butter_bandpass_filter(x,8,30,fs),fs)
        X_mel.append(mel_img); mel_fulls.append(mel_full)
        freq_bins=librosa.mel_frequencies(n_mels=N_MELS,fmin=8,fmax=30)
        mu_idx=np.where((freq_bins>=8)&(freq_bins<=13))[0]
        beta_idx=np.where((freq_bins>=14)&(freq_bins<=30))[0]
        mu_maps.append(np.mean(mel_full[mu_idx,:],axis=0))
        beta_maps.append(np.mean(mel_full[beta_idx,:],axis=0))

    X=torch.tensor(np.array(X_mel),dtype=torch.float32).unsqueeze(1)
    y=torch.tensor(y_raw,dtype=torch.long)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    subjects=np.unique(subj_ids)
    results_mean,results_std=[],[]

    for subj in subjects:
        print(f"\n=== Subject {subj}: 10×10 CV ===")
        subj_dir=os.path.join(SAVE_DIR,f"S{subj}"); os.makedirs(subj_dir,exist_ok=True)
        mask=subj_ids==subj; Xs,ys=X[mask],y[mask]
        skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=SEED)
        accs=[]
        for f,(tr,te) in enumerate(skf.split(Xs,ys),1):
            tr_dl=DataLoader(TensorDataset(Xs[tr],ys[tr]),batch_size=BATCH,shuffle=True)
            te_dl=DataLoader(TensorDataset(Xs[te],ys[te]),batch_size=BATCH)
            model=EEGNet(chans=32,samples=128).to(dev)
            opt=torch.optim.Adam(model.parameters(),lr=LR); crit=nn.CrossEntropyLoss()
            best=0
            for ep in range(EPOCHS):
                acc=train_one_epoch(model,tr_dl,te_dl,opt,crit,dev,ep)
                best=max(best,acc)
            accs.append(best)
            print(f" Fold {f:2d} | Best {best:.3f}")

        # ---- Fold Accuracy ----
        plt.figure(figsize=(6,4))
        plt.bar(range(1,11),accs,color='steelblue',edgecolor='black')
        plt.title(f"Subject {subj} Fold Accuracies (Mel)")
        plt.xlabel("Fold"); plt.ylabel("Accuracy")
        plt.ylim(0,1); plt.tight_layout()
        plt.savefig(os.path.join(subj_dir,"fold_acc_bar.png"),dpi=300); plt.close()

        # ✅ ---- 추가: EEGNet Feature Map Visualization (모델 통과 후) ----
        os.makedirs(os.path.join(subj_dir, "featuremaps"), exist_ok=True)
        subj_indices = np.where(subj_ids == subj)[0][:4]

        model.eval()
        with torch.no_grad():
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for i, idx in enumerate(subj_indices):
                if idx >= len(Xs):
                    idx = len(Xs) - 1  # ✅ 인덱스 초과 방지
                x_in = Xs[idx].unsqueeze(0).to(dev)  # (1, 1, 32, 128)
                feat = model.forward_features(x_in).cpu().numpy()
                # flatten된 feature를 2D로 reshape (대략적 시각화용)
                feat_2d = feat.reshape(8, -1)  # feature 채널 8개 기준으로 펼침

                # μ/β 유사 영역 나눠서 시각화 (상단/하단)
                axes[0, i].imshow(feat_2d[:4, :], aspect="auto", origin="lower", cmap="magma")
                axes[0, i].set_title(f"Feature Map #{i} (μ-like)")
                axes[1, i].imshow(feat_2d[4:, :], aspect="auto", origin="lower", cmap="magma")
                axes[1, i].set_title(f"Feature Map #{i} (β-like)")
                for ax in axes[:, i]:
                    ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(subj_dir, "featuremaps", "mu_beta_featuremaps.png"), dpi=300)
            plt.close()


        np.savetxt(os.path.join(subj_dir,"subject_summary.txt"),[np.mean(accs),np.std(accs)],fmt="%.4f")
        results_mean.append(np.mean(accs)); results_std.append(np.std(accs))
        print(f" → Subject {subj}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

if __name__ == "__main__":
    main_mel()
