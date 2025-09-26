import mne
import numpy as np
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# --------------------------
# 1. dataset 불러오기 (Training 세션만)
# --------------------------
files = sorted(glob.glob("./dataset/*T.gdf"))
print("Using files:", files)

X_all, y_all = [], []

for f in files:
    print(f"\n=== Loading {f} ===")
    raw = mne.io.read_raw_gdf(f, preload=True)
    events, event_dict = mne.events_from_annotations(raw)

    # EEG 채널 선택
    raw.pick_channels(["EEG:C3", "EEG:Cz", "EEG:C4"])
    raw.filter(8., 30., fir_design="firwin")

    # MI cue 이벤트 찾기 (left=769, right=770)
    if "769" in event_dict:
        mi_left = event_dict["769"]
        mi_right = event_dict["770"]
    elif 769 in event_dict:
        mi_left = event_dict[769]
        mi_right = event_dict[770]
    else:
        print(f"⚠️ {f}에서 MI 이벤트 없음 → 건너뜀")
        continue

    # === Rest epoch: cue 이전 (-2~0s) ===
    rest_epochs = mne.Epochs(
        raw, events, event_id={ "rest": mi_left },  # left cue 기준 rest
        tmin=-2.0, tmax=0.0, baseline=None, preload=True
    )

    # === MI epoch: cue 이후 (0.5~2.5s) ===
    mi_epochs = mne.Epochs(
        raw, events, event_id={"left": mi_left, "right": mi_right},
        tmin=0.5, tmax=2.5, baseline=None, preload=True
    )

    # 데이터와 라벨
    X_rest, y_rest = rest_epochs.get_data(), np.zeros(len(rest_epochs))
    X_mi, y_mi = mi_epochs.get_data(), np.ones(len(mi_epochs))

    # 합치기
    X_all.append(np.concatenate([X_rest, X_mi], axis=0))
    y_all.append(np.concatenate([y_rest, y_mi], axis=0))

# --------------------------
# 2. 세션 합치기
# --------------------------
X = np.concatenate(X_all, axis=0)
y = np.concatenate(y_all, axis=0)
print("\nFinal data shape:", X.shape)   # (trials, 3, times)
print("Final labels:", np.unique(y, return_counts=True))

# --------------------------
# 3. 단일 채널별 성능 비교
# --------------------------
channels = ["C3", "Cz", "C4"]
acc_results, cv_results = [], []

for i, ch in enumerate(channels):
    print(f"\n=== Evaluating single channel: {ch} ===")

    # 단일 채널 선택
    X_ch = X[:, i, :][:, np.newaxis, :]

    # CSP + LDA 파이프라인
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_ch, y, test_size=0.2, stratify=y, random_state=42
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Test Accuracy ({ch}): {acc:.3f}")
    acc_results.append(acc)

    # Cross-validation
    scores = cross_val_score(clf, X_ch, y, cv=5)
    print(f"Cross-val Accuracy ({ch}): {scores.mean():.3f} (+/- {scores.std():.3f})")
    cv_results.append(scores.mean())

# --------------------------
# 4. 결과 시각화
# --------------------------
plt.figure(figsize=(6,4))
plt.bar(channels, acc_results, color=["skyblue","orange","green"])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Single-channel CSP+LDA (MI vs Rest)")
for i, v in enumerate(acc_results):
    plt.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=10)
plt.show()
