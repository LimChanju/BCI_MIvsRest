import mne
import numpy as np
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.model_selection import train_test_split, cross_val_score

# --------------------------
# 1. dataset 불러오기 (Training 세션만)
# --------------------------
files = sorted(glob.glob("./dataset/*T.gdf"))
print("Using files:", files)

all_epochs = []

for f in files:
    print(f"\n=== Loading {f} ===")
    raw = mne.io.read_raw_gdf(f, preload=True)
    events, event_dict = mne.events_from_annotations(raw)
    print("Event dict:", event_dict)

    # EEG 채널만 선택
    raw.pick_channels(["EEG:C3", "EEG:Cz", "EEG:C4"])
    raw.filter(8., 30., fir_design="firwin")

    # 매핑 찾기
    if "769" in event_dict:
        left = event_dict["769"]
        right = event_dict["770"]
    elif 769 in event_dict:
        left = event_dict[769]
        right = event_dict[770]
    else:
        print(f"⚠️ {f}에서 769/770 이벤트 없음 → 건너뜀")
        continue

    # === 공통 라벨로 강제 변환 (left=1, right=2) ===
    events_fixed = events.copy()
    events_fixed[events_fixed[:, -1] == left, -1] = 1
    events_fixed[events_fixed[:, -1] == right, -1] = 2

    # event_id는 고정된 값만 사용
    event_id = {"left": 1, "right": 2}

    # Epochs 생성 (cue 이후 0.5~2.5s)
    tmin, tmax = 0.5, 2.5
    epochs = mne.Epochs(raw, events_fixed, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
    all_epochs.append(epochs)

# --------------------------
# 2. 세션 합치기
# --------------------------
if len(all_epochs) == 0:
    raise RuntimeError("⚠️ 유효한 세션이 없음. event_dict 확인 필요!")

epochs = mne.concatenate_epochs(all_epochs)
X = epochs.get_data()   # (trials, channels, times)
y = epochs.events[:, -1]

# left=1 → 0, right=2 → 1 변환
y = np.where(y == 1, 0, 1)

print("\nFinal data shape:", X.shape)
print("Final labels:", np.unique(y, return_counts=True))

# --------------------------
# 3. CSP + LDA 파이프라인
# --------------------------
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()

clf = Pipeline([
    ("CSP", csp),
    ("LDA", lda)
])

# --------------------------
# 4. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"\nCSP+LDA Test Accuracy: {acc:.3f}")

# --------------------------
# 5. Cross-validation
# --------------------------
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-val Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))
