import mne
import numpy as np
import glob, re, os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# --------------------------
# 1. dataset 불러오기 (Training 세션만)
# --------------------------
files = sorted(glob.glob("./dataset/*T.gdf"))
print("Using files:", files)

X_all, y_all, subj_ids = [], [], []

for f in files:
    subj_id = int(re.findall(r"B0(\d)", f)[0])  # subject ID 추출
    print(f"\n=== Loading {f} (Subject {subj_id}) ===")

    raw = mne.io.read_raw_gdf(f, preload=True)
    events, event_dict = mne.events_from_annotations(raw)

    # C3 단일 채널만 사용
    raw.pick_channels(["EEG:C3"])
    raw.filter(8., 30., fir_design="firwin")

    # MI 이벤트 매핑
    if "769" in event_dict:
        left = event_dict["769"]
        right = event_dict["770"]
    elif 769 in event_dict:
        left = event_dict[769]
        right = event_dict[770]
    else:
        print(f"⚠️ {f}에서 MI 이벤트 없음 → 건너뜀")
        continue

    # 이벤트를 공통 코드(1,2)로 변환
    events_fixed = events.copy()
    events_fixed[events_fixed[:, -1] == left, -1] = 1
    events_fixed[events_fixed[:, -1] == right, -1] = 2
    event_id = {"left": 1, "right": 2}

    # === Rest epochs (−4.0 ~ 0.0 s) ===
    rest_epochs = mne.Epochs(raw, events_fixed, event_id=event_id,
                             tmin=-4.0, tmax=0.0,
                             baseline=None, preload=True)
    X_rest = rest_epochs.get_data()   # (n_trials, 1, times)
    y_rest = np.zeros(len(X_rest))    # label = 0

    # === MI epochs (0.0 ~ 4.0 s) ===
    mi_epochs = mne.Epochs(raw, events_fixed, event_id=event_id,
                           tmin=0.0, tmax=4.0,
                           baseline=None, preload=True)
    X_mi = mi_epochs.get_data()
    y_mi = np.ones(len(X_mi))         # label = 1

    # Rest + MI 합치기
    X_subj = np.concatenate([X_rest, X_mi], axis=0)
    y_subj = np.concatenate([y_rest, y_mi], axis=0)

    X_all.append(X_subj)
    y_all.append(y_subj)
    subj_ids.extend([subj_id] * len(y_subj))

# --------------------------
# 2. 데이터 합치기
# --------------------------
X = np.concatenate(X_all, axis=0)  # (trials, 1, times)
y = np.concatenate(y_all, axis=0)
subj_ids = np.array(subj_ids)

print("\nFinal data shape:", X.shape)
print("Final labels:", np.unique(y, return_counts=True))
print("Subjects:", np.unique(subj_ids))

# --------------------------
# 3. CSP + LDA 파이프라인 정의
# --------------------------
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([("CSP", csp), ("LDA", lda)])

# --------------------------
# 4. Subject-dependent 10×10 CV
# --------------------------
save_dir = "./CSP_LDA_10x10"
os.makedirs(save_dir, exist_ok=True)

unique_subjects = np.unique(subj_ids)
subject_results = []

for subj in unique_subjects:
    subj_mask = subj_ids == subj
    X_subj = X[subj_mask]
    y_subj = y[subj_mask]
    print(f"\n=== Subject {subj}: 10×10 CV ({len(y_subj)} trials) ===")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_subj, y_subj), 1):
        X_train, X_test = X_subj[train_idx], X_subj[test_idx]
        y_train, y_test = y_subj[train_idx], y_subj[test_idx]

        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        fold_accs.append(acc)
        print(f"  Fold {fold:2d} | Acc: {acc:.3f}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    subject_results.append(mean_acc)
    print(f" → Subject {subj} mean = {mean_acc:.3f} ± {std_acc:.3f}")

    # --------------------------
    # 5. Fold Accuracy 시각화
    # --------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 11), fold_accs, marker='o', label="Fold Accuracies")
    plt.axhline(mean_acc, color='r', linestyle='--', label=f"Mean = {mean_acc:.3f}")
    plt.fill_between(range(1, 11),
                     mean_acc - std_acc, mean_acc + std_acc,
                     color='r', alpha=0.2, label=f"±1 SD ({std_acc:.3f})")

    plt.title(f"Subject {subj} | 10×10 CV Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Subj{subj}_CSP_LDA.png"), dpi=300)
    plt.close()

# --------------------------
# 6. 전체 평균 및 막대그래프 시각화
# --------------------------
overall_mean = np.mean(subject_results)
overall_std = np.std(subject_results)

print("\n✅ Subject-wise mean accuracies:", np.round(subject_results, 3))
print(f"Overall mean = {overall_mean:.3f} ± {overall_std:.3f}")
print(f"📁 Results saved to: {save_dir}/")

# --------------------------
# 7. 전체 subject 막대그래프 시각화
# --------------------------
subjects = [f"Subj{s}" for s in unique_subjects]
plt.figure(figsize=(8, 5))
plt.bar(subjects, subject_results, color='lightseagreen', edgecolor='k')
plt.axhline(overall_mean, color='red', linestyle='--', label=f'Mean = {overall_mean:.3f}')
plt.fill_between(range(len(subjects)),
                 overall_mean - overall_std, overall_mean + overall_std,
                 color='red', alpha=0.2, label=f'±1 SD ({overall_std:.3f})')

plt.title("Subject-wise Mean Accuracy (CSP+LDA, 10×10 CV, C3)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "AllSubjects_Accuracy_Bar.png"), dpi=300)
plt.show()
