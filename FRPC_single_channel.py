import os, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import mne
from scipy.signal import welch
from scipy.stats import pearsonr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Config
# -----------------------
CFG = dict(
    fs_target=250,
    notch_freqs=[50],
    hp=8.0, lp=40.0,
    tmin=0.5, tmax=2.5,
    fb_start=8, fb_stop=40, fb_step=4,
    top_fb=6, top_feats=40,
    n_repeats=10, n_splits=10,
    random_state=42,
    result_dir="./FRPC_results"
)

MI_LEFT, MI_RIGHT = 769, 770
CANDIDATE_CHS = ["EEG:C3", "EEG:Cz", "EEG:C4"]

# -----------------------
# Feature utilities
# -----------------------
def hjorth_params(x):
    dx = np.diff(x)
    var_x = np.var(x)
    var_dx = np.var(dx) if dx.size > 1 else 0.0
    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + 1e-12))
    ddx = np.diff(dx)
    var_ddx = np.var(ddx) if ddx.size > 1 else 0.0
    mobility_dx = np.sqrt(var_ddx / (var_dx + 1e-12)) if var_dx > 0 else 0.0
    complexity = (mobility_dx / (mobility + 1e-12)) if mobility > 0 else 0.0
    return activity, mobility, complexity

def waveform_length(x):
    return np.sum(np.abs(np.diff(x)))

def bandpower_welch(x, fs, fmin, fmax):
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    mask = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0

def fisher_ratio(class1_vals, class2_vals):
    m1, m2 = np.mean(class1_vals), np.mean(class2_vals)
    v1, v2 = np.var(class1_vals, ddof=1), np.var(class2_vals, ddof=1)
    return (m1 - m2)**2 / (v1 + v2 + 1e-12)

# -----------------------
# Data loading
# -----------------------
def load_2b_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*T.gdf")))
    if len(files) == 0:
        raise FileNotFoundError("No *T.gdf files found in data_dir")
    return files

def preprocess_raw(raw, cfg):
    if raw.info["sfreq"] != cfg["fs_target"]:
        raw.resample(cfg["fs_target"])
    if cfg["notch_freqs"]:
        raw.notch_filter(cfg["notch_freqs"])
    raw.filter(cfg["hp"], cfg["lp"], fir_design="firwin")
    return raw

def extract_epochs_2b(files, task="LR", cfg=CFG):
    X_all, y_all, subj_all = [], [], []
    for f in files:
        raw = mne.io.read_raw_gdf(f, preload=True, verbose=False)
        picks = [ch for ch in CANDIDATE_CHS if ch in raw.ch_names]
        raw.pick_channels(picks)
        raw = preprocess_raw(raw, cfg)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        fs = raw.info["sfreq"]
        n_samp = int((cfg["tmax"] - cfg["tmin"]) * fs)

        # MI event extraction
        left_codes = [v for k, v in event_id.items() if "769" in k or v == MI_LEFT]
        right_codes = [v for k, v in event_id.items() if "770" in k or v == MI_RIGHT]
        ev_mask = np.isin(events[:,2], left_codes + right_codes)
        events_lr = events[ev_mask]

        X_lr, y_lr = [], []
        for onset, _, code in events_lr:
            t0 = int(onset + cfg["tmin"] * fs)
            t1 = t0 + n_samp
            if t1 <= raw.n_times:
                seg = raw.get_data(start=t0, stop=t1)
                X_lr.append(seg)
                if code in left_codes: y_lr.append(1)
                elif code in right_codes: y_lr.append(2)

        X_lr, y_lr = np.array(X_lr), np.array(y_lr)

        # Rest epochs (pre-cue)
        X_rst = []
        if task in ["LvsRst", "RvsRst"]:
            pre_len = int(2.0 * fs)
            for onset, _, _ in events_lr:
                t_end = int(onset)
                if t_end - pre_len > 0 and pre_len >= n_samp:
                    start = np.random.randint(t_end - pre_len, t_end - n_samp + 1)
                    seg = raw.get_data(start=start, stop=start + n_samp)
                    X_rst.append(seg)
            X_rst = np.array(X_rst)
            y_rst = np.zeros(len(X_rst), dtype=int)

        if task == "LR":
            X, y = X_lr, y_lr
        elif task == "LvsRst":
            maskL = (y_lr == 1)
            X = np.concatenate([X_lr[maskL], X_rst], axis=0)
            y = np.concatenate([np.ones(maskL.sum()), y_rst], axis=0)
        elif task == "RvsRst":
            maskR = (y_lr == 2)
            X = np.concatenate([X_lr[maskR], X_rst], axis=0)
            y = np.concatenate([np.ones(maskR.sum()), y_rst], axis=0)

        subj = os.path.basename(f).split(".")[0]
        X_all.append(X)
        y_all.append(y)
        subj_all += [subj]*len(y)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all, raw.ch_names, np.array(subj_all)

# -----------------------
# FR-based channel selection
# -----------------------
def select_best_channel_by_FR(X, y, ch_names):
    FR_scores = []
    for ci in range(X.shape[1]):
        feat = np.log(np.var(X[:, ci, :], axis=1) + 1e-12)
        classes = np.unique(y)
        fr = fisher_ratio(feat[y == classes[0]], feat[y == classes[1]]) if len(classes) == 2 else 0.0
        FR_scores.append(fr)
    best_idx = int(np.argmax(FR_scores))
    return best_idx, FR_scores

# -----------------------
# Filterbank + feature extraction
# -----------------------
def make_filterbanks(f_start=8, f_stop=40, step=4):
    bands = []
    f = f_start
    while f < f_stop:
        bands.append((f, min(f+step, f_stop)))
        f += step
    return bands

def extract_features_single_channel(X1c, fs, bands):
    feats, names = [], []
    for (fmin, fmax) in bands:
        lv = np.log(np.var(X1c, axis=1) + 1e-12)
        hj = np.array([hjorth_params(x) for x in X1c])
        wl = np.array([waveform_length(x) for x in X1c])
        bp = np.array([bandpower_welch(x, fs, fmin, fmax) for x in X1c])
        block = np.column_stack([lv, hj, wl, bp])
        feats.append(block)
        names += [f"LV_{fmin}-{fmax}", f"HJ_Act_{fmin}-{fmax}", f"HJ_Mob_{fmin}-{fmax}",
                  f"HJ_Cmp_{fmin}-{fmax}", f"WL_{fmin}-{fmax}", f"BP_{fmin}-{fmax}"]
    F = np.hstack(feats)
    return F, names

def pearson_rank_features(X, y):
    scores = []
    for i in range(X.shape[1]):
        try:
            r, _ = pearsonr(X[:, i], y)
            scores.append(abs(r))
        except Exception:
            scores.append(0.0)
    return np.array(scores)

# -----------------------
# CV + Visualization + Save
# -----------------------
def run_cv(X, y, fs, ch_names, cfg):
    os.makedirs(cfg["result_dir"], exist_ok=True)

    best_idx, fr_scores = select_best_channel_by_FR(X, y, ch_names)
    X1c = X[:, best_idx, :]
    print(f"[FR] Best channel = {ch_names[best_idx]} (scores: {np.round(fr_scores,3)})")

    bands = make_filterbanks(cfg["fb_start"], cfg["fb_stop"], cfg["fb_step"])
    F_all, feat_names = extract_features_single_channel(X1c, fs, bands)
    rng = np.random.RandomState(cfg["random_state"])
    accs = []

    for rep in range(cfg["n_repeats"]):
        X_rep, y_rep = shuffle(F_all, y, random_state=rng.randint(0, 1e9))
        skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=rng.randint(0, 1e9))
        for _, (tr, te) in enumerate(skf.split(X_rep, y_rep)):
            Xtr, Xte = X_rep[tr], X_rep[te]
            ytr, yte = y_rep[tr], y_rep[te]

            feat_corr = pearson_rank_features(Xtr, ytr)
            nb = len(bands)
            band_scores = [np.mean(feat_corr[bi*6:(bi+1)*6]) for bi in range(nb)]
            band_ranks = np.argsort(band_scores)[::-1]
            selected_bands = band_ranks[:cfg["top_fb"]]

            sel_idx = []
            for bi in selected_bands:
                sel_idx.extend(list(range(bi*6, (bi+1)*6)))
            local_corr = feat_corr[sel_idx]
            local_order = np.argsort(local_corr)[::-1][:cfg["top_feats"]]
            final_idx = np.array(sel_idx)[local_order]

            Xtr_sel, Xte_sel = Xtr[:, final_idx], Xte[:, final_idx]
            clf = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=rng.randint(0, 1e9))
            clf.fit(Xtr_sel, ytr)
            acc = accuracy_score(yte, clf.predict(Xte_sel))
            accs.append(acc)
        print(f"Repeat {rep+1}/{cfg['n_repeats']} done.")

    accs = np.array(accs)
    mean_acc, std_acc = accs.mean()*100, accs.std()*100
    print(f"\n[Result] 10x10 CV Accuracy = {mean_acc:.2f}% ± {std_acc:.2f}% "
          f"(best ch={ch_names[best_idx]})")

    # --- Save results ---
    pd.DataFrame({"Accuracy": accs}).to_csv(os.path.join(cfg["result_dir"], "acc_results.csv"), index=False)
    pd.DataFrame({"Channel": ch_names, "FR_score": fr_scores}).to_csv(os.path.join(cfg["result_dir"], "FR_scores.csv"), index=False)

    # --- Visualization ---
    plt.figure(figsize=(6,4))
    plt.hist(accs*100, bins=15, color="skyblue", edgecolor="black")
    plt.title("FRPC 10×10 CV Accuracy Distribution")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(cfg["result_dir"], "acc_distribution.png"))
    plt.close()

    plt.figure(figsize=(5,4))
    plt.bar(ch_names, fr_scores, color=["#ff9999","#99ccff","#99ff99"])
    plt.title("Fisher’s Ratio by Channel")
    plt.ylabel("FR Score")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(cfg["result_dir"], "FR_scores.png"))
    plt.close()

    print(f"[Saved] Results and plots → {cfg['result_dir']}")
    return accs

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--task", type=str, default="LR", choices=["LR","LvsRst","RvsRst"])
    args = parser.parse_args()

    files = load_2b_files(args.data_dir)
    print("Using files:", files)

    X, y, ch_names, subj = extract_epochs_2b(files, task=args.task, cfg=CFG)
    accs = run_cv(X, y, fs=250.0, ch_names=ch_names, cfg=CFG)
