"""
core.py — logika inti sistem deteksi anomali DDoS (tanpa tampilan).
Dipisah dari app.py supaya bisa diuji dan dipakai ulang.

Dua alur yang TERPISAH:
  PELATIHAN : data_acuan() + train_bundle() + save_bundle()   -> hanya dipakai pada DATASET LATIH
  DETEKSI   : load_bundle() + preprocess() + anomaly_score()   -> hanya MEMUAT model tersimpan, tidak ada .fit()

Aturan yang SAMA dengan notebook pelatihan:
- fitur: inf -> NaN -> 0
- transformasi: "raw" | "slog" (log1p bertanda) | "slog+scaler" (LOF)
- skor anomali = -model.score_samples(X); anomali jika skor > threshold
- threshold = persentil skor data latih (seluruhnya trafik BENIGN) pada FPR target
"""
import os
import re
import glob
import time

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

BENIGN = "BENIGN"
MIN_TYPE_N = 200          # jenis serangan dengan sampel uji >= ini dianggap cukup untuk disimpulkan
PORT_COLS = {"destination_port", "destination port", "source_port", "source port"}
ID_COLS = {"flow id", "flow_id", "source ip", "source_ip", "destination ip", "destination_ip",
           "timestamp", "unnamed: 0", "unnamed: 0.1", "simillarhttp", "similarhttp"}


# ----------------------------------------------------------------------------- data
def slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def read_csv(src):
    df = pd.read_csv(src, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    return df


def find_label(df):
    for c in df.columns:
        if c.strip().lower() in ("label", "class"):
            return c
    return None


def normalize_labels(s):
    s = s.astype(str).str.strip()
    return s.where(s.str.upper() != BENIGN, BENIGN)


def dataset_info(df, label):
    num = df.drop(columns=[label], errors="ignore").select_dtypes("number")
    arr = num.to_numpy(dtype="float64", na_value=np.nan)
    info = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_features": num.shape[1],
        "n_missing": int(df.isna().sum().sum()),
        "n_inf": int(np.isinf(arr).sum()),
        "n_dup": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
    }
    if label:
        lab = normalize_labels(df[label])
        info["n_benign"] = int((lab == BENIGN).sum())
        info["n_attack"] = len(df) - info["n_benign"]
        info["dist"] = lab.value_counts()
    return info


def candidate_features(df, label):
    """Fitur untuk latih ulang: numerik, bukan label/identitas/port, tidak konstan."""
    out = []
    for c in df.columns:
        if c == label or c.lower() in PORT_COLS | ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=False) > 1:
            out.append(c)
    return out


def preprocess(df, label, features, drop_duplicates=False):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} fitur yang dibutuhkan model tidak ada di dataset, "
                         f"mis.: {missing[:5]}")
    log = [("Data awal", len(df))]
    X = df[features].apply(pd.to_numeric, errors="coerce")
    arr = X.to_numpy(dtype="float64", na_value=np.nan)
    n_inf = int(np.isinf(arr).sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    n_nan = int(X.isna().sum().sum())
    X = X.fillna(0).astype("float64")
    labels = normalize_labels(df[label]) if label else None
    log.append(("Setelah cleaning (inf/NaN diganti 0)", len(X)))

    if drop_duplicates:
        tmp = X.copy()
        if labels is not None:
            tmp["__label"] = labels.values
        keep = ~tmp.duplicated().values
        X = X[keep]
        labels = labels[keep] if labels is not None else None
        log.append(("Setelah hapus duplikat", len(X)))

    X = X.reset_index(drop=True)
    labels = labels.reset_index(drop=True) if labels is not None else None
    log.append(("Setelah seleksi fitur", len(X)))
    stats = {"n_inf": n_inf, "n_nan_total": n_nan, "n_features": len(features)}
    return X, labels, log, stats


# ----------------------------------------------------------------------------- split
def _stratified(idx, size, labels, seed, test=False):
    strat = None
    if labels is not None:
        vc = labels.iloc[idx].value_counts()
        if vc.min() >= 2:
            strat = labels.iloc[idx].values
    kw = {"test_size": size} if test else {"train_size": size}
    try:
        return train_test_split(idx, stratify=strat, random_state=seed, **kw)
    except ValueError:
        return train_test_split(idx, stratify=None, random_state=seed, **kw)


def sample_rows(labels, n_rows, n, seed):
    """Deteksi: seluruh baris file baru dipakai. Jika n diisi (< n_rows), diambil sampel (terstratifikasi bila ada label)."""
    idx = np.arange(n_rows)
    if n is None or n >= n_rows:
        return idx
    return np.sort(_stratified(idx, int(n), labels, seed)[0])


def data_acuan(labels):
    """PELATIHAN: data acuan = SEMUA trafik normal pada dataset latih. Tanpa pembagian data dan tanpa batas jumlah.
    Baris serangan tidak dipakai melatih karena model hanya mempelajari pola trafik normal."""
    lab = np.asarray(labels)
    normal = lab == BENIGN
    return {"fit": np.where(normal)[0], "n_attack": int((~normal).sum()), "n_total": int(len(lab))}


# ----------------------------------------------------------------------------- model
def list_bundles(model_dir):
    paths = sorted(glob.glob(os.path.join(model_dir, "*.joblib")))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def parse_name(name):
    m = re.match(r"^(.+)_seed(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


REQUIRED_KEYS = {"model", "model_name", "features", "preprocess", "thresholds"}


def normalize_bundle(b):
    """Mengubah paket model dari notebook pelatihan (kunci: type, threshold, ...) ke format sistem ini."""
    p = dict(b["params"])
    name = b["type"]
    if name == "IF":
        prep, scaler = ("slog" if p.get("log_input") else "raw"), None
    else:
        prep, scaler = "slog+scaler", b["scaler"]
    return {"model": b["model"], "model_name": name, "seed": b.get("seed"), "params": p,
            "features": list(b["features"]), "preprocess": prep, "scaler": scaler,
            "thresholds": {float(b.get("fpr_target", 0.05)): float(b["threshold"])},
            "benign_label": BENIGN, "sklearn_version": b.get("sklearn_version"),
            "n_train": b.get("n_acuan"), "n_cal": None, "fit_time_s": None,
            "trained_on": b.get("trained_on"), "created": b.get("created")}


def load_bundle(path):
    """Memuat model tersimpan. Hanya membaca file; tidak melatih apa pun."""
    b = joblib.load(path)
    if isinstance(b, dict) and "thresholds" not in b and {"threshold", "type"} <= set(b):
        b = normalize_bundle(b)
    if not isinstance(b, dict) or not REQUIRED_KEYS <= set(b):
        raise ValueError(f"{os.path.basename(path)} bukan bundle model yang valid "
                         f"(butuh kunci {sorted(REQUIRED_KEYS)}).")
    return b


def bundle_info(key, b):
    return {"key": key, "algo": b["model_name"], "seed": b.get("seed"), "preprocess": b["preprocess"],
            "n_features": len(b["features"]), "thresholds": sorted(b["thresholds"].items()),
            "sklearn_version": b.get("sklearn_version"), "trained_on": b.get("trained_on"),
            "created": b.get("created"), "n_train": b.get("n_train"), "params": b.get("params")}


def transform(bundle, X):
    A = np.asarray(X, dtype="float64")
    if bundle["preprocess"] in ("slog", "slog+scaler"):
        A = slog(A)
    if bundle["preprocess"] == "slog+scaler":
        A = bundle["scaler"].transform(A)
    return A


def anomaly_score(bundle, X):
    A = transform(bundle, X)
    t0 = time.time()
    s = -bundle["model"].score_samples(A)
    return s, time.time() - t0


def train_bundle(name, params, X_fit, features, fprs, seed, trained_on=None):
    """Melatih model dari SELURUH data acuan (X_fit). Threshold ditentukan dari skor data latih itu sendiri,
    sehingga tidak perlu data kalibrasi terpisah:
      IF  : skor data latih (tiap pohon hanya memakai sampel kecil, bias in-sample sangat kecil)
      LOF : negative_outlier_factor_ (skor tiap titik latih terhadap tetangganya, tanpa dirinya sendiri)."""
    params = dict(params)
    if name == "IF":
        log_in = bool(params.pop("log_input", True))
        prep = "slog" if log_in else "raw"
        scaler = None
        A = slog(X_fit) if log_in else np.asarray(X_fit, dtype="float64")
        model = IsolationForest(random_state=seed, n_jobs=-1, **params)
        params["log_input"] = log_in
    else:
        prep = "slog+scaler"
        scaler = StandardScaler().fit(slog(X_fit))
        A = scaler.transform(slog(X_fit))
        model = LocalOutlierFactor(novelty=True, n_jobs=-1, **params)
    t0 = time.time()
    model.fit(A)
    t_fit = time.time() - t0
    s_train = -model.negative_outlier_factor_ if name == "LOF" else -model.score_samples(A)
    b = {"model": model, "model_name": name, "seed": seed, "params": params,
         "features": list(features), "preprocess": prep, "scaler": scaler,
         "thresholds": {float(f): float(np.percentile(s_train, 100 * (1 - f))) for f in fprs},
         "threshold_source": "skor data latih sendiri", "benign_label": BENIGN,
         "sklearn_version": sklearn.__version__, "n_train": int(len(X_fit)), "n_cal": 0, "fit_time_s": t_fit,
         "trained_on": trained_on, "created": time.strftime("%Y-%m-%d %H:%M"), "train_fp": None}
    return b


def save_bundle(b, model_dir, key):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{key}.joblib")
    joblib.dump(b, path)
    return path


# ----------------------------------------------------------------------------- pengaman: bukan data latih
def row_hashes(X):
    """Sidik jari tiap baris fitur (sudah dibersihkan). Dipakai untuk mengenali data latih."""
    df = pd.DataFrame(np.asarray(X, dtype="float64")).round(6)
    return pd.util.hash_pandas_object(df, index=False).to_numpy(dtype="uint64")


def save_fingerprint(path, X, features, source):
    h = np.unique(row_hashes(X))
    np.savez(path, hashes=h, features=np.array(list(features), dtype=object), source=str(source), n_rows=len(X))


def load_fingerprint(path):
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=True)
    return {"hashes": z["hashes"], "features": list(z["features"]), "source": str(z["source"]), "n_rows": int(z["n_rows"])}


def overlap_ratio(hashes, fp_hashes):
    """Porsi baris file baru yang IDENTIK dengan baris data latih."""
    return float(np.isin(hashes, fp_hashes).mean()) if len(hashes) else 0.0


# ----------------------------------------------------------------------------- evaluasi
def evaluate(labels, pred, score):
    y = (np.asarray(labels) != BENIGN).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    m = {"Accuracy": accuracy_score(y, pred),
         "Precision": precision_score(y, pred, zero_division=0),
         "Recall": recall_score(y, pred, zero_division=0),
         "F1": f1_score(y, pred, zero_division=0),
         "ROC_AUC": roc_auc_score(y, score) if both else np.nan,
         "PR_AUC": average_precision_score(y, score) if both else np.nan,
         "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
         "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    att = y == 1
    per = (pd.DataFrame({"Jenis": np.asarray(labels)[att], "det": pred[att]})
           .groupby("Jenis")["det"].agg(Recall="mean", n="size")
           .sort_values("n", ascending=False))
    major = per[per["n"] >= MIN_TYPE_N]["Recall"]
    m["Macro_Recall_Mayor"] = major.mean() if len(major) else np.nan
    cm = np.array([[tn, fp], [fn, tp]])
    return m, cm, per


def findings(model_name, m, per, score, labels, thr):
    c = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")
    out = [f"Pada data uji ini, {model_name} memperoleh F1 {c(m['F1'])} ({c(m['F1'] * 100, 1)}%) "
           f"dengan Precision {c(m['Precision'])} ({c(m['Precision'] * 100, 1)}%) "
           f"dan Recall {c(m['Recall'])} ({c(m['Recall'] * 100, 1)}%), "
           f"pada FPR aktual {c(m['FPR'] * 100, 1)}%."]
    lab = np.asarray(labels)
    sb, sa = score[lab == BENIGN], score[lab != BENIGN]
    if len(sb) and len(sa):
        rel = "lebih tinggi" if np.median(sa) > np.median(sb) else "TIDAK lebih tinggi"
        out.append(f"Median skor anomali trafik serangan ({c(np.median(sa))} skor) {rel} "
                   f"dibanding trafik normal ({c(np.median(sb))} skor); threshold = {c(thr)} skor.")
    dots = lambda n: f"{int(n):,}".replace(",", ".")
    out.append(f"Terdapat {dots(m['FP'])} flow alarm palsu (normal terdeteksi anomali) dan "
               f"{dots(m['FN'])} flow serangan lolos (serangan terdeteksi normal).")
    ok = per[per["n"] >= MIN_TYPE_N]
    weak = ok[ok["Recall"] < 0.7]
    if len(weak):
        out.append("Recall rendah (< 0,70 atau 70%) pada: " +
                   ", ".join(f"{i} ({c(r, 2)} atau {c(r * 100, 1)}%)" for i, r in weak["Recall"].items()) +
                   ". Jenis serangan ini paling sering lolos.")
    small = per[per["n"] < MIN_TYPE_N]
    if len(small):
        out.append(f"Jenis {', '.join(small.index)} memiliki sampel uji < {MIN_TYPE_N} flow; "
                   "recall-nya tidak dapat dijadikan kesimpulan.")
    out.append("Hasil berasal dari satu model (satu seed). Untuk klaim perbandingan model, "
               "gunakan rata-rata ± std dari beberapa seed.")
    return out
