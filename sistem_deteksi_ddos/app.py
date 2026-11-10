"""
app.py — Sistem Deteksi Anomali Serangan DDoS (Isolation Forest & LOF), Flask + Python.
Jalankan:  python app.py   lalu buka http://127.0.0.1:5000

Halaman utama (sesuai rancangan tampilan di folder tampilan/):
  Dashboard           : vonis per ambang, kurva recall-FPR, kecepatan, macro recall, tabel empat kolom
  Uji Model           : unggah + pilih model + opsi, lalu hasil deteksi per flow pada halaman yang sama
  Perbandingan Model  : IF vs LOF pada kedua ambang (metrik, confusion matrix, recall per jenis, laporan PDF)
  Pengaturan          : aturan yang dipakai tiap model tersimpan + pembersihan sesi
  Riwayat             : hasil pengujian tersimpan, dibuka ulang atau dibandingkan tanpa unggah ulang

Satu kali pengujian menghasilkan prediksi pada SELURUH target FPR yang dimiliki model: skor anomali
dihitung sekali, lalu dibandingkan dengan tiap ambang. Pengujian TIDAK melatih apa pun.
"""
import io
import os
import re
import glob
import json
import math
import time
import base64
import textwrap
import threading
import traceback
import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from matplotlib.backends.backend_pdf import PdfPages
from flask import (Flask, render_template, request, redirect, url_for, flash, send_file, abort, jsonify)

import core

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("DDOS_MODEL_DIR", os.path.join(BASE, "models"))
UPLOAD_DIR = os.environ.get("DDOS_UPLOAD_DIR", os.path.join(BASE, "uploads"))
RUN_DIR = os.environ.get("DDOS_RUN_DIR", os.path.join(BASE, "runs"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

APP_TITLE = ("Penerapan Algoritma Isolation Forest untuk Deteksi Anomali "
             "Trafik Jaringan Serangan Distributed Denial of Service (DDoS)")
ALGO_NAME = {"IF": "Isolation Forest", "LOF": "Local Outlier Factor"}
ALGO_DESC = {"IF": "Deteksi anomali berbasis<br>pohon keputusan", "LOF": "Deteksi anomali berbasis<br>kepadatan data"}
OVERLAP_BLOCK = 0.5             # file deteksi ditolak jika >= 50% barisnya identik dengan data latih
CHUNK = 20_000                  # skor dihitung per potongan agar kemajuan bisa ditampilkan
PER_PAGE_HASIL = 10              # rincian per flow: 10 baris per halaman

app = Flask(__name__)
app.secret_key = "ganti-dengan-kunci-acak"
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024          # unggah maks. 1 GB

LOCK = threading.RLock()
# D = keadaan bagian DETEKSI: berkas uji yang dimuat beserta hasilnya. Sistem ini tidak melatih model.
D = {"df": None, "fname": None, "fsize": 0, "runs": {}, "active": None, "t_load": 0.0, "max_rows": None, "fpr": None,
     "pending": None, "pending_file": None}
SET = {"fpr": 0.05, "active": {}}
JOB = {"state": "idle", "progress": 0.0, "stage": "", "started": None, "message": None, "category": "success",
       "control": "run", "cancelable": False, "paused_at": None, "paused_total": 0.0}

NAV = [("dashboard", "Dashboard", "home"), ("uji", "Uji Model", "clipboard"),
       ("perbandingan", "Perbandingan Model", "compare"), ("pengaturan", "Pengaturan", "gear"),
       ("riwayat", "Riwayat", "clock")]


class UserError(Exception):
    """Galat yang pesannya aman ditampilkan kepada pengguna."""


class JobStopped(Exception):
    """Proses dihentikan pengguna lewat tombol Hentikan."""


# ============================================================================ format angka (gaya Indonesia)
@app.template_filter("fmt")
def fmt(n):
    try:
        return f"{int(round(float(n))):,}".replace(",", ".")
    except (TypeError, ValueError):
        return n


@app.template_filter("dec")
def dec(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x):,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


@app.template_filter("pctc")
def pctc(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x * 100:.{d}f}".replace(".", ",") + "%"


@app.template_filter("hms")
def hms(sec):
    """Format jam digital, dipakai untuk penghitung waktu berjalan pada banner proses."""
    sec = int(round(sec or 0))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


@app.template_filter("dur")
def dur(sec):
    """Durasi dengan satuan tertulis: '0,84 detik', '52 detik', '3 menit 32 detik', '1 jam 5 menit'."""
    sec = float(sec or 0)
    if sec < 1:
        return f"{sec:.2f}".replace(".", ",") + " detik"
    s = int(round(sec))
    h, m, r = s // 3600, s % 3600 // 60, s % 60
    if h:
        return f"{h} jam {m} menit"
    if m:
        return f"{m} menit {r} detik"
    return f"{r} detik"


@app.template_filter("dtk")
def dtk(sec):
    """Waktu prediksi: selalu satu angka di belakang koma selama masih di bawah satu menit."""
    sec = float(sec or 0)
    return (f"{sec:.1f}".replace(".", ",") + " detik") if sec < 60 else dur(sec)


def ratio_txt(v):
    """Metrik proporsi ditulis dua cara: nilai 0–1 (gaya laporan) dan padanan persennya."""
    d = dec(v)
    return d if d == "-" else f"{d} ({pctc(v)})"


def auc_txt(v, how):
    """AUC tidak bersatuan. Persennya ditulis bersama artinya supaya tidak terbaca sebagai Accuracy."""
    d = dec(v)
    return d if d == "-" else f"{d} ({how} {pctc(v)})"


def mbsize(n):
    return f"{n / 1048576:.1f}".replace(".", ",") + " MB"


# ============================================================================ proses latar belakang
def set_stage(text, frac=None):
    with LOCK:
        JOB["stage"] = text
        if frac is not None:
            JOB["progress"] = float(max(0.0, min(1.0, frac)))


def job_control():
    """Titik periksa di sela proses: menahan selama dijeda, melempar JobStopped saat dihentikan."""
    while True:
        with LOCK:
            c = JOB["control"]
        if c == "stop":
            raise JobStopped()
        if c != "pause":
            return
        time.sleep(0.2)


def job_elapsed(now=None):
    """Waktu berjalan tanpa menghitung lama jeda."""
    now = now or time.time()
    if not JOB["started"]:
        return 0.0
    paused = JOB["paused_total"] + ((now - JOB["paused_at"]) if JOB["paused_at"] else 0.0)
    return max(0.0, now - JOB["started"] - paused)


def run_job(fn, *args, cancelable=False):
    """Menjalankan fn(*args) di thread terpisah. Satu proses pada satu waktu.
    cancelable=True hanya untuk proses yang punya titik periksa job_control()."""
    with LOCK:
        if JOB["state"] == "running":
            return False
        JOB.update(state="running", progress=0.02, stage="Memulai...", started=time.time(), message=None,
                   control="run", cancelable=bool(cancelable), paused_at=None, paused_total=0.0)

    def target():
        try:
            msg = fn(*args)
            with LOCK:
                JOB.update(state="done", progress=1.0, message=msg, category="success")
        except JobStopped:
            with LOCK:
                JOB.update(state="done", message="Pengujian dihentikan. Hasil yang sudah selesai tetap tersimpan.",
                           category="warning")
        except UserError as e:
            with LOCK:
                JOB.update(state="error", message=str(e), category="danger")
        except Exception as e:                                        # noqa: BLE001
            traceback.print_exc()
            with LOCK:
                JOB.update(state="error", message=f"Proses gagal: {e}", category="danger")
        finally:
            with LOCK:
                JOB.update(control="run", cancelable=False, paused_at=None)

    threading.Thread(target=target, daemon=True).start()
    return True


@app.route("/job/<aksi>", methods=["POST"])
def job_kontrol(aksi):
    """Jeda, lanjutkan, atau hentikan proses yang sedang berjalan."""
    with LOCK:
        if JOB["state"] != "running" or not JOB["cancelable"]:
            return jsonify(ok=False, control=JOB["control"])
        now = time.time()
        if aksi == "pause" and JOB["control"] == "run":
            JOB.update(control="pause", paused_at=now)
        elif aksi == "resume" and JOB["control"] == "pause":
            JOB.update(control="run", paused_total=JOB["paused_total"] + (now - (JOB["paused_at"] or now)),
                       paused_at=None)
        elif aksi == "stop":
            if JOB["paused_at"]:
                JOB["paused_total"] += now - JOB["paused_at"]
            JOB.update(control="stop", paused_at=None)
        return jsonify(ok=True, control=JOB["control"])


@app.route("/status")
def status():
    with LOCK:
        return jsonify(state=JOB["state"], progress=JOB["progress"], stage=JOB["stage"], elapsed=job_elapsed(),
                       control=JOB["control"], cancelable=JOB["cancelable"])


@app.context_processor
def inject():
    msg = None
    with LOCK:
        if JOB["state"] in ("done", "error") and JOB["message"]:
            msg = (JOB["category"], JOB["message"])
            JOB["message"] = None
    return {"NAV": NAV, "ALGO_NAME": ALGO_NAME, "JOB": JOB, "job_msg": msg}


# ============================================================================ model tersimpan
@lru_cache(maxsize=4)
def _load_bundle(path, mtime):
    return core.load_bundle(path)


def bundle_files():
    return core.list_bundles(MODEL_DIR)


def get_bundle(key):
    path = bundle_files().get(key)
    if not path:
        return None
    return _load_bundle(path, os.path.getmtime(path))


META = {}


def model_meta(key):
    """Metadata ringkas model (tanpa menahan objek model besar di memori). Disimpan juga di <model>.meta.json."""
    path = bundle_files()[key]
    mt = os.path.getmtime(path)
    hit = META.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    side = path[:-len(".joblib")] + ".meta.json"
    info = None
    if os.path.exists(side) and os.path.getmtime(side) >= mt:
        try:
            info = json.load(open(side))
        except (OSError, ValueError):
            info = None
    if info is None:
        try:
            b = core.load_bundle(path)
            info = core.bundle_info(key, b)
            info["thresholds"] = [[float(f), float(v)] for f, v in info["thresholds"]]
            info["seed"] = None if info.get("seed") is None else int(info["seed"])
            info["params"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                              for k, v in (info.get("params") or {}).items()}
            info["features"] = list(b["features"])
            del b
            try:
                json.dump(info, open(side, "w"))
            except OSError:
                pass
        except Exception as e:                                        # noqa: BLE001
            info = {"key": key, "error": str(e)}
    META[key] = (mt, info)
    return info


def bundle_rows():
    return [model_meta(k) for k in bundle_files()]


def algo_keys():
    out = {"IF": [], "LOF": []}
    for k in bundle_files():
        m = model_meta(k)
        if "error" in m:
            continue
        out.setdefault(m["algo"], []).append((m["seed"], k))
    return {a: [k for _, k in sorted(v, key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0))]
            for a, v in out.items()}


def active_key(algo):
    keys = algo_keys().get(algo, [])
    if not keys:
        return None
    sel = SET["active"].get(algo)
    if sel in keys:
        return sel
    other = "LOF" if algo == "IF" else "IF"
    seeds_other = {model_meta(k)["seed"] for k in algo_keys().get(other, [])} - {None}
    for k in keys:
        if model_meta(k)["seed"] in seeds_other:
            return k
    return keys[0]


def fpr_options():
    sets = []
    for a in ("IF", "LOF"):
        k = active_key(a)
        if k:
            sets.append({float(f) for f, _ in model_meta(k)["thresholds"]})
    if not sets:
        return [0.05]
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common or set.union(*sets), reverse=False)


def current_fpr():
    opts = fpr_options()
    return SET["fpr"] if SET["fpr"] in opts else (0.05 if 0.05 in opts else opts[0])




PREP_LABEL = {"raw": "Tanpa transformasi", "slog": "Signed log1p",
              "slog+scaler": "Signed log1p + StandardScaler"}
PARAM_LABEL = {"n_estimators": "Jumlah pohon", "max_samples": "Sampel per pohon",
               "max_features": "Fitur per pohon", "log_input": "Transformasi log",
               "n_neighbors": "Jumlah tetangga (k)", "contamination": "Contamination"}


def param_text(k, v, n_features):
    """Nilai parameter beserta satuannya. max_features berupa pecahan, jadi dijabarkan ke jumlah kolom."""
    if isinstance(v, bool):
        return "Ya" if v else "Tidak"
    num = isinstance(v, (int, float))
    if k == "max_features" and num and 0 < float(v) <= 1 and n_features:
        return f"{dec(float(v), 2)} ({int(round(float(v) * n_features))} dari {n_features} kolom)"
    if k == "max_samples" and num:
        return f"{fmt(v)} baris"
    if k == "n_estimators" and num:
        return f"{fmt(v)} pohon"
    if k == "n_neighbors" and num:
        return f"{fmt(v)} titik"
    return v




# ============================================================================ pengaman: data latih terdaftar
@lru_cache(maxsize=8)
def _fp_cached(path, mtime):
    return core.load_fingerprint(path)


def matching_fingerprints(features):
    out = []
    for p in sorted(glob.glob(os.path.join(MODEL_DIR, "*.fp.npz"))):
        fp = _fp_cached(p, os.path.getmtime(p))
        if fp and set(fp["features"]) == set(features):
            out.append(fp)
    return out


# ============================================================================ inti deteksi (TANPA pelatihan)
def run_detection(df, name, key, b, fpr, max_rows, progress=None, seed=42):
    """Mendeteksi pada FILE BARU memakai model tersimpan. Tidak ada pelatihan di sini.
    Mengembalikan (hasil, None) atau (None, pesan_galat)."""
    label = core.find_label(df)
    missing = [f for f in b["features"] if f not in df.columns]
    if missing:
        return None, (f"File baru kekurangan {len(missing)} kolom fitur yang dibutuhkan model {key}, "
                      f"mis.: {', '.join(missing[:5])}. Format file harus sama dengan dataset latih model.")
    if b.get("trained_on") and name == b["trained_on"]:
        return None, (f"File '{name}' adalah dataset latih model {key}. Pengujian hanya untuk file BARU. "
                      f"Gunakan file uji yang terpisah dari data latih.")
    if fpr not in b["thresholds"]:
        return None, f"Model {key} tidak punya threshold untuk FPR {fpr:.0%}."

    n_file = len(df)
    labels_all = core.normalize_labels(df[label]) if label else None
    idx = core.sample_rows(labels_all, n_file, max_rows, seed)
    sampled = len(idx) < n_file
    if sampled:
        df = df.iloc[idx]

    X, labels, log, stats = core.preprocess(df, label, b["features"], drop_duplicates=False)

    overlap, fps = None, matching_fingerprints(b["features"])
    for fp in fps:
        ov = core.overlap_ratio(core.row_hashes(X[fp["features"]].values), fp["hashes"])
        overlap = ov if overlap is None else max(overlap, ov)
        if ov >= OVERLAP_BLOCK:
            return None, (f"{ov:.0%} baris pada file '{name}' identik dengan data latih ({fp['source']}). "
                          f"Pengujian hanya menerima file BARU yang bukan data latih.")

    arr = X.values
    n = len(arr)
    score = np.empty(n)
    t_pred = 0.0
    for i in range(0, n, CHUNK):
        job_control()                      # titik periksa jeda/hentikan, satu kali per potongan
        s, t = core.anomaly_score(b, arr[i:i + CHUNK])
        score[i:i + len(s)] = s
        t_pred += t
        if progress:
            progress(min(1.0, (i + len(s)) / n))
    lab_vals = None if labels is None else labels.values
    # Skor dihitung SEKALI. Tiap target FPR hanya membandingkan skor yang sama dengan ambangnya masing-masing,
    # jadi seluruh ambang tersedia tanpa biaya hitung tambahan.
    by_fpr = {}
    for f, th in sorted(b["thresholds"].items()):
        p = (score > float(th)).astype(int)
        e = {"thr": float(th), "pred": p, "metrics": None, "cm": None, "per": None}
        if lab_vals is not None:
            e["metrics"], e["cm"], e["per"] = core.evaluate(lab_vals, p, score)
        by_fpr[float(f)] = e
    r = {"algo": b["model_name"], "key": key, "fname": name, "score": score, "by_fpr": by_fpr,
         "rows": idx + 1, "t_pred": t_pred, "labels": lab_vals, "n_file": n_file, "sampled": sampled,
         "stats": stats, "overlap": overlap, "guard_fp": bool(fps), "id": time.time()}
    return apply_view(r, fpr), None


def apply_view(r, fpr):
    """Menyalin hasil pada satu target FPR ke tingkat atas, supaya halaman cukup membaca r["pred"] dan r["thr"]."""
    f = float(fpr)
    if f not in r["by_fpr"]:
        f = min(r["by_fpr"], key=lambda x: abs(x - f))
    e = r["by_fpr"][f]
    r.update(fpr=f, thr=e["thr"], pred=e["pred"], metrics=e["metrics"], cm=e["cm"], per=e["per"])
    return r


def view_fpr():
    """Target FPR yang sedang ditampilkan untuk hasil yang ada di memori."""
    runs = ordered_runs()
    return runs[0][1]["fpr"] if runs else current_fpr()


def all_fprs():
    """Seluruh target FPR yang dimiliki hasil saat ini (urut naik)."""
    runs = ordered_runs()
    return sorted(runs[0][1]["by_fpr"]) if runs else fpr_options()


def model_size(key):
    path = bundle_files().get(key)
    try:
        return os.path.getsize(path) if path else 0
    except OSError:
        return 0


def model_strip(key):
    """Identitas model untuk pita di atas halaman: jumlah data acuan, parameter inti, dan ukuran berkas."""
    m = model_meta(key) if key else {"error": "belum ada"}
    if "error" in m:
        return {"key": key, "ok": False}
    p = m.get("params") or {}
    if m.get("algo") == "IF":
        inti = f"{fmt(p.get('n_estimators', 0))} pohon"
    else:
        inti = f"k = {fmt(p.get('n_neighbors', 0))}"
    return {"key": key, "ok": True, "algo": m.get("algo"), "name": ALGO_NAME.get(m.get("algo"), m.get("algo")),
            "seed": m.get("seed"), "n_train": m.get("n_train"), "inti": inti, "size": mbsize(model_size(key))}


def acuan_terbesar():
    """Jumlah data acuan terbesar di antara seluruh model tersimpan (pembanding untuk peringatan)."""
    vals = [model_meta(k).get("n_train") or 0 for k in bundle_files() if "error" not in model_meta(k)]
    return max(vals) if vals else 0


# ============================================================================ riwayat pengujian (folder runs/)
def _per_rows(per):
    return [] if per is None else [[str(i), int(row["n"]), float(row["Recall"])] for i, row in per.iterrows()]


def _clean(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def riwayat_simpan():
    """Menyimpan skor dan ringkasan satu pengujian ke folder runs/ supaya bisa dibuka lagi tanpa unggah ulang."""
    runs = ordered_runs()
    if not runs:
        return None
    base = runs[0][1]
    rid = time.strftime("%Y%m%d-%H%M%S")
    arrays = {"rows": np.asarray(base["rows"])}
    if base["labels"] is not None:
        arrays["labels"] = np.asarray(base["labels"]).astype(str)
    rec = {"id": rid, "created": time.strftime("%Y-%m-%d %H:%M"), "fname": base["fname"],
           "n_rows": int(len(base["pred"])), "n_file": int(base["n_file"]), "sampled": bool(base["sampled"]),
           "max_rows": D["max_rows"], "fpr_shown": float(base["fpr"]), "fprs": sorted(base["by_fpr"]),
           "n_features": int(base["stats"]["n_features"]), "has_labels": base["labels"] is not None, "algos": {}}
    for a, r in runs:
        arrays[f"score_{a}"] = r["score"].astype("float32")
        m = model_meta(r["key"])
        rec["algos"][a] = {
            "key": r["key"], "name": ALGO_NAME[a], "seed": m.get("seed"), "n_train": m.get("n_train"),
            "params": m.get("params"), "size": model_size(r["key"]), "t_pred": float(r["t_pred"]),
            "t_total": float(r.get("t_total", r["t_pred"])),
            "by_fpr": {str(f): {"thr": e["thr"],
                                "metrics": None if e["metrics"] is None else {k: _clean(v) for k, v in e["metrics"].items()},
                                "cm": None if e["cm"] is None else [[int(x) for x in row] for row in e["cm"]],
                                "per": _per_rows(e["per"])}
                       for f, e in sorted(r["by_fpr"].items())}}
    np.savez_compressed(os.path.join(RUN_DIR, f"{rid}.npz"), **arrays)
    with open(os.path.join(RUN_DIR, f"{rid}.json"), "w") as fh:
        json.dump(rec, fh)
    return rid


def riwayat_list():
    out = []
    for p in sorted(glob.glob(os.path.join(RUN_DIR, "*.json")), reverse=True):
        try:
            with open(p) as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return out


def riwayat_get(rid):
    p = os.path.join(RUN_DIR, f"{os.path.basename(str(rid))}.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def riwayat_hapus(rid):
    for ext in (".json", ".npz"):
        try:
            os.remove(os.path.join(RUN_DIR, f"{os.path.basename(str(rid))}{ext}"))
        except OSError:
            pass


def riwayat_ukuran():
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(RUN_DIR, "*")) if os.path.isfile(p))


def riwayat_muat(rid):
    """Memuat kembali hasil tersimpan ke memori, sehingga Uji Model dan Perbandingan menampilkannya lagi."""
    rec = riwayat_get(rid)
    if rec is None:
        raise UserError("Riwayat tidak ditemukan.")
    z = np.load(os.path.join(RUN_DIR, f"{os.path.basename(str(rid))}.npz"), allow_pickle=True)
    rows = z["rows"]
    labels = z["labels"].astype(str) if "labels" in z.files else None
    newruns = {}
    for a, info in rec["algos"].items():
        score = z[f"score_{a}"].astype("float64")
        by = {}
        for fs, e in info["by_fpr"].items():
            p = (score > float(e["thr"])).astype(int)
            item = {"thr": float(e["thr"]), "pred": p, "metrics": None, "cm": None, "per": None}
            if labels is not None:
                item["metrics"], item["cm"], item["per"] = core.evaluate(labels, p, score)
            by[float(fs)] = item
        r = {"algo": a, "key": info["key"], "fname": rec["fname"], "score": score, "by_fpr": by, "rows": rows,
             "t_pred": info["t_pred"], "t_total": info.get("t_total", info["t_pred"]), "labels": labels,
             "n_file": rec["n_file"], "sampled": rec["sampled"], "stats": {"n_features": rec["n_features"]},
             "overlap": None, "guard_fp": False, "id": rec["id"]}
        newruns[a] = apply_view(r, rec["fpr_shown"])
    with LOCK:
        D.update(df=None, fname=rec["fname"], fsize=0, runs=newruns,
                 active=("IF" if "IF" in newruns else next(iter(newruns))), t_load=0.0,
                 max_rows=rec.get("max_rows"), fpr=rec["fpr_shown"], pending=None, pending_file=None)
    return rec


def job_detect(algos, upload_path, upload_name, upload_size, fpr, max_rows):
    """Proses latar belakang: (opsional) membaca file baru lalu mendeteksi dengan model terpilih."""
    new_file = upload_path is not None
    if new_file:
        set_stage("Membaca file CSV...", 0.03)
        t0 = time.time()
        df = core.read_csv(upload_path)
        job_control()
        with LOCK:
            D.update(df=df, fname=upload_name, fsize=upload_size, runs={}, active=None, t_load=time.time() - t0,
                     max_rows=max_rows, fpr=fpr, pending_file=None)
    else:
        with LOCK:
            if D["df"] is None:
                raise UserError("Unggah file CSV uji terlebih dahulu. Hasil dari Riwayat hanya dapat dilihat, "
                                "tidak dapat diuji ulang tanpa berkasnya.")
            if D["max_rows"] != max_rows:                   # hanya batas baris yang membuat hasil lama tidak berlaku
                D.update(runs={}, active=None, max_rows=max_rows)
            D["fpr"] = fpr
            for r in D["runs"].values():                    # ambang yang ditampilkan boleh berubah tanpa hitung ulang
                apply_view(r, fpr)
    n_alg = len(algos)
    for i, algo in enumerate(algos):
        job_control()
        key = active_key(algo)
        b = get_bundle(key) if key else None
        if b is None:
            raise UserError(f"Model {ALGO_NAME[algo]} belum tersedia. Salin file .joblib ke folder models/ atau latih model di Pengaturan.")
        set_stage(f"Mendeteksi dengan {ALGO_NAME[algo]} ({key})...", 0.08 + 0.9 * i / n_alg)
        t0 = time.time()
        r, err = run_detection(D["df"], D["fname"], key, b, fpr, D["max_rows"],
                               progress=lambda f, i=i: set_stage(JOB["stage"], 0.08 + 0.9 * (i + f) / n_alg))
        if err:
            raise UserError(err)
        r["t_total"] = (time.time() - t0) + (D["t_load"] if (new_file and i == 0) else 0.0)
        with LOCK:
            D["runs"][algo] = r
            D["active"] = algo
    names = " dan ".join(ALGO_NAME[a] for a in algos)
    n = len(D["runs"][algos[-1]]["pred"])
    nf = len(all_fprs())
    try:
        riwayat_simpan()
    except Exception:                                                 # noqa: BLE001
        traceback.print_exc()                                         # riwayat gagal disimpan tidak membatalkan hasil
    return (f"Deteksi selesai: {fmt(n)} flow diperiksa dengan {names} pada {nf} target FPR. "
            f"Hasilnya tersimpan di Riwayat.")


# ============================================================================ bantuan tampilan
def nice_step(vmax, nticks):
    raw = max(vmax, 1) / nticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def heat(v, top):
    t = 0.0 if top <= 0 else min(1.0, v / top)
    c0, c1 = (226, 237, 252), (22, 86, 168)
    rgb = tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", ("#ffffff" if t > 0.42 else "#17315c")


CM_TAGS = (("True Negative", "False Positive"), ("False Negative", "True Positive"))
CM_HINTS = (("Normal terdeteksi normal", "Normal dikira serangan"), ("Serangan lolos", "Serangan terdeteksi"))


def cm_view(cm, model_name):
    """Data confusion matrix untuk tampilan: warna sel (peta panas biru) dan skala bar warna."""
    vmax = int(cm.max()) or 1
    step = nice_step(vmax, 3)
    k = max(1, math.ceil(vmax / step))
    top = step * k
    total = int(cm.sum()) or 1
    cells = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            v = int(cm[i][j])
            bg, fg = heat(v, top)
            cells[i][j] = {"v": v, "pct": v / total, "bg": bg, "fg": fg,
                           "tag": CM_TAGS[i][j], "hint": CM_HINTS[i][j], "ok": i == j}
    return {"cells": cells, "ticks": [int(i * step) for i in range(k + 1)],
            "model_name": model_name, "total": total}




def window(page, pages, width=5):
    start = max(1, min(page - width // 2, pages - width + 1))
    return list(range(start, min(pages, start + width - 1) + 1))


def ordered_runs():
    return [(a, D["runs"][a]) for a in ("IF", "LOF") if a in D["runs"]]


def label_bin(lbl):
    return "Normal" if lbl == core.BENIGN else "DDoS"


def table_rows(sel, runs, labels):
    rows = []
    for i in sel:
        row = {"no": int(runs[0][1]["rows"][i]), "pred": {}, "score": {}}
        if labels is not None:
            row["orig"] = str(labels[i])
            row["actual"] = label_bin(labels[i])
        for a, r in runs:
            row["pred"][a] = "DDoS" if r["pred"][i] == 1 else "Normal"
            row["score"][a] = float(r["score"][i])
        rows.append(row)
    return rows


def file_chip():
    """Info file untuk chip di form unggah (termasuk file yang sedang diunggah/diproses)."""
    pf = D["pending_file"] if JOB["state"] == "running" else None
    if D["df"] is not None:
        return {"has_file": True, "fname": D["fname"], "fsize": mbsize(D["fsize"])}
    if pf:
        return {"has_file": True, "fname": pf[0], "fsize": mbsize(pf[1])}
    return {"has_file": False, "fname": None, "fsize": ""}


# ============================================================================ halaman: Dashboard (hanya menampilkan)
ALGO_COLOR = {"IF": "#2670d4", "LOF": "#1f9460"}


def roc_points(r, max_fpr=0.08, n=70):
    """Kurva recall (TPR) terhadap FPR aktual, dihitung dari skor anomali dan label yang sudah ada."""
    if r["labels"] is None:
        return []
    y = (np.asarray(r["labels"]) != core.BENIGN).astype(int)
    if len(np.unique(y)) < 2:
        return []
    f, t, _ = roc_curve(y, r["score"])
    keep = f <= max_fpr
    f, t = f[keep], t[keep]
    if len(f) > n:
        sel = np.linspace(0, len(f) - 1, n).astype(int)
        f, t = f[sel], t[sel]
    return [(float(a), float(b)) for a, b in zip(f, t)]


def curve_svg(series, max_fpr=0.08):
    """Kurva recall terhadap FPR aktual untuk kedua model, beserta titik operasi tiap target FPR."""
    W, H, ml, mr, mt, mb = 640, 300, 56, 24, 22, 54
    pw, ph = W - ml - mr, H - mt - mb
    px = lambda v: ml + pw * min(v, max_fpr) / max_fpr
    py = lambda v: mt + ph - ph * v
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-label="Kurva recall terhadap FPR aktual">']
    for i in range(5):
        y = mt + ph - ph * i / 4
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W - mr}" y2="{y:.1f}" stroke="#e6edf6" stroke-width="1"/>')
        out.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#6b7893">{i * 25}%</text>')
    for i in range(5):
        x = ml + pw * i / 4
        out.append(f'<text x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle" font-size="11" '
                   f'fill="#6b7893">{max_fpr * 100 * i / 4:.0f}%</text>')
    out.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 8}" text-anchor="middle" font-size="11.5" '
               f'fill="#52617f">FPR aktual (alarm palsu)</text>')
    out.append(f'<text x="{ml - 46}" y="{mt - 6}" font-size="11.5" fill="#52617f">Recall</text>')
    for lab, col, pts, ops in series:
        if pts:
            d = " ".join(f"{'M' if i == 0 else 'L'}{px(a):.1f} {py(b):.1f}" for i, (a, b) in enumerate(pts))
            out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.2" stroke-linejoin="round"/>')
    for lab, col, pts, ops in series:
        for f, rec, tag, below in ops:
            x, y = px(f), py(rec)
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}" stroke="#ffffff" stroke-width="2"/>')
            ty = y + 20 if below else y - 11
            out.append(f'<text x="{x:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="11" fill="{col}">{tag}</text>')
    out.append("</svg>")
    return "".join(out)


BARIS_METRIK = [
    {"label": "Accuracy", "key": "Accuracy", "tetap": False},
    {"label": "Precision", "key": "Precision", "tetap": False},
    {"label": "Recall", "key": "Recall", "tetap": False},
    {"label": "F1-Score", "key": "F1", "tetap": False},
    {"label": "FPR aktual", "key": "FPR", "tetap": False},
    {"label": "Macro recall per jenis", "key": "Macro_Recall_Mayor", "tetap": False},
    {"label": "ROC-AUC", "key": "ROC_AUC", "tetap": True},
    {"label": "PR-AUC", "key": "PR_AUC", "tetap": True},
]


def kolom_hasil(fprs, algos, isi=False):
    """Kolom tabel hasil: satu kolom per pasangan (model, target FPR), urut dari ambang paling longgar."""
    out = []
    for f in fprs:
        for a in algos:
            c = {"algo": a, "fpr": f, "label": f"{a} @ {pctc(f, 0)}", "color": ALGO_COLOR[a]}
            if isi:
                c["vals"] = {b["key"]: nilai(a, f, b["key"]) for b in BARIS_METRIK}
            out.append(c)
    return out


def nilai(algo, fpr, key):
    r = D["runs"].get(algo)
    e = r["by_fpr"].get(fpr) if r else None
    if not e or e["metrics"] is None:
        return None
    v = e["metrics"].get(key)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def vonis(fprs):
    """Model mana yang unggul pada tiap target FPR, beserta alasan singkatnya."""
    runs = ordered_runs()
    out = []
    for f in fprs:
        cand = [(a, r["by_fpr"][f]["metrics"]) for a, r in runs
                if f in r["by_fpr"] and r["by_fpr"][f]["metrics"] is not None]
        if len(cand) < 2:
            continue
        cand.sort(key=lambda x: -x[1]["F1"])
        (wa, wm), (la, lm) = cand[0], cand[1]
        out.append({"fpr": f, "algo": wa, "name": ALGO_NAME[wa], "color": ALGO_COLOR[wa],
                    "f1": wm["F1"], "f1_lain": lm["F1"], "recall": wm["Recall"], "recall_lain": lm["Recall"],
                    "fn": wm["FN"], "n_att": wm["FN"] + wm["TP"], "lain": ALGO_NAME[la]})
    return out


@app.route("/")
def dashboard():
    """Ringkasan penelitian: model mana unggul pada tiap ambang, kurva recall-FPR, kecepatan, dan angka lengkap.
    Halaman ini tidak mengubah apa pun — pengujian dijalankan di halaman Uji Model."""
    runs = ordered_runs()
    algos = [a for a, _ in runs]
    strips = [model_strip(active_key(a)) for a in ("IF", "LOF")]
    besar = acuan_terbesar()
    kecil = [s for s in strips if s["ok"] and (s["n_train"] or 0) < besar]
    ctx = {"nav": "dashboard", "strips": strips, "n_models": len(bundle_files()),
           "acuan_besar": besar, "acuan_kecil": kecil, "fname": D["fname"],
           "run_algos": algos, "ada": bool(runs)}
    if not runs:
        return render_template("dashboard.html", **ctx)

    fprs = sorted(all_fprs(), reverse=True)
    has_labels = runs[0][1]["labels"] is not None
    ctx.update(fprs=fprs, has_labels=has_labels, cols=kolom_hasil(fprs, algos),
               n_rows=len(runs[0][1]["pred"]), sampled=runs[0][1]["sampled"], n_file=runs[0][1]["n_file"],
               speed=[{"algo": a, "name": ALGO_NAME[a], "t": r["t_pred"], "color": ALGO_COLOR[a],
                       "frac": r["t_pred"] / (max(x["t_pred"] for _, x in runs) or 1)} for a, r in runs])
    if has_labels:
        ctx["cols"] = kolom_hasil(fprs, algos, isi=True)
        ctx["baris"] = BARIS_METRIK
        ctx["vonis"] = vonis(fprs)
        ctx["macro"] = [{"algo": a, "name": ALGO_NAME[a], "color": ALGO_COLOR[a],
                         "v": nilai(a, runs[0][1]["fpr"], "Macro_Recall_Mayor")} for a, _ in runs]
        series = []
        for a, r in runs:
            ops = [(nilai(a, f, "FPR"), nilai(a, f, "Recall"), f"{a} @{pctc(f, 0)}", f < 0.03)
                   for f in fprs if nilai(a, f, "FPR") is not None]
            series.append((a, ALGO_COLOR[a], roc_points(r), ops))
        ctx["curve"] = curve_svg(series)
    return render_template("dashboard.html", **ctx)


# ============================================================================ halaman: Uji Model + aksi Mulai Uji
@app.route("/uji", methods=["GET", "POST"])
def uji():
    """Unggah, pilih model, jalankan — lalu hasil deteksi per flow tampil di halaman yang sama."""
    if request.method == "POST":
        return uji_post()
    feats, cards = [], []
    for a in ("IF", "LOF"):
        k = active_key(a)
        m = model_meta(k) if k else {}
        if k and not feats:
            feats = m.get("features") or []
        cards.append({"algo": a, "name": ALGO_NAME[a], "desc": ALGO_DESC[a], "ok": bool(k), "key": k,
                      "options": [{"key": o, "n": model_meta(o).get("n_train")} for o in algo_keys().get(a, [])],
                      "n_train": m.get("n_train")})
    runs = ordered_runs()
    algos = [a for a, _ in runs]
    sel = D["pending"] if D["pending"] in ("IF", "LOF", "BOTH") else None
    if sel is None:
        sel = "BOTH" if len(algos) > 1 else (algos[0] if algos else next((c["algo"] for c in cards if c["ok"]), "IF"))
    finfo = None
    if D["df"] is not None:                       # dicek terhadap syarat kolom supaya ketahuan sebelum tombol ditekan
        lab = core.find_label(D["df"])
        miss = [f for f in feats if f not in D["df"].columns]
        finfo = {"rows": len(D["df"]), "cols": D["df"].shape[1], "label": lab, "n_missing": len(miss),
                 "missing": miss[:6], "fname": D["fname"], "fsize": mbsize(D["fsize"])}
    ctx = {"nav": "uji", "algos": cards, "selected": sel, "features": feats, "n_features": len(feats),
           "fpr_options": fpr_options(), "fpr_now": D["fpr"] or current_fpr(), "max_rows": D["max_rows"],
           "finfo": finfo, "ada": bool(runs), "acuan_besar": acuan_terbesar(), **file_chip()}
    if not runs:
        return render_template("uji.html", **ctx)

    m = request.args.get("m") if request.args.get("m") in algos else (D["active"] if D["active"] in algos else algos[0])
    flt = request.args.get("f", "all")
    base = D["runs"][m]
    labels = base["labels"]
    has_labels = labels is not None
    pred = base["pred"]
    actual = (np.asarray(labels) != core.BENIGN).astype(int) if has_labels else None
    if flt == "ddos":
        idx = np.where(pred == 1)[0]
    elif flt == "normal":
        idx = np.where(pred == 0)[0]
    elif flt == "salah" and has_labels:
        idx = np.where(actual != pred)[0]
    elif flt == "fp" and has_labels:
        idx = np.where((actual == 0) & (pred == 1))[0]
    elif flt == "fn" and has_labels:
        idx = np.where((actual == 1) & (pred == 0))[0]
    else:
        flt, idx = "all", np.arange(len(pred))
    total = len(idx)
    pages = max(1, math.ceil(total / PER_PAGE_HASIL))
    page = min(max(1, int(request.args.get("p", 1))), pages)
    sel_idx = idx[(page - 1) * PER_PAGE_HASIL: page * PER_PAGE_HASIL]
    n = len(pred)
    summ = {"all": n, "ddos": int(pred.sum()), "normal": int((pred == 0).sum())}
    if has_labels:
        summ.update(salah=int((actual != pred).sum()), fp=int(((actual == 0) & (pred == 1)).sum()),
                    fn=int(((actual == 1) & (pred == 0)).sum()))
    meta = {"fname": base["fname"], "key": base["key"], "name": ALGO_NAME[m], "fpr": base["fpr"], "thr": base["thr"],
            "t": base.get("t_total", base["t_pred"]), "n_file": base["n_file"], "sampled": base["sampled"],
            "n_features": base["stats"]["n_features"],
            "model_txt": " + ".join(D["runs"][a]["key"] for a in algos),
            "t_txt": " + ".join(dtk(D["runs"][a]["t_pred"]) for a in algos),
            "acuan": model_meta(base["key"]).get("n_train")}
    ctx.update(run_algos=algos, m=m, flt=flt, has_labels=has_labels, rows=table_rows(sel_idx, runs, labels),
               total=total, page=page, pages=pages, page_window=window(page, pages),
               start=(page - 1) * PER_PAGE_HASIL + 1 if total else 0, end=min(total, page * PER_PAGE_HASIL),
               summ=summ, meta=meta, thr_by={a: D["runs"][a]["thr"] for a in algos},
               strips=[model_strip(D["runs"][a]["key"]) for a in algos])
    return render_template("uji.html", **ctx)


def uji_post():
    if JOB["state"] == "running":
        flash("Masih ada proses yang berjalan. Tunggu sampai selesai.", "warning")
        return redirect(url_for("uji"))
    pilih = request.form.get("algo", "IF")
    algos = ["IF", "LOF"] if pilih == "BOTH" else [pilih if pilih in ALGO_NAME else "IF"]
    for a in ("IF", "LOF"):                       # seed dipilih langsung di kartu modelnya
        v = request.form.get(f"key_{a}")
        if v and v in algo_keys().get(a, []):
            SET["active"][a] = v
    try:
        fpr = float(request.form["fpr"]) if "fpr" in request.form else (D["fpr"] or current_fpr())
    except ValueError:
        fpr = current_fpr()
    mr = request.form.get("max_rows", "").strip()
    max_rows = int(mr) if mr.isdigit() and int(mr) > 0 else None

    f = request.files.get("file")
    path = name = None
    size = 0
    if f and f.filename:
        if not f.filename.lower().endswith(".csv"):
            flash("Pilih file .csv.", "danger")
            return redirect(url_for("uji"))
        name = f.filename
        path = os.path.join(UPLOAD_DIR, os.path.basename(name))
        f.save(path)
        size = os.path.getsize(path)
        if "train" in name.lower():
            flash(f"Nama file '{name}' mengandung 'train'. Pastikan ini bukan data latih.", "warning")
    elif D["df"] is None:
        flash("Unggah berkas CSV uji terlebih dahulu.", "danger")
        return redirect(url_for("uji"))
    for a in algos:
        if not active_key(a):
            flash(f"Model {ALGO_NAME[a]} belum tersedia. Salin berkas .joblib ke folder models/.", "danger")
            return redirect(url_for("uji"))
    D["pending"] = pilih if pilih in ("IF", "LOF", "BOTH") else algos[-1]
    D["pending_file"] = (name, size) if path else None
    run_job(job_detect, algos, path, name, size, fpr, max_rows, cancelable=True)
    return redirect(url_for("uji"))


@app.route("/hapus-file", methods=["POST"])
def hapus_file():
    with LOCK:
        D.update(df=None, fname=None, fsize=0, runs={}, active=None, t_load=0.0, max_rows=None, fpr=None,
                 pending=None, pending_file=None)
    return "ok"


# ============================================================================ unduhan hasil per flow
def result_frame():
    runs = ordered_runs()
    base = runs[0][1]
    out = pd.DataFrame({"Baris": base["rows"]})
    if base["labels"] is not None:
        out["Label aktual"] = [label_bin(x) for x in base["labels"]]
        out["Jenis asli"] = base["labels"]
    for a, r in runs:
        out[f"Prediksi {a}"] = np.where(r["pred"] == 1, "DDoS", "Normal")
        out[f"Skor {a}"] = np.round(r["score"], 5)
    return out


@app.route("/hasil/unduh")
def hasil_unduh():
    if not ordered_runs():
        abort(404)
    data = result_frame().to_csv(index=False).encode()
    return send_file(io.BytesIO(data), as_attachment=True, download_name="hasil_deteksi.csv", mimetype="text/csv")


# ============================================================================ halaman: Perbandingan Model
def comparison():
    runs = ordered_runs()
    items = []
    for a, r in runs:
        name = ALGO_NAME[a]
        x = {"algo": a, "name": name, "r": r}
        if r["metrics"] is not None:
            x["cm"] = cm_view(r["cm"], name)
            x["findings"] = core.findings(name, r["metrics"], r["per"], r["score"], r["labels"], r["thr"])
        items.append(x)
    return items


def temuan_ringkas(fprs, algos):
    """Tiga kalimat temuan untuk layar. Uraian lengkapnya ada di laporan PDF."""
    out = []
    v = vonis(fprs)
    if len(v) >= 2:
        a, b = v[0], v[1]
        out.append(f"Pada target FPR {pctc(a['fpr'], 0)} {a['name']} unggul (F1 {pctc(a['f1'], 2)} berbanding "
                   f"{pctc(a['f1_lain'], 2)}), sedangkan pada {pctc(b['fpr'], 0)} {b['name']} yang unggul "
                   f"(recall {pctc(b['recall'], 2)} berbanding {pctc(b['recall_lain'], 2)}).")
    elif v:
        a = v[0]
        out.append(f"Pada target FPR {pctc(a['fpr'], 0)}, {a['name']} unggul dengan F1 {pctc(a['f1'], 2)} "
                   f"berbanding {pctc(a['f1_lain'], 2)}.")
    runs = ordered_runs()
    if len(runs) > 1:
        cepat = min(runs, key=lambda kv: kv[1]["t_pred"])
        lambat = max(runs, key=lambda kv: kv[1]["t_pred"])
        kali = (lambat[1]["t_pred"] / cepat[1]["t_pred"]) if cepat[1]["t_pred"] else 0
        out.append(f"{ALGO_NAME[cepat[0]]} {dec(kali, 1)} kali lebih cepat "
                   f"({dtk(cepat[1]['t_pred'])} berbanding {dtk(lambat[1]['t_pred'])}) dan berkasnya "
                   f"{mbsize(model_size(cepat[1]['key']))} berbanding {mbsize(model_size(lambat[1]['key']))}.")
    mac = [(a, nilai(a, fprs[0], "Macro_Recall_Mayor")) for a in algos]
    mac = [(a, m) for a, m in mac if m is not None]
    if len(mac) > 1:
        mac.sort(key=lambda x: -x[1])
        out.append(f"Macro recall per jenis serangan berbeda jauh: {ALGO_NAME[mac[0][0]]} {pctc(mac[0][1], 2)} "
                   f"berbanding {ALGO_NAME[mac[1][0]]} {pctc(mac[1][1], 2)}, sehingga keduanya saling melengkapi "
                   f"sebagai penyaring awal dan pemeriksa lanjutan.")
    return out


@app.route("/perbandingan")
def perbandingan():
    runs = ordered_runs()
    if not runs:
        return render_template("perbandingan.html", nav="perbandingan", ada=False)
    algos = [a for a, _ in runs]
    fprs = sorted(all_fprs(), reverse=True)
    try:
        shown = float(request.args.get("fpr", runs[0][1]["fpr"]))
    except ValueError:
        shown = runs[0][1]["fpr"]
    if shown not in fprs:
        shown = runs[0][1]["fpr"]
    has_labels = runs[0][1]["labels"] is not None
    ctx = {"nav": "perbandingan", "ada": True, "fname": D["fname"], "run_algos": algos, "fprs": fprs,
           "shown": shown, "has_labels": has_labels, "min_n": core.MIN_TYPE_N,
           "strips": [model_strip(D["runs"][a]["key"]) for a in algos],
           "cols": kolom_hasil(fprs, algos), "n_rows": len(runs[0][1]["pred"]),
           "sizes": {a: mbsize(model_size(D["runs"][a]["key"])) for a in algos},
           "waktu": {a: D["runs"][a]["t_pred"] for a in algos},
           "satu_model": len(algos) < 2}
    seeds = {model_meta(D["runs"][a]["key"]).get("seed") for a in algos}
    acuan = {model_meta(D["runs"][a]["key"]).get("n_train") for a in algos}
    ctx["seed_mismatch"] = (", ".join(f"{ALGO_NAME[a]}: {model_meta(D['runs'][a]['key']).get('seed')}" for a in algos)
                            if len(seeds) > 1 else None)
    ctx["acuan_mismatch"] = (", ".join(f"{ALGO_NAME[a]}: {fmt(model_meta(D['runs'][a]['key']).get('n_train') or 0)}"
                                       for a in algos) if len(acuan) > 1 else None)
    if has_labels:
        ctx["cols"] = kolom_hasil(fprs, algos, isi=True)
        ctx["baris"] = BARIS_METRIK
        ctx["cms"] = [{"algo": a, "name": ALGO_NAME[a], "color": ALGO_COLOR[a],
                       "cm": cm_view(D["runs"][a]["by_fpr"][shown]["cm"], ALGO_NAME[a])} for a in algos]
        allt = {}
        for a in algos:
            per = D["runs"][a]["by_fpr"][shown]["per"]
            for jenis, row in per.iterrows():
                allt.setdefault(jenis, {"n": int(row["n"]), "vals": {}})["vals"][a] = float(row["Recall"])
        ctx["types"] = [{"jenis": j, "n": v["n"], "vals": {a: v["vals"].get(a) for a in algos},
                         "cukup": v["n"] >= core.MIN_TYPE_N}
                        for j, v in sorted(allt.items(), key=lambda kv: -kv[1]["n"])]
        ctx["macro"] = {a: nilai(a, shown, "Macro_Recall_Mayor") for a in algos}
        ctx["temuan"] = temuan_ringkas(fprs, algos)
    return render_template("perbandingan.html", **ctx)


# ============================================================================ laporan PDF perbandingan
PDF_SIZE = (8.27, 11.69)                      # A4 tegak
PDF_INK, PDF_MUTED, PDF_LINE = "#12264a", "#5a6b88", "#d8e2ef"
PDF_SOFT, PDF_RED, PDF_GREEN = "#eef4fc", "#d93a40", "#2f9e68"
PDF_COLOR = {"IF": "#2670d4", "LOF": "#2f9e68"}
L, R = 0.085, 0.915                           # batas kiri dan kanan isi halaman


def _pdf_page(pdf_pages, title, sub=None):
    fig = plt.figure(figsize=PDF_SIZE, facecolor="white")
    fig.text(L, 0.962, "Laporan Perbandingan Model Deteksi Anomali DDoS", size=8, color=PDF_MUTED)
    fig.add_artist(plt.Line2D([L, R], [0.952, 0.952], color=PDF_LINE, lw=1, transform=fig.transFigure))
    fig.text(L, 0.925, title, size=18, weight="bold", color=PDF_INK, va="top")
    y = 0.893
    if sub:
        fig.text(L, y, sub, size=9.5, color=PDF_MUTED, va="top")
        y -= 0.022
    return fig, y - 0.016


def _pdf_text(fig, y, text, size=10, color=None, weight="normal", width=94, indent=0.0, gap=None):
    gap = gap or (size * 0.00175 + 0.0048)
    for line in (textwrap.wrap(text, width) or [""]):
        fig.text(L + indent, y, line, size=size, color=color or PDF_INK, va="top", weight=weight)
        y -= gap
    return y


def _pdf_bullets(fig, y, rows, size=9.5, width=88):
    for t in rows:
        fig.text(L + 0.004, y, "•", size=size, color=PDF_MUTED, va="top")
        y = _pdf_text(fig, y, t, size=size, indent=0.022, width=width) - 0.006
    return y


def _pdf_box(fig, y, height, color=PDF_SOFT, edge=PDF_LINE):
    fig.patches.append(plt.Rectangle((L - 0.02, y - height), (R - L) + 0.04, height, transform=fig.transFigure,
                                     facecolor=color, edgecolor=edge, lw=1, zorder=0))


def _pdf_close(pdf_pages, fig):
    pdf_pages.savefig(fig)
    plt.close(fig)


def _simpulan(items):
    """Kesimpulan ringkas dalam bahasa sehari-hari."""
    out = []
    n = len(items[0]["r"]["pred"])
    out.append(f"Sebanyak {fmt(n)} catatan lalu lintas jaringan (flow) diperiksa oleh "
               f"{' dan '.join(x['name'] for x in items)} pada berkas uji yang sama.")
    if len(items) > 1:
        best = max(items, key=lambda x: x["r"]["metrics"]["F1"])
        fast = min(items, key=lambda x: x["r"]["t_pred"])
        slow = max(items, key=lambda x: x["r"]["t_pred"])
        out.append(f"Dari sisi ketepatan, {best['name']} unggul dengan F1-Score "
                   f"{dec(best['r']['metrics']['F1'])} atau setara {pctc(best['r']['metrics']['F1'])}. "
                   f"F1-Score menggabungkan ketelitian dan kelengkapan deteksi menjadi satu angka.")
        if slow["r"]["t_pred"] > 0 and fast is not slow:
            kali = slow["r"]["t_pred"] / max(fast["r"]["t_pred"], 1e-9)
            out.append(f"Dari sisi kecepatan, {fast['name']} jauh lebih cepat: "
                       f"{dec(fast['r']['t_pred'], 1)} detik berbanding {dec(slow['r']['t_pred'], 1)} detik, "
                       f"atau sekitar {dec(kali, 1)} kali lipat.")
    for x in items:
        m = x["r"]["metrics"]
        out.append(f"{x['name']} melewatkan {fmt(m['FN'])} serangan dan menimbulkan {fmt(m['FP'])} alarm palsu "
                   f"(lalu lintas normal yang dikira serangan).")
    return out


def laporan_pdf(items):
    """Laporan perbandingan lengkap: ringkasan, penjelasan istilah, grafik, dan temuan."""
    buf = io.BytesIO()
    algos = [x["algo"] for x in items]
    n = len(items[0]["r"]["pred"])
    tanggal = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    with PdfPages(buf) as pdf:
        # ---------------------------------------------------------------- 1. sampul & kesimpulan
        fig = plt.figure(figsize=PDF_SIZE, facecolor="white")
        fig.patches.append(plt.Rectangle((0, 0.80), 1, 0.20, transform=fig.transFigure,
                                         facecolor="#132a45", edgecolor="none", zorder=0))
        fig.text(L, 0.925, "Laporan Perbandingan Model", size=26, weight="bold", color="white", va="top")
        fig.text(L, 0.878, "Sistem Deteksi Anomali Trafik Jaringan Serangan DDoS", size=12, color="#b8c6db", va="top")
        fig.text(L, 0.845, "Isolation Forest dan Local Outlier Factor", size=10, color="#8fa6c6", va="top")

        y = 0.755
        _pdf_box(fig, y + 0.018, 0.105)
        info = [("Berkas uji", D["fname"] or "-"), ("Jumlah flow diperiksa", f"{fmt(n)} flow"),
                ("Model dibandingkan", " vs ".join(f"{x['name']} ({x['r']['key']})" for x in items)),
                ("Tanggal laporan", tanggal)]
        for k, v in info:
            fig.text(L, y, k, size=9, color=PDF_MUTED, va="top")
            fig.text(L + 0.22, y, str(v), size=9.5, weight="bold", color=PDF_INK, va="top")
            y -= 0.0235

        y -= 0.025
        y = _pdf_text(fig, y, "Kesimpulan Singkat", size=14, weight="bold") - 0.012
        y = _pdf_bullets(fig, y, _simpulan(items), size=10, width=84) - 0.02

        y = _pdf_text(fig, y, "Cara Membaca Laporan Ini", size=14, weight="bold") - 0.012
        y = _pdf_text(fig, y, "Laporan ini membandingkan dua cara otomatis untuk mengenali serangan DDoS pada "
                              "lalu lintas jaringan. Keduanya tidak dilatih ulang di sini; keduanya hanya diuji "
                              "pada berkas yang sama agar perbandingannya adil. Halaman berikutnya menjelaskan "
                              "arti setiap istilah sebelum angkanya ditampilkan.", size=10, width=92)
        fig.text(L, 0.06, "Catatan: hasil menunjukkan lalu lintas yang dianggap tidak wajar, bukan vonis pasti "
                          "bahwa suatu serangan terjadi.", size=8.5, color=PDF_MUTED, va="top", wrap=True)
        _pdf_close(pdf, fig)

        # ---------------------------------------------------------------- 2. istilah
        fig, y = _pdf_page(pdf, "Arti Istilah yang Dipakai",
                           "Dibaca dulu supaya angka di halaman berikutnya tidak disalahartikan.")
        istilah = [
            ("Flow", "Satu catatan lalu lintas antara dua komputer, berisi ringkasan seperti lama hubungan, "
                     "jumlah paket, dan ukuran data. Satu flow adalah satu baris data."),
            ("Accuracy", "Berapa persen keputusan model yang benar dari seluruh flow. Angka ini bisa menyesatkan "
                         "bila jumlah serangan dan lalu lintas normal timpang."),
            ("Precision", "Dari semua flow yang dituduh serangan, berapa persen yang memang benar serangan. "
                          "Precision rendah berarti petugas sering diganggu alarm palsu."),
            ("Recall", "Dari semua serangan yang sebenarnya ada, berapa persen yang berhasil tertangkap. "
                       "Recall rendah berarti banyak serangan lolos tanpa ketahuan."),
            ("F1-Score", "Gabungan Precision dan Recall menjadi satu angka. Dipakai ketika keduanya sama penting."),
            ("Alarm palsu", "Lalu lintas normal yang dikira serangan. Istilah teknisnya False Positive (FP)."),
            ("Serangan lolos", "Serangan yang dikira normal sehingga tidak terdeteksi. Istilahnya False Negative (FN)."),
            ("Ambang (threshold)", "Batas nilai kecurigaan. Flow dengan nilai di atas ambang dinyatakan serangan. "
                                   "Ambang diambil dari target alarm palsu yang ditetapkan saat model dibuat."),
            ("ROC-AUC", "Peluang sebuah flow serangan mendapat nilai kecurigaan lebih tinggi daripada flow normal "
                        "yang diambil acak. Bukan persentase kebenaran."),
        ]
        for nama, arti in istilah:
            fig.text(L, y, nama, size=10.5, weight="bold", color=PDF_INK, va="top")
            y = _pdf_text(fig, y - 0.0185, arti, size=9.5, color=PDF_MUTED, width=98) - 0.011
        _pdf_close(pdf, fig)

        # ---------------------------------------------------------------- 3. perbandingan metrik
        fig, y = _pdf_page(pdf, "Perbandingan Ketepatan",
                           "Keempat ukuran berikut berskala 0 sampai 1. Makin tinggi makin baik.")
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        keys = ["Accuracy", "Precision", "Recall", "F1"]
        ax = fig.add_axes([L, 0.52, R - L, 0.33])
        w = 0.8 / len(items)
        pos = np.arange(len(labels))
        for i, x in enumerate(items):
            vals = [float(x["r"]["metrics"][k]) for k in keys]
            bars = ax.bar(pos + i * w - 0.4 + w / 2, vals, w * 0.88, label=x["name"], color=PDF_COLOR[x["algo"]])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.018, dec(v), ha="center", size=8.5,
                        weight="bold", color=PDF_INK)
        ax.set_xticks(pos, labels, size=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Nilai (0 = buruk, 1 = sempurna)", size=9, color=PDF_MUTED)
        ax.legend(frameon=False, ncol=len(items), loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(PDF_LINE)
        ax.tick_params(colors=PDF_MUTED, labelsize=9)
        ax.grid(axis="y", color=PDF_LINE, lw=0.7)
        ax.set_axisbelow(True)

        y = 0.455
        y = _pdf_text(fig, y, "Angka lengkap", size=12, weight="bold") - 0.012
        head = ["Ukuran"] + [x["name"] for x in items]
        baris = []
        for lab, k in zip(labels, keys):
            baris.append([lab] + [f"{dec(x['r']['metrics'][k])}  ({pctc(x['r']['metrics'][k])})" for x in items])
        baris.append(["Alarm palsu"] + [f"{fmt(x['r']['metrics']['FP'])} flow" for x in items])
        baris.append(["Serangan lolos"] + [f"{fmt(x['r']['metrics']['FN'])} flow" for x in items])
        baris.append(["Waktu memeriksa"] + [f"{dec(x['r']['t_pred'], 1)} detik" for x in items])
        baris.append(["Ambang dipakai"] + [f"{dec(x['r']['thr'], 4)}" for x in items])
        colx = [L, L + 0.40, L + 0.66]
        for j, h in enumerate(head):
            fig.text(colx[j], y, h, size=9.5, weight="bold", color=PDF_INK, va="top")
        y -= 0.008
        fig.add_artist(plt.Line2D([L, R], [y, y], color=PDF_LINE, lw=1, transform=fig.transFigure))
        y -= 0.018
        for row in baris:
            for j, cell in enumerate(row):
                fig.text(colx[j], y, str(cell), size=9.5, color=PDF_INK if j == 0 else PDF_INK,
                         weight="normal" if j == 0 else "bold", va="top")
            y -= 0.0225
        _pdf_close(pdf, fig)

        # ---------------------------------------------------------------- 4. confusion matrix
        fig, y = _pdf_page(pdf, "Rincian Benar dan Salah",
                           "Empat kotak di bawah memecah seluruh keputusan model menjadi dua yang benar dan dua yang salah.")
        tags = [["Normal dikenali\nnormal (benar)", "Normal dikira\nserangan (salah)"],
                ["Serangan lolos\n(salah)", "Serangan tertangkap\n(benar)"]]
        wax = (R - L - 0.06) / len(items)
        for i, x in enumerate(items):
            cm = x["r"]["cm"]
            ax = fig.add_axes([L + i * (wax + 0.06), 0.55, wax, 0.28])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)
            for a in range(2):
                for b in range(2):
                    v = int(cm[a][b])
                    warna = "white" if v > (cm.max() or 1) * 0.45 else PDF_INK
                    ax.text(b, a - 0.16, fmt(v), ha="center", va="center", size=13, weight="bold", color=warna)
                    ax.text(b, a + 0.22, tags[a][b], ha="center", va="center", size=6.6, color=warna)
            ax.set_xticks([0, 1], ["Ditebak\nnormal", "Ditebak\nserangan"], size=8)
            ax.set_yticks([0, 1], ["Asli\nnormal", "Asli\nserangan"], size=8)
            ax.tick_params(length=0, colors=PDF_MUTED)
            ax.set_title(x["name"], size=11, weight="bold", color=PDF_INK, pad=10)
            for s in ax.spines.values():
                s.set_visible(False)
        y = 0.49
        y = _pdf_text(fig, y, "Cara membacanya", size=12, weight="bold") - 0.012
        y = _pdf_bullets(fig, y, [
            "Dua kotak gelap pada diagonal adalah keputusan yang benar. Makin besar angkanya makin baik.",
            "Kotak kanan atas adalah alarm palsu: lalu lintas normal yang dikira serangan. Terlalu banyak alarm "
            "palsu membuat petugas lelah dan akhirnya mengabaikan peringatan.",
            "Kotak kiri bawah adalah serangan yang lolos. Ini yang paling berbahaya karena serangan berjalan "
            "tanpa ketahuan.",
            "Tidak ada model yang bisa menekan keduanya sekaligus. Menurunkan ambang akan menangkap lebih banyak "
            "serangan tetapi menambah alarm palsu, dan sebaliknya.",
        ], size=9.5, width=92)
        _pdf_close(pdf, fig)

        # ---------------------------------------------------------------- 5. distribusi & kecepatan
        fig, y = _pdf_page(pdf, "Hasil Pemeriksaan dan Kecepatan",
                           "Berapa banyak yang dinyatakan serangan, dan berapa lama pemeriksaannya.")
        ax = fig.add_axes([L, 0.58, (R - L) * 0.52, 0.27])
        kat = ["Normal", "Serangan"]
        w = 0.8 / len(items)
        pos = np.arange(2)
        for i, x in enumerate(items):
            p = x["r"]["pred"]
            vals = [int((p == 0).sum()), int((p == 1).sum())]
            bars = ax.bar(pos + i * w - 0.4 + w / 2, vals, w * 0.88, label=x["name"], color=PDF_COLOR[x["algo"]])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt(v), ha="center", va="bottom", size=8, color=PDF_INK)
        ax.set_xticks(pos, kat, size=9.5)
        ax.set_title("Jumlah flow menurut hasil", size=10.5, weight="bold", color=PDF_INK)
        ax.legend(frameon=False, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(PDF_LINE)
        ax.tick_params(colors=PDF_MUTED, labelsize=8.5)
        ax.grid(axis="y", color=PDF_LINE, lw=0.7)
        ax.set_axisbelow(True)
        ax.margins(y=0.16)

        ax2 = fig.add_axes([L + (R - L) * 0.62, 0.58, (R - L) * 0.38, 0.27])
        nm = [x["name"].replace(" ", "\n") for x in items]
        tv = [float(x["r"]["t_pred"]) for x in items]
        b2 = ax2.bar(nm, tv, 0.55, color=[PDF_COLOR[x["algo"]] for x in items])
        for b, v in zip(b2, tv):
            ax2.text(b.get_x() + b.get_width() / 2, v, f"{dec(v, 1)} dtk", ha="center", va="bottom",
                     size=8.5, weight="bold", color=PDF_INK)
        ax2.set_title("Lama memeriksa (detik)", size=10.5, weight="bold", color=PDF_INK)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.spines[["left", "bottom"]].set_color(PDF_LINE)
        ax2.tick_params(colors=PDF_MUTED, labelsize=8.5)
        ax2.grid(axis="y", color=PDF_LINE, lw=0.7)
        ax2.set_axisbelow(True)
        ax2.margins(y=0.2)

        y = 0.51
        y = _pdf_bullets(fig, y, [
            "Grafik kiri menunjukkan berapa banyak flow yang dinyatakan normal dan berapa yang dinyatakan serangan "
            "oleh masing-masing model pada berkas yang sama.",
            "Grafik kanan membandingkan lama pemeriksaan. Isolation Forest menelusuri sekumpulan pohon keputusan "
            "dengan biaya tetap, sedangkan Local Outlier Factor mengukur jarak ke seluruh data acuan yang "
            "disimpannya, sehingga waktunya ikut membesar bila data acuannya bertambah.",
            "Waktu ini hanya menghitung proses penilaian, di luar waktu membaca berkas.",
        ], size=9.5, width=92)
        _pdf_close(pdf, fig)

        # ---------------------------------------------------------------- 6. recall per jenis
        per_all = {}
        for x in items:
            for jenis, row in x["r"]["per"].iterrows():
                per_all.setdefault(jenis, {"n": int(row["n"]), "v": {}})["v"][x["algo"]] = float(row["Recall"])
        urut = sorted(per_all.items(), key=lambda kv: -kv[1]["n"])[:12]
        if urut:
            fig, y = _pdf_page(pdf, "Kemampuan per Jenis Serangan",
                               "Angka total bisa menutupi kelemahan pada jenis serangan tertentu.")
            ax = fig.add_axes([L + 0.14, 0.46, (R - L) - 0.16, 0.40])
            names = [k for k, _ in urut][::-1]
            yy = np.arange(len(names))
            h = 0.8 / len(items)
            for i, x in enumerate(items):
                vals = [per_all[k]["v"].get(x["algo"], 0.0) for k in names]
                bars = ax.barh(yy + i * h - 0.4 + h / 2, vals, h * 0.85, label=x["name"], color=PDF_COLOR[x["algo"]])
                for b, v in zip(bars, vals):
                    ax.text(v + 0.012, b.get_y() + b.get_height() / 2, dec(v, 2), va="center", size=7.5, color=PDF_INK)
            ax.set_yticks(yy, [f"{k}\n({fmt(per_all[k]['n'])} flow)" for k in names], size=8)
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("Bagian serangan jenis ini yang berhasil tertangkap", size=9, color=PDF_MUTED)
            ax.legend(frameon=False, ncol=len(items), loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(PDF_LINE)
            ax.tick_params(colors=PDF_MUTED, labelsize=8)
            ax.grid(axis="x", color=PDF_LINE, lw=0.7)
            ax.set_axisbelow(True)
            y = 0.40
            y = _pdf_bullets(fig, y, [
                "Nilai 1,00 berarti seluruh serangan jenis itu tertangkap; 0,00 berarti semuanya lolos.",
                f"Jenis dengan contoh kurang dari {core.MIN_TYPE_N} flow tidak dapat dijadikan kesimpulan karena "
                "jumlah datanya terlalu sedikit.",
                "Jenis dengan nilai rendah adalah celah yang perlu diperhatikan, meskipun angka keseluruhan "
                "model terlihat bagus.",
            ], size=9.5, width=92)
            _pdf_close(pdf, fig)

        # ---------------------------------------------------------------- 7. temuan & catatan
        fig, y = _pdf_page(pdf, "Temuan dan Catatan Penting",
                           "Rangkuman teknis per model, diikuti batasan yang perlu disampaikan.")
        for x in items:
            fig.text(L, y, x["name"], size=12, weight="bold", color=PDF_COLOR[x["algo"]], va="top")
            fig.text(R, y, x["r"]["key"], size=9, color=PDF_MUTED, va="top", ha="right")
            y = _pdf_bullets(fig, y - 0.024, x.get("findings", []), size=9, width=94) - 0.014
        y -= 0.006
        y = _pdf_text(fig, y, "Batasan laporan ini", size=12, weight="bold") - 0.012
        _pdf_bullets(fig, y, [
            "Hasil berasal dari satu berkas uji dan satu seed per model. Untuk menyimpulkan model mana yang lebih "
            "baik, ulangi pengujian dengan beberapa seed lalu laporkan rata-rata beserta simpangannya.",
            "Model hanya mempelajari pola lalu lintas normal. Yang ditandai adalah lalu lintas yang tidak wajar, "
            "yang belum tentu merupakan serangan.",
            "Perbandingan hanya adil bila kedua model memakai berkas uji, target alarm palsu, dan seed yang sama.",
        ], size=9, width=94)
        _pdf_close(pdf, fig)

        d = pdf.infodict()
        d["Title"] = "Laporan Perbandingan Model Deteksi DDoS"
        d["Subject"] = f"Perbandingan {' dan '.join(ALGO_NAME[a] for a in algos)} pada {D['fname']}"
        d["CreationDate"] = datetime.datetime.now()
    buf.seek(0)
    return buf


@app.route("/laporan/unduh")
def laporan_unduh():
    items = comparison()
    if not items or items[0]["r"]["metrics"] is None:
        abort(404)
    buf = laporan_pdf(items)
    return send_file(buf, as_attachment=True, download_name="laporan_perbandingan_model.pdf",
                     mimetype="application/pdf")


# ============================================================================ halaman: Pengaturan
@app.route("/pengaturan", methods=["GET", "POST"])
def pengaturan():
    """Hanya memperlihatkan aturan tiap model tersimpan. Model aktif dipilih di halaman Uji Model."""
    if request.method == "POST":
        if request.form.get("aksi") == "reset":
            hapus_file()
            flash("Berkas uji dan hasil deteksi dihapus dari memori.", "success")
        return redirect(url_for("pengaturan"))
    besar = acuan_terbesar()
    fprs = sorted(fpr_options(), reverse=True)
    rows = []
    for k in bundle_files():
        m = model_meta(k)
        if "error" in m:
            rows.append({"key": k, "error": m["error"]})
            continue
        p = m.get("params") or {}
        if m.get("algo") == "IF":
            par = f"{fmt(p.get('n_estimators', 0))} pohon · mf {dec(p.get('max_features', 0), 1)}"
        else:
            par = f"k = {fmt(p.get('n_neighbors', 0))}"
        thr = {float(f): v for f, v in (m.get("thresholds") or [])}
        rows.append({"key": k, "algo": m.get("algo"), "name": ALGO_NAME.get(m.get("algo"), m.get("algo")),
                     "seed": m.get("seed"), "n_train": m.get("n_train") or 0, "par": par,
                     "prep": PREP_LABEL.get(m.get("preprocess"), m.get("preprocess")),
                     "n_features": m.get("n_features"), "size": mbsize(model_size(k)),
                     "thr": [thr.get(f) for f in fprs], "kecil": (m.get("n_train") or 0) < besar})
    rows.sort(key=lambda r: (r.get("algo") or "zz", -(r.get("n_train") or 0), r["key"]))
    return render_template("pengaturan.html", nav="pengaturan", rows=rows, fprs=fprs, acuan_besar=besar,
                           n_kecil=sum(1 for r in rows if r.get("kecil")))


# ============================================================================ halaman: Riwayat
def _rec_view(rec):
    algos = rec.get("algos", {})
    urut = [a for a in ("IF", "LOF") if a in algos]
    f = str(rec.get("fpr_shown"))
    f1 = {}
    for a in urut:
        mm = (algos[a]["by_fpr"].get(f) or {}).get("metrics")
        f1[a] = None if not mm else mm.get("F1")
    return {"id": rec["id"], "created": rec.get("created", ""), "fname": rec.get("fname", ""),
            "model_txt": " + ".join(algos[a]["key"] for a in urut),
            "acuan": max((algos[a].get("n_train") or 0) for a in urut) if urut else 0,
            "ambang": " & ".join(pctc(x, 0) for x in sorted(rec.get("fprs", []), reverse=True)),
            "n_rows": rec.get("n_rows", 0), "f1": f1, "algos": urut, "rec": rec}


@app.route("/riwayat")
def riwayat():
    views = [_rec_view(r) for r in riwayat_list()]
    besar = acuan_terbesar()
    for v in views:
        v["kecil"] = 0 < v["acuan"] < besar
    pilih = [p for p in request.args.getlist("sel") if any(v["id"] == p for v in views)]
    for v in views:
        v["dipilih"] = v["id"] in pilih
    banding = None
    if len(pilih) >= 2:
        sub = [v for v in views if v["id"] in pilih]
        baris = []
        for v in sub:
            for a in v["algos"]:
                info = v["rec"]["algos"][a]
                f = str(v["rec"].get("fpr_shown"))
                mm = (info["by_fpr"].get(f) or {}).get("metrics") or {}
                baris.append({"tanggal": v["created"], "key": info["key"], "name": info["name"],
                              "seed": info.get("seed"), "n_train": info.get("n_train") or 0,
                              "par": info.get("params"), "fpr": v["rec"].get("fpr_shown"),
                              "recall": mm.get("Recall"), "f1": mm.get("F1"), "fpr_akt": mm.get("FPR"),
                              "t": info.get("t_pred")})
        banding = {"baris": baris, "n": len(sub)}
    return render_template("riwayat.html", nav="riwayat", views=views, pilih=pilih, banding=banding,
                           total=len(views), ukuran=mbsize(riwayat_ukuran()),
                           n_berkas=len({v["fname"] for v in views}))


@app.route("/riwayat/lihat/<rid>")
def riwayat_lihat(rid):
    try:
        rec = riwayat_muat(rid)
    except UserError as e:
        flash(str(e), "danger")
        return redirect(url_for("riwayat"))
    flash(f"Riwayat {rec['created']} dimuat: {rec['fname']}. Berkas CSV-nya tidak ikut tersimpan, "
          f"jadi pengujian ulang membutuhkan unggahan baru.", "success")
    return redirect(url_for("uji"))


@app.route("/riwayat/hapus/<rid>", methods=["POST"])
def riwayat_hapus_satu(rid):
    riwayat_hapus(rid)
    flash("Satu riwayat dihapus.", "success")
    return redirect(url_for("riwayat"))


@app.route("/riwayat/hapus-semua", methods=["POST"])
def riwayat_hapus_semua():
    n = 0
    for rec in riwayat_list():
        riwayat_hapus(rec["id"])
        n += 1
    flash(f"{fmt(n)} riwayat dihapus.", "success")
    return redirect(url_for("riwayat"))


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
