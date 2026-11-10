"""
app.py — Sistem Deteksi Anomali Serangan DDoS (Isolation Forest & LOF), Flask + Python.
Jalankan:  python app.py   lalu buka http://127.0.0.1:5000

Halaman utama (sesuai desain):
  Dashboard           : unggah dataset uji, pilih model, Mulai Uji Model, ringkasan, tabel, confusion matrix, distribusi
  Uji Model           : form yang sama dengan opsi (FPR, batas baris, jalankan kedua model)
  Hasil Deteksi       : seluruh hasil per flow (filter, halaman, unduh CSV)
  Perbandingan Model  : IF vs LOF pada file yang sama (metrik, confusion matrix, recall per jenis, laporan)
  Pengaturan          : model aktif, target FPR, model tersimpan, pengaman data latih, pelatihan model (opsional)

Pengujian TIDAK melatih apa pun: sistem hanya memuat model tersimpan di folder models/ lalu mendeteksi file yang diunggah.
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
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

APP_TITLE = ("Penerapan Algoritma Isolation Forest untuk Deteksi Anomali "
             "Trafik Jaringan Serangan Distributed Denial of Service (DDoS)")
ALGO_NAME = {"IF": "Isolation Forest", "LOF": "Local Outlier Factor"}
ALGO_DESC = {"IF": "Deteksi anomali berbasis<br>pohon keputusan", "LOF": "Deteksi anomali berbasis<br>kepadatan data"}
OVERLAP_BLOCK = 0.5             # file deteksi ditolak jika >= 50% barisnya identik dengan data latih
CHUNK = 20_000                  # skor dihitung per potongan agar kemajuan bisa ditampilkan
PER_PAGE_DASH, PER_PAGE_HASIL = 10, 15

app = Flask(__name__)
app.secret_key = "ganti-dengan-kunci-acak"
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024          # unggah maks. 1 GB

LOCK = threading.RLock()
# T = bagian PELATIHAN (dataset latih, opsional). D = bagian DETEKSI (file uji). Keduanya tidak saling memakai data.
T = {"df": None, "fname": None, "label": None, "prep": None, "split": None, "trained": {}}
D = {"df": None, "fname": None, "fsize": 0, "runs": {}, "active": None, "t_load": 0.0, "max_rows": None, "fpr": None,
     "pending": None, "pending_file": None}
SET = {"fpr": 0.05, "active": {}}
JOB = {"state": "idle", "progress": 0.0, "stage": "", "started": None, "message": None, "category": "success",
       "control": "run", "cancelable": False, "paused_at": None, "paused_total": 0.0}

NAV = [("dashboard", "Dashboard", "home"), ("uji", "Uji Model", "clipboard"), ("hasil", "Hasil Deteksi", "detect"),
       ("perbandingan", "Perbandingan Model", "compare"), ("pengaturan", "Pengaturan", "gear")]


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
    return {"NAV": NAV, "ALGO_NAME": ALGO_NAME, "JOB": JOB, "job_msg": msg, "TRAIN": T}


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


def algo_cards(selected):
    cards = []
    for a in ("IF", "LOF"):
        cards.append({"algo": a, "name": ALGO_NAME[a], "desc": ALGO_DESC[a], "ok": bool(active_key(a)), "key": active_key(a)})
    return cards


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


def model_spec(algo, key=None):
    """Profil model untuk ditampilkan (Dashboard & Perbandingan). Memakai metadata, tidak memuat objek model."""
    key = key or active_key(algo)
    spec = {"algo": algo, "name": ALGO_NAME[algo], "desc": ALGO_DESC[algo], "ok": False, "key": key}
    if not key:
        return spec
    m = model_meta(key)
    if "error" in m:
        spec["error"] = m["error"]
        return spec
    spec.update(ok=True, seed=m.get("seed"), n_features=m.get("n_features"), n_train=m.get("n_train"),
                trained_on=m.get("trained_on"), created=m.get("created"), sklearn=m.get("sklearn_version"),
                prep=PREP_LABEL.get(m.get("preprocess"), m.get("preprocess")),
                params=[(PARAM_LABEL.get(k, k), param_text(k, v, m.get("n_features")))
                        for k, v in sorted((m.get("params") or {}).items())],
                thresholds=sorted(m.get("thresholds") or [], key=lambda t: t[0]))
    return spec


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
    thr = b["thresholds"][fpr]
    pred = (score > thr).astype(int)
    r = {"algo": b["model_name"], "key": key, "fname": name, "fpr": fpr, "thr": thr, "score": score, "pred": pred,
         "rows": idx + 1, "t_pred": t_pred, "labels": None if labels is None else labels.values, "metrics": None,
         "cm": None, "per": None, "n_file": n_file, "sampled": sampled, "stats": stats, "overlap": overlap,
         "guard_fp": bool(fps), "id": time.time()}
    if labels is not None:
        r["metrics"], r["cm"], r["per"] = core.evaluate(labels.values, pred, score)
    return r, None


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
                raise UserError("Unggah file CSV uji terlebih dahulu.")
            if D["fpr"] != fpr or D["max_rows"] != max_rows:
                D.update(runs={}, active=None, max_rows=max_rows, fpr=fpr)
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
    return f"Deteksi selesai: {fmt(n)} flow diperiksa dengan {names}."


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


def bar_svg(cats, series):
    """Diagram batang berkelompok sebagai SVG (tanpa pustaka luar). series = [(label, warna, [nilai per kategori])]."""
    W, H, ml, mr, mt, mb = 680, 430, 90, 30, 78, 68
    pw, ph = W - ml - mr, H - mt - mb
    vmax = max(max(v) for _, _, v in series) or 1
    step = nice_step(vmax, 6)
    nst = max(1, math.ceil(vmax / step))
    top = step * nst
    grand = sum(sum(v) for _, _, v in series) or 1
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Distribusi prediksi">']
    for i in range(nst + 1):
        y = mt + ph - ph * i / nst
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W - mr}" y2="{y:.1f}" stroke="#e6edf6" stroke-width="1"/>')
        out.append(f'<text x="{ml - 12}" y="{y + 4.5:.1f}" text-anchor="end" font-size="12" fill="#52617f">{fmt(i * step)}</text>')
    gw = pw / len(cats)
    bw = gw * (0.42 if len(series) == 1 else 0.3)
    gap = gw * 0.05
    for ci, cat in enumerate(cats):
        total_w = len(series) * bw + (len(series) - 1) * gap
        x0 = ml + gw * ci + (gw - total_w) / 2
        for si, (lab, col, vals) in enumerate(series):
            v = vals[ci]
            h = ph * v / top if top else 0
            x = x0 + si * (bw + gap)
            y = mt + ph - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h, 0):.1f}" rx="5" fill="{col}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{y - 24:.1f}" text-anchor="middle" font-size="15" font-weight="700" fill="#17315c">{fmt(v)}</text>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-size="11.5" fill="#6b7893">{v / grand * 100:.1f}%</text>')
        out.append(f'<text x="{ml + gw * ci + gw / 2:.1f}" y="{mt + ph + 26}" text-anchor="middle" font-size="13.5" font-weight="600" fill="#2a3a57">{cat}</text>')
    out.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{W - mr}" y2="{mt + ph}" stroke="#b9c6da" stroke-width="1"/>')
    out.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 10}" text-anchor="middle" font-size="12.5" font-weight="600" fill="#52617f">Jenis Trafik</text>')
    out.append(f'<text transform="translate(16 {mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="12.5" font-weight="600" fill="#52617f">Jumlah Data</text>')
    lx = ml  # legenda di pita atas kiri agar tidak bertabrakan dengan angka di atas batang
    for lab, col, _ in series:
        out.append(f'<rect x="{lx:.1f}" y="13" width="13" height="13" rx="3" fill="{col}"/>')
        out.append(f'<text x="{lx + 19:.1f}" y="24" font-size="12.5" fill="#2a3a57">{lab}</text>')
        lx += 19 + len(lab) * 7.2 + 26
    out.append("</svg>")
    return "".join(out)


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
@app.route("/")
def dashboard():
    """Gambaran umum: profil model aktif, asal data latihnya, dan ringkasan deteksi terakhir.
    Halaman ini tidak mengubah apa pun — pengujian dijalankan di halaman Uji Model."""
    runs = ordered_runs()
    act = D["active"] if D["active"] in D["runs"] else (runs[0][0] if runs else None)
    r = D["runs"].get(act) if act else None
    # Rincian tiap model tidak ditampilkan di sini, hanya jumlah yang siap pakai (lihat Perbandingan Model).
    n_ready = sum(1 for a in ("IF", "LOF") if model_spec(a)["ok"])
    ctx = {"nav": "dashboard", "n_models": len(bundle_files()), "fpr_now": current_fpr(),
           "n_ready": n_ready, "fname": D["fname"], "r": r, "run_algos": [a for a, _ in runs]}
    if r is not None:
        n = len(r["pred"])
        n_att = int(r["pred"].sum())
        ctx["st"] = {"total": n, "n_att": n_att, "n_norm": n - n_att, "p_att": n_att / n, "p_norm": (n - n_att) / n,
                     "time": r.get("t_total", r["t_pred"]), "model_short": act}
        ctx["has_labels"] = r["labels"] is not None
        if r["metrics"] is not None:
            ctx["cm"] = cm_view(r["cm"], ALGO_NAME[act])
            ctx["metrics"] = [(ic_, nm, {a: x["metrics"][k] for a, x in runs if x["metrics"] is not None})
                              for ic_, nm, k in (("target", "Accuracy", "Accuracy"), ("precision", "Precision", "Precision"),
                                                 ("recall", "Recall", "Recall"), ("star", "F1-Score", "F1"))]
        colors = {"IF": "#2670d4", "LOF": "#37ac77"}
        ctx["chart_svg"] = bar_svg(["Normal", "DDoS"], [(ALGO_NAME[a] if a == "IF" else "LOF", colors[a],
                                                         [int((x["pred"] == 0).sum()), int((x["pred"] == 1).sum())])
                                                        for a, x in runs])
    return render_template("dashboard.html", **ctx)


# ============================================================================ halaman: Uji Model + aksi Mulai Uji
@app.route("/uji", methods=["GET", "POST"])
def uji():
    if request.method == "POST":
        return uji_post()
    rows = [model_meta(active_key(a)) | {"name": ALGO_NAME[a]} for a in ("IF", "LOF") if active_key(a)]
    feats = rows[0]["features"] if rows else []
    runs = ordered_runs()
    act = D["active"] if D["active"] in D["runs"] else None
    cards = algo_cards(act)
    # Berkas yang sedang dimuat: dicek terhadap syarat kolom supaya pengguna tahu sebelum menekan Mulai.
    finfo = None
    if D["df"] is not None:
        lab = core.find_label(D["df"])
        miss = [f for f in feats if f not in D["df"].columns]
        finfo = {"rows": len(D["df"]), "cols": D["df"].shape[1], "label": lab, "n_missing": len(miss),
                 "missing": miss[:6], "fname": D["fname"], "fsize": mbsize(D["fsize"])}
    last = [{"algo": a, "name": ALGO_NAME[a], "key": x["key"], "n": len(x["pred"]),
             "n_att": int(x["pred"].sum()), "fpr": x["fpr"], "thr": x["thr"],
             "t": x.get("t_total", x["t_pred"]), "sampled": x["sampled"], "n_file": x["n_file"],
             "acc": (x["metrics"]["Accuracy"] if x["metrics"] is not None else None)} for a, x in runs]
    return render_template("uji.html", nav="uji", algos=cards,
                           selected=act or D["pending"] or next((c["algo"] for c in cards if c["ok"]), "IF"), **file_chip(),
                           model_rows=rows, features=feats, n_features=len(feats), fpr_options=fpr_options(),
                           fpr_now=D["fpr"] or current_fpr(), max_rows=D["max_rows"], next="uji",
                           finfo=finfo, last=last)


def uji_post():
    target = "dashboard" if request.form.get("next") == "dashboard" else "uji"
    if JOB["state"] == "running":
        flash("Masih ada proses yang berjalan. Tunggu sampai selesai.", "warning")
        return redirect(url_for(target))
    algo = request.form.get("algo", "IF")
    algos = ["IF", "LOF"] if request.form.get("both") == "1" else [algo if algo in ALGO_NAME else "IF"]
    fpr = D["fpr"] if (D["df"] is not None and "fpr" not in request.form) else None
    try:
        fpr = float(request.form["fpr"]) if "fpr" in request.form else (fpr or current_fpr())
    except ValueError:
        fpr = current_fpr()
    if "max_rows" in request.form:
        mr = request.form.get("max_rows", "").strip()
        max_rows = int(mr) if mr.isdigit() and int(mr) > 0 else None
    else:
        max_rows = D["max_rows"] if D["df"] is not None else None

    f = request.files.get("file")
    path = name = None
    size = 0
    if f and f.filename:
        if not f.filename.lower().endswith(".csv"):
            flash("Pilih file .csv.", "danger")
            return redirect(url_for(target))
        name = f.filename
        path = os.path.join(UPLOAD_DIR, os.path.basename(name))
        f.save(path)
        size = os.path.getsize(path)
        if "train" in name.lower():
            flash(f"Nama file '{name}' mengandung 'train'. Pastikan ini bukan data latih.", "warning")
    elif D["df"] is None:
        flash("Unggah file CSV uji terlebih dahulu.", "danger")
        return redirect(url_for(target))
    for a in algos:
        if not active_key(a):
            flash(f"Model {ALGO_NAME[a]} belum tersedia. Salin file .joblib ke folder models/ atau latih model di Pengaturan.", "danger")
            return redirect(url_for(target))
    D["pending"] = algos[-1]
    D["pending_file"] = (name, size) if path else None
    run_job(job_detect, algos, path, name, size, fpr, max_rows, cancelable=True)
    return redirect(url_for(target))


@app.route("/hapus-file", methods=["POST"])
def hapus_file():
    with LOCK:
        D.update(df=None, fname=None, fsize=0, runs={}, active=None, t_load=0.0, max_rows=None, fpr=None,
                 pending=None, pending_file=None)
    return "ok"


# ============================================================================ halaman: Hasil Deteksi
@app.route("/hasil")
def hasil():
    runs = ordered_runs()
    if not runs:
        return render_template("hasil.html", nav="hasil", r=None)
    algos = [a for a, _ in runs]
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
    sel = idx[(page - 1) * PER_PAGE_HASIL: page * PER_PAGE_HASIL]
    # Ringkasan dipakai sebagai kartu yang sekaligus menjadi pintasan filter.
    n = len(pred)
    summ = {"all": n, "ddos": int(pred.sum()), "normal": int((pred == 0).sum())}
    if has_labels:
        summ.update(salah=int((actual != pred).sum()),
                    fp=int(((actual == 0) & (pred == 1)).sum()),
                    fn=int(((actual == 1) & (pred == 0)).sum()))
    # Skor anomali tidak bersatuan dan skalanya beda tiap algoritma; yang memberi arti adalah ambangnya.
    thr_by = {a: D["runs"][a]["thr"] for a in algos}
    meta = {"fname": D["fname"], "key": base["key"], "name": ALGO_NAME[m], "fpr": base["fpr"], "thr": base["thr"],
            "t": base.get("t_total", base["t_pred"]), "n_file": base["n_file"], "sampled": base["sampled"],
            "n_features": base["stats"]["n_features"]}
    return render_template("hasil.html", nav="hasil", r=base, run_algos=algos, m=m, flt=flt, has_labels=has_labels,
                           fname=D["fname"], rows=table_rows(sel, runs, labels), total=total, page=page, pages=pages,
                           page_window=window(page, pages), start=(page - 1) * PER_PAGE_HASIL + 1 if total else 0,
                           end=min(total, page * PER_PAGE_HASIL), summ=summ, meta=meta, thr_by=thr_by)


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


@app.route("/perbandingan")
def perbandingan():
    items = comparison()
    if not items:
        return render_template("perbandingan.html", nav="perbandingan", runs=[])
    has_labels = items[0]["r"]["metrics"] is not None
    rows = []

    def add(label, fn):
        rows.append({"label": label, "vals": {x["algo"]: fn(x["r"]) for x in items}})
    add("Flow diperiksa", lambda r: f"{fmt(len(r['pred']))} flow")
    add("Prediksi DDoS", lambda r: f"{fmt(int(r['pred'].sum()))} flow ({pctc(r['pred'].mean())})")
    if has_labels:
        for lab, k in (("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1-Score", "F1")):
            add(lab, lambda r, k=k: ratio_txt(r["metrics"][k]))
        # AUC adalah luas di bawah kurva: tanpa satuan. Persennya diberi nama artinya, bukan "benar sekian persen".
        for lab, k, how in (("ROC-AUC", "ROC_AUC", "peluang"), ("PR-AUC", "PR_AUC", "rata-rata precision")):
            add(lab, lambda r, k=k, how=how: auc_txt(r["metrics"][k], how))
        add("FPR aktual", lambda r: pctc(r["metrics"]["FPR"]))
        add(f"Macro recall (jenis ≥ {core.MIN_TYPE_N} flow)", lambda r: ratio_txt(r["metrics"]["Macro_Recall_Mayor"]))
    add("Waktu prediksi", lambda r: f"{r['t_pred']:.1f} detik".replace(".", ","))
    add("Target FPR / threshold", lambda r: f"{pctc(r['fpr'], 0)} / {dec(r['thr'], 4)} skor")
    types = []
    if has_labels:
        allt = {}
        for x in items:
            for jenis, row in x["r"]["per"].iterrows():
                allt.setdefault(jenis, {"n": int(row["n"]), "vals": {}})["vals"][x["algo"]] = float(row["Recall"])
        for jenis, v in sorted(allt.items(), key=lambda kv: -kv[1]["n"]):
            types.append({"jenis": jenis, "n": v["n"], "vals": {x["algo"]: v["vals"].get(x["algo"]) for x in items},
                          "cukup": v["n"] >= core.MIN_TYPE_N})
    fprs = {x["r"]["fpr"] for x in items}
    seeds = {model_meta(x["r"]["key"]).get("seed") for x in items}
    seed_mismatch = ", ".join(f"{x['name']}: {model_meta(x['r']['key']).get('seed')}" for x in items) if len(seeds) > 1 else None
    # Rincian model yang BENAR-BENAR dipakai pada run ini (bukan sekadar model aktif saat ini).
    specs = [model_spec(x["algo"], x["r"]["key"]) for x in items]
    bars, speed = [], []
    if has_labels:
        for lab, k in (("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1-Score", "F1")):
            vals = [{"algo": x["algo"], "name": x["name"], "v": float(x["r"]["metrics"][k])} for x in items]
            best = max((v["v"] for v in vals), default=0)
            for v in vals:
                v["best"] = len(vals) > 1 and v["v"] >= best > 0
            bars.append({"label": lab, "models": vals})   # bukan "items": bentrok dengan dict.items di Jinja
    tmax = max((x["r"]["t_pred"] for x in items), default=1) or 1
    for x in items:
        speed.append({"algo": x["algo"], "name": x["name"], "t": x["r"]["t_pred"],
                      "frac": x["r"]["t_pred"] / tmax,
                      "fastest": len(items) > 1 and x["r"]["t_pred"] <= min(y["r"]["t_pred"] for y in items)})
    return render_template("perbandingan.html", nav="perbandingan", runs=items, metric_rows=rows, types=types,
                           has_labels=has_labels, fname=D["fname"], min_n=core.MIN_TYPE_N, fpr_mismatch=len(fprs) > 1,
                           seed_mismatch=seed_mismatch, specs=specs, bars=bars, speed=speed)


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
    if request.method == "POST":
        act = request.form.get("aksi")
        if act == "simpan":
            for a in ("IF", "LOF"):
                v = request.form.get(f"active_{a}")
                if v:
                    SET["active"][a] = v
            try:
                SET["fpr"] = float(request.form.get("fpr", SET["fpr"]))
            except ValueError:
                pass
            with LOCK:
                D.update(runs={}, active=None, fpr=None)
            flash("Pengaturan disimpan. Hasil deteksi sebelumnya dihapus.", "success")
        elif act == "reset":
            hapus_file()
            flash("File uji dan hasil deteksi dihapus dari memori.", "success")
        return redirect(url_for("pengaturan"))
    algos = [{"algo": a, "name": ALGO_NAME[a], "options": algo_keys().get(a, []), "active": active_key(a)} for a in ("IF", "LOF")]
    return render_template("pengaturan.html", nav="pengaturan", algos=algos, fpr_options=fpr_options(), fpr_now=current_fpr(),
                           rows=bundle_rows())


# ============================================================================ detail model
def tree_png(b, idx, depth):
    m = b["model"]
    idx = max(0, min(idx, len(m.estimators_) - 1))
    names = [b["features"][i] for i in m.estimators_features_[idx]]
    fig, ax = plt.subplots(figsize=(15, 5.5))
    plot_tree(m.estimators_[idx], max_depth=depth, feature_names=names, impurity=False, filled=True, fontsize=7, ax=ax)
    ax.set_title(f"Isolation Tree #{idx} (ditampilkan sampai kedalaman {depth})")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/model")
def model():
    ok = [k for k in bundle_files() if "error" not in model_meta(k)]
    key = request.args.get("m") if request.args.get("m") in ok else (ok[0] if ok else None)
    b = get_bundle(key) if key else None
    tree_idx = int(request.args.get("tree", 0))
    depth = int(request.args.get("depth", 3))
    img = tree_png(b, tree_idx, depth) if b is not None and b["model_name"] == "IF" else None
    version_warn = b is not None and b.get("sklearn_version") not in (None, sklearn.__version__)
    params_fmt = ([(PARAM_LABEL.get(k, k), param_text(k, v, len(b["features"])))
                   for k, v in sorted((b.get("params") or {}).items())] if b else [])
    return render_template("model.html", nav="pengaturan", key=key, b=b, thr=sorted(b["thresholds"].items()) if b else [],
                           img=img, tree_idx=tree_idx, depth=depth, sk_ver=sklearn.__version__,
                           version_warn=version_warn, params_fmt=params_fmt)


@app.route("/model/unduh")
def model_unduh():
    key = request.args.get("m")
    b = get_bundle(key) if key else None
    if b is None:
        abort(404)
    buf = io.BytesIO()
    joblib.dump(b, buf, compress=3)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"{key}.joblib")


# ============================================================================ pelatihan (opsional, dari Pengaturan)
def reset_train(stage):
    if stage == "data":
        T["prep"] = T["split"] = None
        T["trained"] = {}
    elif stage == "prep":
        T["split"] = None
        T["trained"] = {}
    elif stage == "split":
        T["trained"] = {}


MSG = {"tdata": ("Muat dataset latih terlebih dahulu.", "latih_dataset"),
       "tlabel": ("Dataset latih membutuhkan kolom label (untuk memilih trafik normal sebagai data acuan).", "latih_dataset"),
       "tprep": ("Jalankan Preprocessing terlebih dahulu.", "latih_preprocessing"),
       "tsplit": ("Tetapkan Data Acuan terlebih dahulu.", "latih_acuan")}


def train_missing(*items):
    for it in items:
        ok = {"tdata": T["df"] is not None, "tlabel": T["label"] is not None,
              "tprep": T["prep"] is not None, "tsplit": T["split"] is not None}[it]
        if not ok:
            return MSG[it]
    return None


def blocked(title, step):
    return render_template("blocked.html", nav="pengaturan", title=title, msg=step[0], link=step[1])


def load_train(path, name):
    df = core.read_csv(path)
    with LOCK:
        T.update(df=df, fname=name, label=core.find_label(df))
        reset_train("data")
    flash(f"Dataset latih dimuat: {name} ({fmt(len(df))} baris).", "success")


@app.route("/latih/dataset", methods=["GET", "POST"])
def latih_dataset():
    if request.method == "POST":
        act = request.form.get("aksi")
        try:
            if act == "unggah":
                f = request.files.get("file")
                if not f or not f.filename.lower().endswith(".csv"):
                    flash("Pilih file .csv.", "danger")
                    return redirect(url_for("latih_dataset"))
                path = os.path.join(UPLOAD_DIR, os.path.basename(f.filename))
                f.save(path)
                load_train(path, f.filename)
            elif act == "label":
                v = request.form.get("label")
                T["label"] = None if v == "__none__" else v
                reset_train("data")
                flash("Kolom label diperbarui.", "success")
        except Exception as e:                                        # noqa: BLE001
            flash(f"Gagal membaca dataset: {e}", "danger")
        return redirect(url_for("latih_dataset"))
    if T["df"] is None:
        return render_template("latih_dataset.html", nav="pengaturan", info=None)
    df = T["df"]
    info = core.dataset_info(df, T["label"])
    dist = None
    if T["label"]:
        d = info["dist"].rename_axis("Label").reset_index(name="Jumlah")
        d["Porsi"] = (d["Jumlah"] / d["Jumlah"].sum() * 100).round(2)
        dist = d.to_dict("records")
    preview = df.head(50)
    return render_template("latih_dataset.html", nav="pengaturan", info=info, dist=dist, cols=list(df.columns),
                           preview_cols=list(preview.columns), preview=preview.astype(str).values.tolist())


@app.route("/latih/preprocessing", methods=["GET", "POST"])
def latih_preprocessing():
    step = train_missing("tdata", "tlabel")
    if step:
        return blocked("Preprocessing Data Latih", step)
    features = core.candidate_features(T["df"], T["label"])
    if request.method == "POST" and request.form.get("aksi") == "daftar":
        if T["prep"] is None:
            flash("Jalankan Preprocessing terlebih dahulu.", "warning")
            return redirect(url_for("latih_preprocessing"))
        name = "latih_" + os.path.splitext(os.path.basename(T["fname"]))[0] + ".fp.npz"
        core.save_fingerprint(os.path.join(MODEL_DIR, name), T["prep"]["X"][T["prep"]["features"]].values,
                              T["prep"]["features"], T["fname"])
        flash(f"Dataset {T['fname']} didaftarkan sebagai data latih. Pengujian akan menolak file yang berisi data ini. "
              f"Tidak ada model yang dilatih.", "success")
        return redirect(url_for("latih_preprocessing"))
    if request.method == "POST":
        drop_dup = request.form.get("drop_dup") == "1"
        X, labels, log, stats = core.preprocess(T["df"], T["label"], features, drop_dup)
        with LOCK:
            T["prep"] = {"X": X, "labels": labels, "log": log, "stats": stats, "features": features}
            reset_train("prep")
        flash("Preprocessing selesai.", "success")
        return redirect(url_for("latih_preprocessing"))
    src = f"fitur numerik tidak konstan, tanpa label/identitas/port ({len(features)} fitur)"
    return render_template("latih_preprocessing.html", nav="pengaturan", src=src, prep=T["prep"])


@app.route("/latih/acuan", methods=["GET", "POST"])
def latih_acuan():
    step = train_missing("tdata", "tlabel", "tprep")
    if step:
        return blocked("Data Acuan Pelatihan", step)
    if request.method == "POST":
        seed = int(request.form.get("seed", 42))
        sp = core.data_acuan(T["prep"]["labels"])                     # SEMUA trafik normal, tanpa pembagian dan tanpa batas
        with LOCK:
            T["split"] = {**sp, "seed": seed, "id": time.time()}
            reset_train("split")
        flash("Data acuan ditetapkan.", "success")
        return redirect(url_for("latih_acuan"))
    return render_template("latih_acuan.html", nav="pengaturan", sp=T["split"])


def job_train(algo, params, fprs, key):
    sp, feats, X = T["split"], T["prep"]["features"], T["prep"]["X"]
    set_stage(f"Melatih {ALGO_NAME[algo]} dari {fmt(len(sp['fit']))} baris data normal...", 0.15)
    b = core.train_bundle(algo, params, X.iloc[sp["fit"]][feats].values, feats, fprs, sp["seed"], trained_on=T["fname"])
    set_stage("Menyimpan model dan sidik jari data latih...", 0.9)
    fp_name = "latih_" + os.path.splitext(os.path.basename(T["fname"]))[0] + ".fp.npz"
    core.save_fingerprint(os.path.join(MODEL_DIR, fp_name), X[feats].values, feats, T["fname"])
    b["train_fp"] = fp_name
    core.save_bundle(b, MODEL_DIR, key)
    META.pop(key, None)
    with LOCK:
        T["trained"][algo] = {"key": key, "b": b}
    return f"Model {key} dilatih ({b['fit_time_s']:.1f} detik) dan disimpan ke folder models/."


@app.route("/latih/model", methods=["GET", "POST"])
def latih_model():
    step = train_missing("tdata", "tlabel", "tprep", "tsplit")
    if step:
        return blocked("Pelatihan Model", step)
    algo = request.values.get("algo", "IF")
    algo = algo if algo in ALGO_NAME else "IF"
    if request.method == "POST":
        if JOB["state"] == "running":
            flash("Masih ada proses yang berjalan. Tunggu sampai selesai.", "warning")
            return redirect(url_for("latih_model", algo=algo))
        if algo == "IF":
            ms = request.form.get("max_samples", "2048")
            params = {"n_estimators": int(request.form.get("n_estimators", 300)),
                      "max_samples": ms if ms == "auto" else int(ms),
                      "max_features": float(request.form.get("max_features", 0.5)),
                      "log_input": request.form.get("log_input") == "1"}
        else:
            params = {"n_neighbors": int(request.form.get("n_neighbors", 50))}
        fprs = [float(f) for f in request.form.getlist("fpr")] or [0.05]
        key = f"{algo}_seed{T['split']['seed']}"
        if os.path.exists(os.path.join(MODEL_DIR, f"{key}.joblib")) and request.form.get("timpa") != "1":
            flash(f"Model {key}.joblib sudah ada. Centang 'Timpa model lama' atau ganti seed di Data Acuan.", "danger")
            return redirect(url_for("latih_model", algo=algo))
        run_job(job_train, algo, params, fprs, key)
        return redirect(url_for("latih_model", algo=algo))
    return render_template("latih_model.html", nav="pengaturan", algo=algo, sp=T["split"], res=T["trained"].get(algo),
                           n_models=len(bundle_files()))


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
