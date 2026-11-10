# Sistem Deteksi Anomali Serangan DDoS (Isolation Forest & LOF) — Flask

Aplikasi web lokal (Flask + Python, tanpa Streamlit) untuk **menguji model yang sudah dibangun** pada file uji baru.
Tampilan tidak memakai CDN: CSS dan ikon sudah ada di dalam proyek, jadi berjalan juga tanpa internet.

## Halaman
| Menu | Isi |
|---|---|
| **Dashboard** | Unggah dataset uji, pilih model (Isolation Forest / LOF), **Mulai Uji Model**, ringkasan (total, normal, serangan, waktu proses), tabel hasil, confusion matrix + metrik, distribusi prediksi |
| **Uji Model** | Formulir yang sama dengan opsi: target alarm palsu (FPR), batas jumlah baris, jalankan kedua model sekaligus |
| **Hasil Deteksi** | Seluruh hasil per flow: filter (DDoS / Normal / salah deteksi), halaman, skor anomali, unduh CSV |
| **Perbandingan Model** | IF vs LOF pada file yang sama: metrik lengkap, confusion matrix, recall per jenis serangan, temuan, unduh laporan (.md) |
| **Pengaturan** | Model aktif (seed), target FPR, daftar model tersimpan, pelatihan model (opsional), hapus sesi |

## Cara menjalankan (Windows, VS Code)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Buka http://127.0.0.1:5000. Gunakan Python 3.10–3.12.

## Cara menguji
1. Salin file model `.joblib` ke folder `models/` (mis. `IF_seed42.joblib`, `LOF_seed42.joblib`).
2. Di **Dashboard**, pilih file CSV uji (klik atau seret), pilih model, lalu klik **Mulai Uji Model**.
   Proses berjalan di latar belakang: bilah kemajuan muncul di bagian atas halaman.
3. Untuk membandingkan, pilih model lainnya dan klik **Mulai Uji Model** lagi (file yang sama tidak perlu diunggah ulang),
   atau centang "jalankan kedua model sekaligus" di **Uji Model**. Hasilnya muncul di **Perbandingan Model**.

**Pengujian tidak melatih apa pun.** Sistem hanya memuat model tersimpan lalu mendeteksi seluruh isi file yang diunggah.
Pembersihan nilai kosong/tak hingga dan transformasi fitur dijalankan otomatis sesuai model (aturan yang sama saat pelatihan).

## Pengaman: pengujian tidak boleh memakai data latih
Pengujian menolak file yang namanya sama dengan dataset latih yang tercatat di model (`trained_on`), dan file yang ≥ 50% barisnya identik
dengan dataset latih yang sidik jarinya tersimpan (otomatis dibuat saat melatih model di sistem ini).
Model dari notebook Kaggle versi perbaikan sudah membawa `trained_on`, sehingga file `train_sandbox_v3.csv` ditolak berdasarkan namanya.

## Pelatihan model (opsional, di Pengaturan)
**Pengaturan → Buka halaman pelatihan**: Dataset Latih → Preprocessing → Data Acuan → Pelatihan Model.
Model dilatih dari **seluruh trafik normal** dataset latih (tanpa pembagian data dan tanpa batas jumlah); baris serangan tidak dipakai melatih.
Threshold ditentukan dari skor data latih itu sendiri. LOF pada data penuh membutuhkan waktu sekitar 10 menit dan file model ratusan MB; IF cepat.
Model juga bisa dibuat di notebook (`02_pelatihan_model.ipynb`) lalu disalin ke folder `models/`.

## Struktur folder
```
sistem_deteksi_ddos/
├── app.py            # backend Flask: halaman, proses latar belakang, grafik SVG
├── core.py           # logika: preprocessing, pelatihan, deteksi, evaluasi, pengaman data latih
├── requirements.txt
├── static/style.css  # desain (tanpa CDN)
├── templates/        # halaman HTML (+ ikon SVG inline)
├── models/           # file model .joblib dan sidik jari data latih (*.fp.npz)
└── uploads/          # salinan file yang diunggah (boleh dikosongkan)
```

## Batasan
- File uji harus memuat kolom fitur yang sama dengan data latih model (73 kolom, gaya CICFlowMeter). Kolom yang hilang disebutkan di pesan galat.
- Hasil adalah **anomali trafik**, bukan vonis pasti DDoS, dan tidak menyebut jenis serangan.
- scikit-learn dikunci 1.6.1 (sesuai model dari Kaggle). Model yang dibuat dengan versi lain bisa gagal dimuat.
- Aplikasi untuk satu pengguna lokal; data disimpan di memori dan hilang saat aplikasi dimatikan.
- Waktu proses LOF pada file besar lama (±9 menit untuk 352 ribu baris dengan model dari data penuh).
