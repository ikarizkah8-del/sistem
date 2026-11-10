# Sistem Deteksi Anomali Trafik DDoS — Isolation Forest & Local Outlier Factor

Aplikasi web lokal berbasis **Flask** untuk menguji dan membandingkan dua algoritma deteksi anomali
*unsupervised* pada berkas trafik jaringan: **Isolation Forest (IF)** dan **Local Outlier Factor (LOF)**.

Sistem ini **hanya menguji** model yang sudah dilatih — tidak ada proses pelatihan di dalamnya.
Model tersimpan dimuat dari folder `models/`, lalu berkas CSV yang diunggah dinilai terhadap ambang
batas yang sudah tersimpan di dalam model itu sendiri.

Dikerjakan untuk skripsi *Evaluasi Performa Algoritma Isolation Forest dan Local Outlier Factor untuk
Deteksi Anomali Trafik Jaringan Serangan Distributed Denial of Service (DDoS)* — Teknik Informatika,
UIN Alauddin Makassar.

---

## Gagasan utama sistem

Satu kali pengujian menghitung skor anomali **sekali saja**, lalu membandingkannya dengan **seluruh
ambang batas** yang dimiliki model (target FPR 5% dan 1%). Karena itu satu kali tekan tombol langsung
menghasilkan empat kolom hasil — IF\@5%, LOF\@5%, IF\@1%, LOF\@1% — tanpa biaya hitung tambahan.

Itu penting karena temuan penelitiannya memang bergantung pada ambang: pada FPR 5% LOF lebih teliti,
tetapi pada FPR 1% recall LOF runtuh sementara Isolation Forest bertahan.

---

## Kebutuhan

| Kebutuhan | Versi |
|---|---|
| Python | 3.10 – 3.13 |
| scikit-learn | **1.6.1** (dipaku, lihat catatan di bawah) |
| Flask | 3.0 atau lebih baru |
| pandas, numpy, matplotlib, joblib | lihat `sistem_deteksi_ddos/requirements.txt` |

Tidak membutuhkan koneksi internet saat dijalankan: seluruh CSS dan ikon sudah ada di dalam proyek.

---

## Memasang dan menjalankan

### Windows

```bash
git clone https://github.com/nnabilahan/Skripsi-ika.git
cd Skripsi-ika
python -m venv venv
venv\Scripts\activate
pip install -r sistem_deteksi_ddos/requirements.txt
python sistem_deteksi_ddos/app.py
```

### macOS / Linux

```bash
git clone https://github.com/nnabilahan/Skripsi-ika.git
cd Skripsi-ika
python3 -m venv venv
source venv/bin/activate
pip install -r sistem_deteksi_ddos/requirements.txt
python3 sistem_deteksi_ddos/app.py
```

Lalu buka **http://127.0.0.1:5000** di peramban.

---

## Berkas besar: dipindahkan lewat flashdisk

GitHub menolak berkas di atas 100 MB, jadi dua jenis berkas **tidak ikut** di dalam repositori ini.
Salin keduanya lewat flashdisk dari komputer asal, lalu letakkan **persis** di jalur berikut:

| Berkas | Ukuran | Letakkan di | Wajib? |
|---|---|---|---|
| `test_sandbox_v3.csv` | 152,5 MB | `sistem_deteksi_ddos/uploads/` | ya, untuk menguji |
| `LOF_seed42.joblib` | 311,0 MB | `sistem_deteksi_ddos/models/` | opsional |
| `train_sandbox_v3.csv` | 355,7 MB | `sistem_deteksi_ddos/uploads/` | tidak |

### Langkah memindahkannya

**Di komputer asal**

1. Buka folder proyek, masuk ke `sistem_deteksi_ddos\uploads\`.
2. Salin `test_sandbox_v3.csv` ke flashdisk.
3. Masuk ke `sistem_deteksi_ddos\models\`, salin `LOF_seed42.joblib` ke flashdisk.

**Di komputer tujuan**

1. `git clone` repositori ini lebih dulu (langkah pemasangan di atas).
2. Dari flashdisk, salin `test_sandbox_v3.csv` ke folder `sistem_deteksi_ddos\uploads\`.
3. Salin `LOF_seed42.joblib` ke folder `sistem_deteksi_ddos\models\`.
4. Jalankan aplikasi. Model yang baru disalin langsung muncul di halaman Pengaturan tanpa perlu diatur apa pun.

**Penjelasan singkat**

- `test_sandbox_v3.csv` adalah berkas yang diunggah ke halaman Uji Model. Tanpa berkas ini sistem tetap
  berjalan, hanya saja Anda perlu menyiapkan CSV uji sendiri.
- `LOF_seed42.joblib` adalah model LOF yang dilatih dari seluruh 326.127 baris trafik normal — model
  inilah yang dipakai sebagai hasil utama pada naskah. Sembilan model lain sudah ikut di repositori.
- `train_sandbox_v3.csv` **tidak perlu disalin**. Pengaman penolakan data latih bekerja dari berkas
  sidik jari `models/latih_train_sandbox_v3.fp.npz` yang sudah ikut di repositori, bukan dari CSV-nya.

---

## Isi repositori

```
Skripsi-ika/
├─ sistem_deteksi_ddos/
│  ├─ app.py                  Rute halaman, proses latar belakang, laporan PDF
│  ├─ core.py                 Praproses, skor anomali, evaluasi, pengaman data latih
│  ├─ requirements.txt
│  ├─ templates/              Halaman HTML (Jinja2)
│  ├─ static/style.css
│  ├─ models/                 9 berkas model .joblib + metadata + sidik jari data latih
│  ├─ uploads/                kosong — tempat berkas CSV uji
│  └─ runs/                   kosong — riwayat pengujian terisi sendiri saat dipakai
├─ tampilan/                  Rancangan tampilan, flowchart, dan grafik penelitian
├─ notebook_perbandingan_if_lof.ipynb   Percobaan 5 seed di Kaggle (grid search parameter)
└─ README.md
```

---

## Lima halaman

| Menu | Isi |
|---|---|
| **Dashboard** | Model mana yang unggul pada tiap ambang, kurva recall terhadap FPR aktual, kecepatan deteksi, macro recall per jenis serangan, dan tabel angka lengkap empat kolom |
| **Uji Model** | Unggah CSV, pilih model (IF / LOF / keduanya) beserta seed-nya, atur ambang yang ditampilkan dan batas baris — lalu hasil deteksi per flow tampil di halaman yang sama |
| **Perbandingan Model** | Tabel kinerja seluruh ambang, confusion matrix dengan saklar 5% / 1%, recall per jenis serangan, temuan utama, dan unduhan laporan PDF |
| **Pengaturan** | Aturan yang dipakai tiap model tersimpan: data acuan, parameter, praproses, ukuran berkas, dan ambang batasnya |
| **Riwayat** | Hasil pengujian tersimpan otomatis; dapat dibuka ulang atau dibandingkan lintas konfigurasi tanpa mengunggah ulang CSV |

---

## Cara memakai

1. Buka **Uji Model**.
2. Pilih berkas CSV uji (klik atau seret ke kotak unggah).
3. Pilih model: **Isolation Forest**, **Local Outlier Factor**, atau **Keduanya (IF + LOF)**.
   Dropdown di tiap kartu menentukan seed yang dipakai, lengkap dengan jumlah data acuannya.
4. Tekan **Mulai Uji Model**. Bilah kemajuan muncul di bagian atas halaman dan bisa dijeda atau dihentikan.
5. Hasilnya muncul di halaman yang sama; perbandingan lengkap ada di **Perbandingan Model**,
   dan ringkasannya di **Dashboard**.

### Syarat berkas CSV uji

- Berformat CSV dengan **73 kolom fitur** bernama sama persis dengan data latih model (gaya CICFlowMeter).
  Urutan kolom bebas, kolom tambahan diabaikan.
- Kolom `Label` boleh ada untuk evaluasi, tetapi tidak wajib. Tanpa label, sistem hanya menghasilkan
  prediksi tanpa metrik.
- **Bukan data latih.** Berkas yang isinya dikenali sebagai data latih akan ditolak, termasuk bila
  namanya diubah, karena sistem membandingkan sidik jari tiap barisnya.

---

## Dataset

Penelitian ini memakai **DDoS-Sandbox**, dataset trafik lapisan aplikasi dari lingkungan pengujian
terkontrol, gabungan dari proyek INSPIRE-5Gplus, CSE-CIC-IDS2018, dan testbed 5G University of Oulu.

Disusun menjadi dua berkas terpisah dengan 78 kolom (73 fitur numerik dipakai, sisanya label dan kolom
konstan):

| Berkas | Baris | BENIGN | Serangan |
|---|---|---|---|
| `train_sandbox_v3.csv` | 821.063 | 370.623 | 450.440 |
| `test_sandbox_v3.csv` | 351.895 | 159.295 | 192.600 |

Jenis serangan: `DoS Hulk`, `DoS slowloris`, `GoldenEye`, `Slowloris`, `Hulken`. Dua jenis terakhir
jumlahnya di bawah 200 baris sehingga tidak dijadikan dasar kesimpulan.

---

## Model tersimpan

Sembilan model ikut di repositori; `LOF_seed42.joblib` dipindahkan lewat flashdisk karena ukurannya 311 MB.

| Model | Seed | Data acuan | Parameter |
|---|---|---|---|
| `IF_seed42` | 42 | 326.127 | 300 pohon · max_features 0,5 |
| `IF_seed7` | 7 | 19.686 | 100 pohon · max_features 0,5 |
| `IF_seed99` | 99 | 19.705 | 300 pohon · max_features 0,5 |
| `IF_seed123` | 123 | 19.709 | 300 pohon · max_features 1,0 |
| `IF_seed2024` | 2024 | 19.685 | 300 pohon · max_features 1,0 |
| `LOF_seed42` *(flashdisk)* | 42 | 326.127 | k = 50 |
| `LOF_seed7` | 7 | 19.686 | k = 10 |
| `LOF_seed99` | 99 | 19.705 | k = 50 |
| `LOF_seed123` | 123 | 19.709 | k = 50 |
| `LOF_seed2024` | 2024 | 19.685 | k = 50 |

> **Penting.** Kesepuluh model ini **tidak hanya berbeda pada seed**. Jumlah data acuan dan sebagian
> parameternya juga berbeda, karena delapan di antaranya berasal dari percobaan *grid search* di
> notebook Kaggle (20.000 baris acuan), sedangkan pasangan seed42 dilatih ulang dari seluruh data acuan.
> Dua model hanya layak dibandingkan bila jumlah data acuan dan parameternya sama. Halaman Pengaturan
> menandai baris yang data acuannya lebih kecil, dan halaman Perbandingan memberi peringatan bila
> pasangan yang dipilih tidak setara.

---

## Catatan reproduksibilitas

- **Versi scikit-learn dipaku ke 1.6.1** karena berkas model dibuat dengan versi tersebut. Memakai versi
  lain umumnya tetap bisa memuat model, tetapi skor anomali dapat bergeser sedikit sehingga angka hasil
  tidak persis sama dengan yang dilaporkan pada naskah.
- **Tahap pengujian tidak mengandung keacakan.** Sistem hanya memuat model beserta ambangnya, jadi
  pengujian ulang dengan model, berkas, dan ambang yang sama selalu memberi confusion matrix identik.
  Satu-satunya angka yang dapat berbeda antar pengujian adalah waktu proses.
- **Riwayat menyimpan skor, bukan salinan CSV.** Hasil lama dapat dibuka dan dibandingkan kembali, tetapi
  pengujian ulang tetap membutuhkan unggahan berkas baru.

---

## Lisensi dan penggunaan

Repositori ini dibuat untuk keperluan akademik. Silakan dipakai sebagai rujukan dengan menyebutkan sumbernya.
