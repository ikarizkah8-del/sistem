# sistem_deteksi_ddos — catatan teknis

Petunjuk pemasangan, pemindahan berkas besar, dan cara memakai ada di [README utama](../README.md)
di akar repositori. Berkas ini hanya memuat catatan teknis bagi yang ingin membaca kodenya.

## Susunan berkas

| Berkas / folder | Isi |
|---|---|
| `app.py` | Rute halaman, proses latar belakang (thread + bilah kemajuan), penyusunan laporan PDF |
| `core.py` | Praproses, perhitungan skor anomali, evaluasi, dan pengaman data latih |
| `templates/` | Halaman Jinja2: `dashboard`, `uji`, `perbandingan`, `pengaturan`, `riwayat`, plus `base`, `_icons`, `_widgets`, `_strip` |
| `static/style.css` | Seluruh gaya tampilan; tidak memakai CDN sehingga berjalan tanpa internet |
| `models/` | Berkas model `.joblib`, metadata `.meta.json`, dan sidik jari data latih `.fp.npz` |
| `uploads/` | Berkas CSV yang diunggah pengguna |
| `runs/` | Riwayat pengujian: satu `.npz` (skor per flow) dan satu `.json` (ringkasan) per pengujian |

## Alur deteksi

1. `core.read_csv` membaca berkas, `core.sample_rows` mengambil sampel berstratifikasi bila ada batas baris.
2. `core.preprocess` mengganti nilai tak hingga dan kosong dengan 0, lalu menyusun kolom sesuai daftar
   fitur yang tersimpan di dalam model.
3. Sidik jari tiap baris dibandingkan dengan `*.fp.npz`. Bila 50% atau lebih baris identik dengan data
   latih, pengujian ditolak.
4. `core.anomaly_score` menghitung skor per potongan 20.000 baris agar kemajuan dapat ditampilkan dan
   pemakaian memori terkendali.
5. **Skor dihitung sekali**, lalu dibandingkan dengan setiap ambang di `b["thresholds"]`. Hasil tiap
   ambang disimpan di `r["by_fpr"]`; `apply_view()` menyalin salah satunya ke tingkat atas untuk
   ditampilkan. Inilah yang membuat tabel empat kolom tersedia tanpa biaya hitung tambahan.
6. Hasilnya disimpan ke `runs/` oleh `riwayat_simpan()` dan dapat dimuat kembali oleh `riwayat_muat()`.

## Isi berkas model (.joblib)

Satu berkas model memuat seluruh yang dibutuhkan untuk menilai data baru, sehingga tidak ada aturan
praproses yang tercecer di luar model:

```
model            objek scikit-learn (IsolationForest / LocalOutlierFactor novelty=True)
features         daftar 73 nama kolom, berikut urutannya
preprocess       'slog' atau 'slog+scaler'
scaler           StandardScaler yang dipasang dari data latih (khusus LOF)
thresholds       {0.05: ..., 0.01: ...} — persentil skor data latih
n_train          jumlah baris data acuan
trained_on       nama berkas data latih
seed, params, sklearn_version, created
```

## Catatan

- Satu proses berat berjalan dalam satu waktu (`run_job`), dengan titik periksa `job_control()` di sela
  potongan sehingga tombol Jeda dan Hentikan benar-benar bekerja.
- Metadata model di-cache di `*.meta.json` agar daftar model tidak perlu memuat objek model yang besar.
- Sistem ini **tidak melatih** model. Berkas model dibuat di luar sistem; percobaan parameternya ada di
  `notebook_perbandingan_if_lof.ipynb` pada akar repositori.
