---
title: Pakcoy Vision
emoji: 🌱
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Pakcoy Vision

Aplikasi web untuk membaca **fase pertumbuhan pakcoy hidroponik** dari satu foto.
Aplikasi menjalankan pipeline penelitian secara utuh: preprocessing citra →
segmentasi instance daun (YOLOv11-seg) → pengukuran bentuk dengan OpenCV →
agregasi fitur → klasifikasi fase → estimasi usia → rekomendasi budidaya.

Antarmuka berbahasa Indonesia. Dapat dijalankan lokal, atau sebagai demo publik
lewat Docker (lihat `Dockerfile`) tanpa mengubah kode aplikasi.

## Menjalankan aplikasi

Prasyarat: **Python 3.12** dan Windows (skrip `run.bat`). Di Linux/macOS gunakan
perintah `python` yang setara.

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-ml.txt   :: opsional, untuk YOLO
run.bat
```

`run.bat` membuka `http://127.0.0.1:8000` secara otomatis. Tanpa `run.bat`:

```bat
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Deploy sebagai demo publik (Hugging Face Spaces)

Vercel tidak dapat menjalankan model ini: `torch` berukuran ~1 GB dan batas
fungsi serverless Vercel 250 MB. Gunakan Docker.

1. Buat Space baru di <https://huggingface.co/new-space> — SDK **Docker**,
   visibility **Public**, nama misalnya `pakcoy-vision`.
2. Push repositori (token Hugging Face dipakai sebagai password):

```bat
git add -f models/best.pt
git add Dockerfile .dockerignore README.md
git commit -m "Deploy: Dockerfile untuk Hugging Face Spaces"
git remote add hf https://huggingface.co/spaces/<username>/pakcoy-vision
git push hf main
```

3. Tunggu build selesai. Aplikasi tersedia di
   `https://<username>-pakcoy-vision.hf.space`.

`models/best.pt` di-ignore oleh `.gitignore`, jadi wajib `git add -f` agar bobot
ikut terkirim. Space gratis akan tidur setelah lama tidak diakses; kunjungan
pertama butuh 30–60 detik untuk bangun.

## Dua mode operasi

| Mode | Kapan aktif | Yang dipakai |
| --- | --- | --- |
| **YOLOv11-seg** | `models/best.pt` (atau bobot lain yang dikenali) tersedia dan `ultralytics` terpasang | Segmentasi instance daun oleh model utama. |
| **Demo klasik** | Bobot belum tersedia atau model tidak menemukan daun | HSV + Excess Green + Otsu + watershed, lalu aturan heuristik. |

Model segmentasi aktif pada repositori ini adalah `models/best.pt`
(YOLOv11n-seg, satu kelas `daun-pakcoy`, mAP50 mask 0,953 pada validasi).
Ringkasan itu dibaca langsung dari checkpoint, bukan ditulis manual di halaman.

Klasifikasi fase masih memakai aturan heuristik karena artefak model klasifikasi
(`pakcoy_mlp.tflite` atau `best_rf.pkl`) belum tersedia. Halaman menandainya
sebagai "Aturan heuristik (demo)" dan menambahkan saran untuk melatih classifier.
Jangan menuliskan hasil fase sebagai keluaran model terlatih sebelum artefak itu
ada.

Mode demo bukan pengganti model penelitian. Tujuannya agar seluruh alur dan
antarmuka tetap dapat diuji sebelum artefak training tersedia. Status mode aktif
selalu ditampilkan di header dan badge mesin pada panel analisis.

Cara memasang model: lihat [`models/README.md`](models/README.md).

## Alur yang diimplementasikan

Alur ini mengikuti diagram penelitian di `app/static/assets/diagram-alur.png`.

1. **Akuisisi & preprocessing**: resize sisi terpanjang 640 px, CLAHE, Gaussian
   blur ringan (`app/pipeline.py`). Citra hasil ini dipakai untuk tampilan dan
   segmentasi klasik.
2. **Segmentasi instance**: YOLOv11n-seg memakai citra asli setelah resize 640 px
   (tanpa CLAHE) agar sesuai distribusi data latih; fallback klasik saat bobot
   belum ada.
3. **Pengukuran OpenCV**: kontur terbesar per mask, `minAreaRect` untuk panjang
   dan lebar, luas, rasio L/W, dan centroid. Bila ada sisi kontur lurus yang
   panjangnya minimal 70% panjang daun, daun dianggap terpotong objek lurus
   (misalnya penggaris). Bagian yang tertutup diperkirakan dengan mencerminkan
   bagian yang terlihat terhadap garis penghalang, lalu lebarnya diukur ulang.
4. **Agregasi per tanaman**: `leaf_count`, `mean_length`, `mean_width`,
   `max_length`, `max_width`, `total_area`, `mean_ratio` (7 fitur model).
5. **Klasifikasi fase**: TFLite → scikit-learn → heuristik demo
   (`app/classifier.py`).
6. **Estimasi usia & rekomendasi**: aturan pakar per fase dan per diagnosis
   (`app/recommendations.py`).

Semua keluaran antara ditampilkan: citra asli, citra hasil preprocessing, citra
anotasi, thumbnail mask tiap daun, tabel fitur (CSV), fitur agregat, probabilitas
tiga kelas, estimasi usia, dan saran untuk petani.

## Struktur proyek

```text
app/
  config.py           path, ambang, daftar sampel, urutan fitur
  pipeline.py         preprocessing, segmentasi, pengukuran, agregasi, anotasi
  classifier.py       pemuatan TFLite/scikit-learn + heuristik demo
  recommendations.py  aturan rekomendasi per fase
  main.py             route FastAPI
  static/
    index.html        antarmuka satu halaman
    styles.css        styling responsif
    app.js            unggah, analisis, render hasil, unduh CSV/JSON
    samples/          contoh foto hari ke-5, 20, 30, 42
    assets/           diagram alur penelitian
models/               bobot YOLO & artefak klasifikasi (lihat README di dalam)
tests/                uji asap pipeline (stdlib unittest)
```

## Pengujian

Uji asap memakai `unittest` bawaan Python, tanpa dependensi tambahan:

```bat
.venv\Scripts\python -m unittest discover -s tests -v
```

Uji mencakup preprocessing, segmentasi klasik, pengukuran, agregasi, prediksi,
kalibrasi satuan, dan rekomendasi tanpa fase.

## API

| Metode | Path | Fungsi |
| --- | --- | --- |
| `GET` | `/` | Antarmuka web. |
| `GET` | `/api/health` | Cek server hidup. |
| `GET` | `/health` | Alias health check untuk monitor layanan. |
| `GET` | `/api/status` | Status model, sampel, dan batas unggahan. |
| `POST` | `/api/analyze` | Analisis satu foto (multipart). |
| `GET` | `/docs` | Dokumentasi OpenAPI interaktif. |

Contoh pemanggilan:

```bat
curl -X POST http://127.0.0.1:8000/api/analyze ^
  -F "file=@app/static/samples/day-20.jpg" ^
  -F "conf=0.25" ^
  -F "iou=0.5" ^
  -F "engine=auto"
```

Parameter `POST /api/analyze`:

| Field | Default | Keterangan |
| --- | --- | --- |
| `file` | wajib | JPG/PNG/WEBP, maks 20 MB. |
| `conf` | `0.25` | Ambang confidence YOLO (0,01–0,99). |
| `iou` | `0.50` | Ambang NMS IoU (0,05–0,95). |
| `pixels_per_cm` | kosong | Kalibrasi opsional untuk tampilan cm. |
| `engine` | `auto` | `auto`, `yolo`, atau `classic`. |

Respons memuat `image` (data URL asli/preprocessing/anotasi), `leaves`, `aggregated`,
`aggregated_raw`, `prediction`, `recommendations`, `timings`, `units`, dan
`warnings`. Objek `engine.yolo.model` memuat arsitektur, kelas, tanggal training,
dan metrik validasi yang dibaca dari checkpoint aktif.

## Catatan metodologis

- Foto diasumsikan memuat **satu tanaman**. Banyak tanaman dalam satu foto membuat
  agregasi fitur tidak lagi mewakili satu individu.
- Fitur klasifikasi selalu dihitung dalam piksel. Kalibrasi piksel/cm hanya
  memengaruhi satuan yang ditampilkan, agar distribusi fitur tetap konsisten
  dengan data latih.
- Kalibrasi tidak diisi otomatis. Selama kolom kalibrasi kosong, hasil tampil
  dalam piksel. `DEFAULT_PIXELS_PER_CM` (`app/config.py`, dapat diganti lewat
  environment variable `PAKCOY_PIXELS_PER_CM`) hanya dipakai sebagai contoh
  angka di placeholder; nilai `38,5` px/cm belum tervalidasi, jadi ukur sendiri
  dengan penggaris sebelum pengambilan data penelitian.
- Daun yang saling menutup rapat dapat menyatu atau terpecah saat segmentasi.
  Verifikasi visual tetap diperlukan.
- Lebar daun adalah lebar proyeksi tegak lurus sumbu daun. Bila daun terpotong
  objek lurus seperti penggaris, piksel yang tertutup tidak ada di citra. Untuk
  daun seperti ini, lebar direkonstruksi dengan asumsi simetri terhadap garis
  penghalang dan ditandai `≈` pada tabel. Panjang dan luas tetap dihitung dari
  piksel yang terlihat, jadi keduanya bisa lebih kecil dari ukuran asli.
- Mode demo memakai aturan heuristik, bukan model terlatih. Gunakan bobot dan
  classifier hasil training untuk hasil yang dapat dipertanggungjawabkan secara
  ilmiah.

## Pemecahan masalah

| Gejala | Penyebab umum | Solusi |
| --- | --- | --- |
| Header menampilkan “Mode demo” | `models/best.pt` tidak ditemukan atau `ultralytics` belum terpasang | Letakkan bobot di `models/best.pt`, lalu muat ulang halaman. Bobot yang diganti saat server berjalan terdeteksi otomatis. |
| “Bobot ditemukan, tetapi ultralytics belum terpasang” | Dependensi ML belum diinstal | `pip install -r requirements-ml.txt`. |
| Prediksi memakai “Aturan heuristik (demo)” | Artefak classifier belum ada | Lihat opsi TFLite/scikit-learn di `models/README.md`. |
| YOLO tidak menemukan daun, hasil pindah ke mode klasik | Foto sangat dekat, daun saling menutup, atau skala berbeda dari data latih | Jauhkan kamera agar satu tanaman utuh masuk bingkai, pakai latar kontras, lalu ulangi. |
| Tidak ada daun terdeteksi | Latar ramai, cahaya kurang, objek terlalu kecil | Gunakan latar kontras, foto lebih dekat, atau naikkan confidence. |
| `ModuleNotFoundError` | Perintah dijalankan di luar `.venv` | Gunakan `.venv\Scripts\python`. |
