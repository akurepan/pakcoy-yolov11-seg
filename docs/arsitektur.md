# Arsitektur dan alur pipeline

## Tumpukan teknologi

- **FastAPI + Uvicorn** untuk server dan API.
- **OpenCV headless + NumPy** untuk pengolahan citra dan pengukuran.
- **Ultralytics YOLOv11n-seg** untuk segmentasi instance daun (opsional).
- **scikit-learn / TensorFlow Lite** untuk klasifikasi fase (opsional; lihat
  [klasifikasi.md](klasifikasi.md)).
- **HTML/CSS/JS tanpa framework** untuk antarmuka satu halaman.

## Alur yang diimplementasikan

Alur mengikuti diagram penelitian di `app/static/assets/diagram-alur.png`.

1. **Akuisisi & preprocessing** — resize sisi terpanjang 640 px, CLAHE, dan
   Gaussian blur ringan (`app/pipeline.py`). Citra ini dipakai untuk tampilan
   dan segmentasi klasik.
2. **Segmentasi instance** — YOLOv11n-seg menerima citra asli setelah resize
   640 px (tanpa CLAHE) agar sesuai distribusi data latih. Fallback ke
   segmentasi klasik bila bobot tidak ada atau model tidak menemukan daun.
3. **Pengukuran OpenCV** — kontur terbesar per mask, `minAreaRect` untuk panjang
   dan lebar, luas, rasio L/W, dan centroid. Bila ada sisi kontur lurus yang
   panjangnya minimal 70% panjang daun, daun dianggap terpotong objek lurus
   (misalnya penggaris). Bagian tertutup diperkirakan dengan mencerminkan bagian
   yang terlihat terhadap garis penghalang, lalu lebarnya diukur ulang.
4. **Agregasi per tanaman** — `leaf_count`, `mean_length`, `mean_width`,
   `max_length`, `max_width`, `total_area`, `mean_ratio` (7 fitur model).
5. **Klasifikasi fase** — TFLite → scikit-learn → heuristik demo
   (`app/classifier.py`). Detail dan cara melatih ada di
   [klasifikasi.md](klasifikasi.md).
6. **Estimasi usia & rekomendasi** — aturan pakar per fase dan per diagnosis
   (`app/recommendations.py`).

Semua keluaran antara ditampilkan: citra asli, hasil preprocessing, citra
anotasi, thumbnail mask tiap daun, tabel fitur (CSV), fitur agregat,
probabilitas tiga kelas, estimasi usia, dan saran untuk petani.

## Dua mode operasi

| Mode | Kapan aktif | Yang dipakai |
| --- | --- | --- |
| **YOLOv11-seg** | `models/best.pt` (atau bobot lain yang dikenali) tersedia dan `ultralytics` terpasang | Segmentasi instance daun oleh model utama. |
| **Demo klasik** | Bobot belum tersedia atau model tidak menemukan daun | HSV + Excess Green + Otsu + watershed, lalu aturan heuristik. |

Mode demo bukan pengganti model penelitian. Tujuannya agar seluruh alur dan
antarmuka tetap dapat diuji sebelum artefak training tersedia. Status mode aktif
selalu ditampilkan di header dan badge mesin pada panel analisis.

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
models/               best.pt (disertakan) & artefak klasifikasi
hasil/                zip artefak, notebook training, metrik
tests/                uji asap pipeline + integritas artefak
tools/                skrip bangun CSV fitur dan latih classifier
docs/                 seluruh dokumentasi
Dockerfile            image untuk Hugging Face Spaces / VPS
vercel.json           rewrite untuk deployment Vercel
api/index.py          entrypoint serverless Vercel (mode klasik)
tests/                uji pipeline, artefak, dan tautan dokumentasi
```

## Prinsip data

- Fitur klasifikasi dihitung **dalam piksel** dari citra 640 px. Kalibrasi
  piksel/cm hanya mengubah satuan tampilan, supaya distribusi fitur tetap sama
  dengan data latih.
- Satu foto = satu tanaman. Banyak tanaman dalam satu foto membuat agregasi
  fitur tidak lagi mewakili satu individu.
- Angka model di halaman diambil dari checkpoint melalui `/api/status`, bukan
  ditulis manual.
