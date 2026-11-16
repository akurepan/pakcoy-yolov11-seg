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
Pipeline penelitian berjalan utuh: preprocessing citra → segmentasi instance
daun (YOLOv11-seg) → pengukuran bentuk dengan OpenCV → agregasi fitur →
klasifikasi fase → estimasi usia → rekomendasi budidaya. Antarmuka berbahasa
Indonesia.

## Status

- **Kode dan bobot model** publik di
  <https://github.com/akurepan/pakcoy-yolov11-seg>. `models/best.pt` ikut
  repositori, jadi hasil clone langsung memakai YOLOv11-seg.
- **Demo publik:** belum aktif. Konfigurasi sudah siap; langkah deploy ada di
  [docs/deploy.md](docs/deploy.md).
- **Klasifikasi fase:** masih aturan heuristik (demo). Artefak model belum ada;
  cara melatih ada di [docs/klasifikasi.md](docs/klasifikasi.md).

## Mulai cepat

```bat
git clone https://github.com/akurepan/pakcoy-yolov11-seg.git
cd pakcoy-yolov11-seg
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-ml.txt
run.bat
```

Aplikasi terbuka di `http://127.0.0.1:8000`. Detail prasyarat, pengujian, dan
pemecahan masalah: [docs/instalasi.md](docs/instalasi.md).

## Dua mode operasi

| Mode | Kapan aktif | Yang dipakai |
| --- | --- | --- |
| **YOLOv11-seg** | `models/best.pt` tersedia dan `ultralytics` terpasang | Segmentasi instance daun oleh model penelitian. |
| **Demo klasik** | Bobot tidak ada atau model tidak menemukan daun | HSV + Excess Green + Otsu + watershed, lalu aturan heuristik. |

Status mode aktif selalu tampil di header dan badge mesin pada panel analisis.
Mode demo bukan pengganti model penelitian.

## Dokumentasi

| Dokumen | Isi |
| --- | --- |
| [docs/README.md](docs/README.md) | Indeks dokumentasi dan urutan baca. |
| [docs/instalasi.md](docs/instalasi.md) | Prasyarat, menjalankan lokal, pengujian. |
| [docs/penggunaan.md](docs/penggunaan.md) | Antarmuka, endpoint API, parameter. |
| [docs/arsitektur.md](docs/arsitektur.md) | Alur pipeline, dua mode, struktur folder. |
| [docs/model.md](docs/model.md) | Bobot YOLO, artefak klasifikasi, urutan fitur. |
| [docs/klasifikasi.md](docs/klasifikasi.md) | Melatih classifier fase (RF/MLP/TFLite). |
| [docs/artefak.md](docs/artefak.md) | Isi zip artefak, metrik, provenance. |
| [docs/metodologi.md](docs/metodologi.md) | Asumsi, batasan, dan klaim yang aman. |
| [docs/deploy.md](docs/deploy.md) | Hugging Face, Vercel, Docker, checklist. |
| [docs/desain.md](docs/desain.md) | Arah visual dan keputusan desain. |

## Struktur singkat

```text
app/        server FastAPI, pipeline, classifier, antarmuka web
models/     best.pt (disertakan) dan artefak classifier
hasil/      zip artefak, notebook training, metrik
tools/      bangun CSV fitur dan latih classifier fase
tests/      uji asap pipeline + integritas artefak
docs/       seluruh dokumentasi
Dockerfile  image untuk Hugging Face Spaces / VPS
api/        entrypoint serverless Vercel (mode klasik)
```

## Lisensi

MIT — lihat [LICENSE](LICENSE).
