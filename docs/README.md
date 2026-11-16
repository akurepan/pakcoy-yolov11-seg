# Dokumentasi Pakcoy Vision

Semua panduan proyek ada di folder ini. Perintah contoh dijalankan dari **root
repositori** (folder yang berisi `app/`, `models/`, dan `README.md`) memakai
Python dari `.venv`.

## Urutan baca yang disarankan

| Urutan | Dokumen | Isi |
| --- | --- | --- |
| 1 | [instalasi.md](instalasi.md) | Prasyarat, memasang dependensi, menjalankan server, pengujian. |
| 2 | [penggunaan.md](penggunaan.md) | Cara memakai antarmuka, daftar endpoint API, dan parameter analisis. |
| 3 | [arsitektur.md](arsitektur.md) | Alur pipeline, dua mode operasi, dan struktur folder. |
| 4 | [model.md](model.md) | Bobot YOLO, artefak klasifikasi, urutan fitur, dan metadata. |
| 5 | [klasifikasi.md](klasifikasi.md) | **Kenapa fase masih demo dan cara melatih modelnya.** |
| 6 | [artefak.md](artefak.md) | Isi zip artefak, metrik validasi/testing, dan provenance. |
| 7 | [metodologi.md](metodologi.md) | Asumsi, batasan pengukuran, dan hal yang tidak boleh diklaim. |
| 8 | [deploy.md](deploy.md) | Deploy ke Hugging Face, Vercel, atau Docker + checklist verifikasi. |
| 9 | [desain.md](desain.md) | Arah visual dan keputusan desain antarmuka. |

## Peta cepat berdasarkan kebutuhan

| Saya ingin... | Buka |
| --- | --- |
| Menjalankan aplikasi di laptop | [instalasi.md](instalasi.md) |
| Mengunggah foto dan membaca hasil | [penggunaan.md](penggunaan.md) |
| Tahu kenapa prediksi fase masih "demo" | [klasifikasi.md](klasifikasi.md) |
| Melatih classifier fase (RF/MLP) | [klasifikasi.md](klasifikasi.md) |
| Membuat link demo publik | [deploy.md](deploy.md) |
| Memeriksa metrik untuk laporan | [artefak.md](artefak.md) |
| Menghindari klaim yang salah saat sidang | [metodologi.md](metodologi.md) |

## Berkas di luar folder ini

- `README.md` di root: ringkasan proyek, status, dan mulai cepat.
- `models/README.md` dan `hasil/README.md`: penunjuk singkat ke dokumen di sini.
- `requirements.txt`, `requirements-ml.txt`: daftar dependensi.
- `Dockerfile`, `vercel.json`, `api/index.py`: konfigurasi deploy.
