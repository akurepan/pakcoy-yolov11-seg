# Penggunaan aplikasi

## Antarmuka web

Alur pemakaian di `http://127.0.0.1:8000`:

1. Pilih salah satu **foto sampel** (hari 5, 20, 30, 42) atau unggah foto sendiri
   (JPG/PNG/WEBP, maksimal 20 MB).
2. Atur **mesin** (`auto`, `yolo`, `classic`), **confidence**, dan **IoU** bila
   perlu. Biarkan nilai bawaan untuk pemakaian normal.
3. Isi **kalibrasi piksel/cm** hanya jika sudah mengukur dengan penggaris.
   Selama kosong, hasil tampil dalam piksel.
4. Tekan **Analisis**. Halaman menampilkan:
   - citra asli, hasil preprocessing, dan citra anotasi,
   - thumbnail mask tiap daun beserta ukurannya,
   - tabel fitur per daun (bisa diunduh CSV),
   - fitur agregat dan probabilitas tiga fase,
   - estimasi usia, rekomendasi budidaya, dan daftar peringatan,
   - badge mesin yang menandai apakah hasil berasal dari YOLO, klasik, atau
     classifier demo.
5. Hasil lengkap dapat diunduh sebagai **CSV** atau **JSON**.

Badge **Mode demo** berarti satu dari dua hal: bobot YOLO tidak tersedia, atau
classifier fase masih memakai aturan heuristik. Keduanya ditulis terpisah di
panel analisis agar tidak tertukar.

## Tips foto

- Satu tanaman per foto; agregasi fitur mengasumsikan satu individu.
- Seluruh tajuk masuk bingkai, kamera tidak terlalu dekat.
- Latar kontras dan cahaya merata; hindari bayangan keras.
- Sertakan penggaris bila ingin kalibrasi, tetapi jangan menutupi daun.

## API

| Metode | Path | Fungsi |
| --- | --- | --- |
| `GET` | `/` | Antarmuka web. |
| `GET` | `/api/health` | Cek server hidup. |
| `GET` | `/health` | Alias health check untuk monitor layanan. |
| `GET` | `/api/status` | Status model, daftar sampel, dan batas unggahan. |
| `POST` | `/api/analyze` | Analisis satu foto (multipart). |
| `GET` | `/docs` | Dokumentasi OpenAPI interaktif. |

Contoh (Windows `cmd`, tanda `^` adalah lanjutan baris):

```bat
curl -X POST http://127.0.0.1:8000/api/analyze ^
  -F "file=@app/static/samples/day-20.jpg" ^
  -F "conf=0.25" ^
  -F "iou=0.5" ^
  -F "engine=auto"
```

Di PowerShell, gunakan `curl.exe` dan tanda backtick, atau satu baris penuh.

### Parameter `POST /api/analyze`

| Field | Default | Keterangan |
| --- | --- | --- |
| `file` | wajib | JPG/PNG/WEBP, maks 20 MB. |
| `conf` | `0.25` | Ambang confidence YOLO (0,01–0,99). |
| `iou` | `0.50` | Ambang NMS IoU (0,05–0,95). |
| `pixels_per_cm` | kosong | Kalibrasi opsional untuk tampilan cm. Menerima `38,5` atau `38.5`. |
| `engine` | `auto` | `auto`, `yolo`, atau `classic`. |

Kesalahan validasi dikembalikan sebagai JSON `detail` dengan status:

| Status | Arti |
| --- | --- |
| `413` | Berkas melebihi 20 MB. |
| `415` | Tipe berkas bukan citra. |
| `422` | Berkas kosong/rusak, parameter di luar rentang, atau engine tidak dikenal. |
| `500` | Pipeline gagal saat runtime. |

### Bentuk respons

Objek utama:

| Kunci | Isi |
| --- | --- |
| `image` | Data URL citra asli, preprocessing, dan anotasi. |
| `engine` | Mesin yang diminta (`requested`), yang dipakai (`used`), catatan, dan metadata bobot. |
| `leaves` | Daftar daun: panjang, lebar, luas, rasio, centroid, mask. |
| `aggregated` | Fitur agregat untuk tampilan (dibulatkan, mengikuti satuan). |
| `aggregated_raw` | Fitur agregat dalam piksel, dipakai classifier. |
| `prediction` | Fase, probabilitas, estimasi usia, sumber model, dan `is_demo`. |
| `recommendations` | Saran per fase dan per diagnosis. |
| `timings` | Waktu preprocessing, segmentasi, fitur, klasifikasi, total. |
| `units` | Satuan yang dipakai (`px` atau `cm`). |
| `warnings` | Peringatan yang harus dibaca pengguna. |

`engine.yolo.model` memuat arsitektur, kelas, tanggal training, dan metrik
validasi yang dibaca dari checkpoint aktif. `prediction.source` bernilai
`heuristic`, `sklearn`, atau `tflite`.

> Jangan memakai angka fase dari `prediction` sebagai keluaran model terlatih
> selama `prediction.is_demo` masih `true`. Lihat [metodologi.md](metodologi.md).
