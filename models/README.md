# Folder model

Letakkan artefak hasil training di folder ini. Aplikasi mendeteksi berkas secara
otomatis, termasuk saat bobot diganti ketika server sedang berjalan (hot reload),
jadi tidak perlu mengubah kode.

## 1. Model segmentasi (wajib untuk mode YOLO)

| Berkas | Keterangan |
| --- | --- |
| `best.pt` | Model utama yang dipakai aplikasi: YOLOv11n-seg, satu kelas `daun-pakcoy`. |
| `pakcoy_leaf_seg.pt` | Nama alternatif yang juga dikenali. |
| `yolo11s-seg-pakcoy.pt` | Nama alternatif yang juga dikenali. |
| `pakcoy_yolo11s_seg.pt` | Nama alternatif yang juga dikenali. |

Urutan pencarian: `best.pt` → `pakcoy_leaf_seg.pt` → `yolo11s-seg-pakcoy.pt` →
`pakcoy_yolo11s_seg.pt` → `../hasil/yolo11n-seg.pt` →
`../dataset/dataset-pakcoy.pt`. Path lain dapat dipaksa lewat environment
variable:

```bat
set PAKCOY_YOLO_WEIGHTS=D:\path\ke\best.pt
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Metrik validasi yang tertanam di checkpoint `best.pt` (dibaca otomatis dan
ditampilkan di halaman):

| Metrik | Box | Mask |
| --- | --- | --- |
| Precision | 0,938 | 0,941 |
| Recall | 0,915 | 0,915 |
| mAP50 | 0,956 | 0,953 |
| mAP50-95 | 0,844 | 0,835 |

Model dilatih 100 epoch pada `imgsz` 640 dengan satu kelas. Inferensi memakai
citra asli setelah resize 640 px, bukan citra CLAHE, agar sesuai distribusi data
latih. Mask instance dirasterisasi dari poligon `masks.xy` pada resolusi citra
asli supaya tidak bergeser akibat letterbox.

> **Penting:** jangan memakai bobot COCO bawaan (`yolo11s-seg.pt` dari Ultralytics)
> untuk hasil ilmiah. Bobot tersebut tidak mengenal kelas daun pakcoy. Jika bobot
> COCO tetap dipakai, objek lain (mis. pot) dapat terbaca sebagai daun.

Checkpoint hasil training yang disimpan di `dataset/dataset-pakcoy.pt` juga
terdeteksi otomatis, tetapi hanya sebagai cadangan terakhir.

Tanpa bobot terlatih, aplikasi otomatis memakai segmentasi klasik
(HSV + Excess Green + watershed) sebagai mode demo. Seluruh halaman tetap dapat
diuji end-to-end.

## 2. Model klasifikasi fase (opsional)

Prioritas pemuatan: **TFLite → scikit-learn → heuristik demo**.

### Opsi A: TFLite (rekomendasi untuk deployment)

| Berkas | Keterangan |
| --- | --- |
| `pakcoy_mlp.tflite` | Model MLP yang sudah dikonversi ke TensorFlow Lite. |
| `feature_scaler.pkl` | Scaler yang sama dengan saat training. |
| `label_encoder.pkl` | Label encoder kelas fase. |

Runtime yang didukung: `tensorflow` atau `tflite-runtime`.

```bat
.venv\Scripts\python -m pip install tensorflow
```

### Opsi B: scikit-learn (baseline penelitian)

| Berkas | Keterangan |
| --- | --- |
| `best_mlp_sklearn.pkl` | Model MLP scikit-learn. |
| `best_rf.pkl` | Random Forest baseline. |
| `label_encoder.pkl` | Label encoder kelas fase. |

### Urutan fitur masukan (harus sama dengan saat training)

```text
leaf_count, mean_length, mean_width, max_length, max_width, total_area, mean_ratio
```

Fitur dihitung dari citra yang sudah di-resize ke sisi terpanjang **640 px**.
Kalibrasi piksel/cm hanya mengubah satuan yang ditampilkan; model selalu menerima
fitur dalam piksel.

## 3. Metadata opsional

Metadata model segmentasi (arsitektur, kelas, epoch, tanggal training, metrik
validasi) dibaca langsung dari checkpoint `best.pt`, jadi tidak perlu diisi
manual. Informasi ini tampil di `/api/status` dan di halaman.

`model_info.json` hanya untuk ringkasan tambahan model klasifikasi. Isinya
diteruskan apa adanya ke `/api/status`. Contoh:

```json
{
  "dataset": "pakcoy-hidroponik",
  "classes": ["Penyemaian", "Vegetatif", "Siap Panen"],
  "mlp_accuracy": 0.94,
  "trained_at": "2026-01-01"
}
```

Angka pada contoh ini hanya ilustrasi. Jangan menyalinnya ke laporan kecuali
berasal dari hasil evaluasi yang sebenarnya.
