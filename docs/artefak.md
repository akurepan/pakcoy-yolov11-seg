# Artefak penelitian

Dokumen ini merujuk pada isi folder [`hasil/`](../hasil). Folder tersebut
menyimpan artefak untuk memverifikasi hasil penelitian: bobot model, berkas
pendukung training, dan notebook yang dipakai di Kaggle. Semua perintah di bawah
dijalankan dari root repositori.

## `pakcoy_artifacts.zip`

Zip ini **dibangun ulang pada 16 November 2026**. Salinan sebelumnya rusak:
CRC `best.pt` tidak cocok sehingga `best.pt`, `BoxPR_curve.png`,
`MaskPR_curve.png`, dan `results.csv` tidak dapat diekstrak (Windows maupun
Python sama-sama gagal). Kurva dan CSV lama tidak dapat dipulihkan; yang
tersisa hanya entri yang masih sehat, lalu dilengkapi bobot dari `models/`
dan satu gambar prediksi dari output notebook.

| Berkas | Ukuran (byte) | SHA-256 |
| --- | ---: | --- |
| `best.pt` | 6.034.724 | `19d61d93a5494a47b04ae99807bc9a535c6164e666e17aad0f5bf6823e69bfa1` |
| `last.pt` | 6.047.780 | `be1f59e3213216e2f60e95221f84d2fa8498ee0fea4ca5aff855e292ccc59896` |
| `data.yaml` | 182 | `0f8651ca72a4e90e3a4f5f771e9c255e91cb87a2dcb310553d83796ec41d6cd8` |
| `prediction_test.png` | 387.878 | `9220da4b58f473cf94696d4d0f04e0aec936820fedba7b6d89cc97edf0578192` |

`best.pt` di dalam zip identik dengan `models/best.pt` yang dipakai aplikasi.

Verifikasi integritas:

```bat
.venv\Scripts\python -c "import zipfile; print(zipfile.ZipFile('hasil/pakcoy_artifacts.zip').testzip())"
```

Keluaran harus `None`. Untuk mengekstrak seluruh isi:

```bat
.venv\Scripts\python -c "import zipfile; zipfile.ZipFile('hasil/pakcoy_artifacts.zip').extractall('hasil/artifacts')"
```

## `checkpoint-skripsi-pakcoy.ipynb`

Notebook Kaggle yang dipakai untuk training, evaluasi validasi/testing, dan
prediksi sampel YOLOv11n-seg. Isinya juga mencatat log evaluasi; angka di bawah
diambil dari log tersebut.

## Metrik

### Validasi (253 gambar, 1.054 instance)

Angka ini yang tertanam di checkpoint dan ditampilkan aplikasi.

| Metrik | Box | Mask |
| --- | ---: | ---: |
| Precision | 0,938 | 0,941 |
| Recall | 0,915 | 0,915 |
| mAP50 | 0,956 | 0,953 |
| mAP50-95 | 0,844 | 0,835 |

### Testing (127 gambar, 613 instance)

Evaluasi ulang di notebook memakai Ultralytics 8.4.208.

| Metrik | Box | Mask |
| --- | ---: | ---: |
| Precision | 0,962 | 0,965 |
| Recall | 0,896 | 0,898 |
| mAP50 | 0,943 | 0,946 |
| mAP50-95 | 0,828 | 0,811 |

Selisih kecil antara tabel validasi dan metadata checkpoint wajar karena
evaluasi dijalankan pada versi Ultralytics yang berbeda.

## Yang belum tersedia

- **Artefak klasifikasi fase** (`pakcoy_mlp.tflite`, `feature_scaler.pkl`,
  `label_encoder.pkl`, `best_rf.pkl`) belum ada. Prediksi fase memakai aturan
  heuristik dan selalu diberi label "Aturan heuristik (demo)". Langkah
  melatihnya ada di [klasifikasi.md](klasifikasi.md).
- **Deteksi YOLO pada foto close-up ekstrem.** `app/static/samples/day-42.jpg`
  tidak menghasilkan instance daun bahkan pada `conf=0.05`, sehingga
  `engine=auto` turun ke segmentasi klasik (hasil fase tetap "Siap Panen").
  Penyebabnya domain gap: foto memenuhi bingkai, daun saling menutup, dan latar
  berbeda dari data latih. Menurunkan ambang tidak menolong; perbaikannya adalah
  menambah data hari 35–45 dengan variasi jarak dan sudut.
- Kurva PR dan confusion matrix dari zip lama hilang karena kompresinya rusak.
  Jangan membuat ulang kurva secara manual lalu menyajikannya sebagai keluaran
  training; latih ulang atau ekspor dari notebook bila kurva itu dibutuhkan
  untuk laporan.
