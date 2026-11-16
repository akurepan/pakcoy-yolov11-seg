# Klasifikasi fase: status dan cara melatih

## Ringkasan: kenapa masih "demo"?

**Bukan karena fiturnya belum dibuat.** Kode untuk memuat dan memakai model
klasifikasi sudah ada di `app/classifier.py`, begitu juga tampilan probabilitas
tiga fase, estimasi usia, dan label sumber model di antarmuka.

Yang belum ada adalah **berkas model hasil training** (artefak) dan **data
berlabel fase**. Tanpa artefak, aplikasi otomatis memakai aturan heuristik dan
menandainya "Aturan heuristik (demo)". Notebook di `hasil/` juga hanya melatih
YOLO untuk segmentasi, belum ada tahap klasifikasi fase.

Urutan pemuatan model di aplikasi:

```text
pakcoy_mlp.tflite + feature_scaler.pkl + label_encoder.pkl   (butuh TensorFlow)
        ↓ jika tidak ada
best_mlp_sklearn.pkl  atau  best_rf.pkl + label_encoder.pkl  (butuh scikit-learn)
        ↓ jika tidak ada
Aturan heuristik (demo)
```

## Berkas yang diharapkan aplikasi

| Berkas di `models/` | Dipakai untuk | Catatan |
| --- | --- | --- |
| `pakcoy_mlp.tflite` | TFLite | Butuh `tensorflow` atau `tflite-runtime`. |
| `feature_scaler.pkl` | TFLite | Scaler yang sama saat training. |
| `label_encoder.pkl` | TFLite & sklearn | `classes_` menentukan urutan label. |
| `best_mlp_sklearn.pkl` | scikit-learn | Harus punya `predict_proba`. |
| `best_rf.pkl` | scikit-learn | Random Forest baseline. |
| `model_info.json` | Opsional | Ringkasan tambahan, diteruskan ke `/api/status`. |

Aturan penting:

- Model scikit-learn dimuat dengan `pickle.load`, jadi **simpan dengan
  `pickle.dump`**, bukan `joblib.dump`.
- Untuk MLP scikit-learn, **bungkus scaler dan MLP dalam satu `Pipeline`**.
  Aplikasi tidak memuat `feature_scaler.pkl` pada jalur scikit-learn.
- Urutan fitur harus persis: `leaf_count`, `mean_length`, `mean_width`,
  `max_length`, `max_width`, `total_area`, `mean_ratio`.
- Fitur dilatih dalam **piksel** dari citra 640 px, memakai nilai mentah
  (`aggregated_raw`), bukan nilai tampilan yang sudah dibulatkan atau dikonversi
  ke cm.
- `label_encoder.pkl` harus dilatih dari label string yang sama dengan data.

## Data yang dibutuhkan

Buat satu baris per **tanaman/foto**, bukan per daun:

```csv
leaf_count,mean_length,mean_width,max_length,max_width,total_area,mean_ratio,label
5,99.1,80.2,187.7,136.3,35181.0,1.19,Vegetatif
```

Syarat minimal:

- Label hanya tiga kelas: `Penyemaian`, `Vegetatif`, `Siap Panen`.
- Idealnya **≥30 foto per kelas**; di bawah 10 per kelas hanya layak untuk uji
  alur, bukan untuk klaim akurasi.
- Satu tanaman tidak boleh muncul di data latih dan data uji sekaligus.
- Hanya masukkan hasil dengan `engine.used = "yolo"` dan `leaf_count > 0`.
  Mencampur fitur dari segmentasi klasik akan mengubah distribusi.

## Langkah 1 — bangun CSV fitur

### Jika nama file memuat umur (HST)

Skrip akan membaca pola `day12`, `day_12`, atau `day-12`, lalu memetakan:

| HST | Label |
| --- | --- |
| 0–14 | Penyemaian |
| 15–30 | Vegetatif |
| 31–45 | Siap Panen |

```bat
.venv\Scripts\python tools\build_feature_csv.py ^
  --images dataset\train\images dataset\valid\images ^
  --out fitur-pakcoy.csv
```

### Jika label sudah ada di CSV lain

Siapkan `label.csv` berisi `file,label` (nama file harus sama), lalu:

```bat
.venv\Scripts\python tools\build_feature_csv.py ^
  --images path\ke\foto ^
  --labels label.csv ^
  --out fitur-pakcoy.csv
```

Periksa hasilnya: jumlah baris per kelas, tidak ada `leaf_count = 0`, dan
`engine` semuanya `yolo`. Skrip akan memberi peringatan bila ada kelas yang
terlalu sedikit. Berkas yang namanya memuat `thumb` dilewati otomatis supaya
thumbnail tidak menduplikasi foto asli. Tambahkan `--include-classic` hanya bila
Anda memang ingin mencampur kedua mesin (tidak disarankan untuk laporan).

## Langkah 2 — latih model

Random Forest (ringan, tidak butuh TensorFlow):

```bat
.venv\Scripts\python tools\train_classifier.py --csv fitur-pakcoy.csv --model rf
```

MLP scikit-learn (tetap tanpa TensorFlow; scaler sudah dibungkus otomatis):

```bat
.venv\Scripts\python tools\train_classifier.py --csv fitur-pakcoy.csv --model mlp
```

Keduanya menulis:

- `models/best_rf.pkl` atau `models/best_mlp_sklearn.pkl`
- `models/label_encoder.pkl`

Tambahkan `--metrics-out hasil/klasifikasi-metrik.json` bila ingin menyimpan
akurasi, classification report, dan confusion matrix untuk lampiran laporan.

Contoh keluaran yang perlu Anda catat:

```text
Akurasi uji : 0.87
Cross-val   : 0.82 ± 0.05 (5 fold)
Kelas       : Penyemaian, Siap Panen, Vegetatif
```

Angka di atas hanya ilustrasi format, bukan hasil penelitian.

## Langkah 3 — verifikasi di aplikasi

Skrip training sudah mencetak verifikasi otomatis di akhir. Keluaran yang benar:

```text
Verifikasi pemuatan: kind=sklearn, label=Random Forest · baseline, is_demo=False
```

Bila yang muncul `kind=heuristic`, artefak gagal dimuat; baca catatan yang
dicetak sebelum melanjutkan.

1. **Jalankan ulang server** (classifier dibaca saat startup).
2. Buka `/api/status`. Yang harus berubah:

   | Sebelum | Sesudah |
   | --- | --- |
   | `classifier.kind = "heuristic"` | `classifier.kind = "sklearn"` |
   | `classifier.is_demo = true` | `classifier.is_demo = false` |
   | `classifier.model_file = null` | `classifier.model_file = "best_rf.pkl"` |

3. Analisis satu foto sampel. `prediction.model` harus menyebut Random Forest
   atau MLP, dan `prediction.is_demo` bernilai `false`.
4. Periksa `prediction.low_confidence`. Nilai `true` berarti keyakinan model di
   bawah 0,55; jangan menyembunyikan informasi ini di laporan.
5. Jika hasilnya aneh, cek urutan fitur dan pastikan training memakai
   `aggregated_raw`, bukan nilai yang sudah dikonversi ke cm.

## Langkah 4 — simpan dan deploy

```bat
git add models/best_rf.pkl models/label_encoder.pkl
git add models/model_info.json   :: bila file ini ada
git commit -m "Tambahkan classifier fase hasil training"
git push origin main
```

`.gitignore` sudah memberi pengecualian untuk artefak classifier final, jadi
perintah di atas tidak perlu `-f`. Untuk Hugging Face, push ulang ke remote
Space agar image dibangun dengan artefak baru (lihat [deploy.md](deploy.md)).

> Vercel memakai `requirements.txt` yang **tidak** memuat `scikit-learn`, jadi
> classifier tidak aktif di Vercel meskipun artefaknya ada. Gunakan Hugging Face
> atau Docker untuk demo yang memakai classifier.

## Opsi TFLite (lanjutan)

TFLite berguna bila ingin menjalankan MLP tanpa scikit-learn, tetapi menambah
ukuran image (±600 MB untuk TensorFlow) dan memperlambat build HF. Langkah
umumnya:

1. Latih MLP di Keras dengan input `(7,)` dan output softmax 3 kelas.
2. Simpan scaler yang dipakai saat training sebagai `feature_scaler.pkl`.
3. Simpan `LabelEncoder` sebagai `label_encoder.pkl`.
4. Konversi model:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
open("models/pakcoy_mlp.tflite", "wb").write(converter.convert())
```

5. Pastikan urutan fitur dan urutan kelas sama dengan training.
6. Jalankan ulang server dan cek `/api/status`: `classifier.kind` harus
   `"tflite"`.

Selama `TensorFlow` belum terpasang, aplikasi akan menulis catatan di
`classifier.notes` lalu tetap memakai scikit-learn atau heuristik.

## Pemecahan masalah classifier

| Gejala | Penyebab umum | Solusi |
| --- | --- | --- |
| `kind` tetap `heuristic` | Nama berkas salah atau server belum dijalankan ulang | Gunakan nama persis di tabel, restart server. |
| `Gagal memuat model scikit-learn` | Disimpan dengan joblib, atau objek tidak bisa di-pickle | Simpan ulang dengan `pickle.dump`. |
| Semua prediksi jatuh ke satu kelas | Data tidak seimbang atau fitur tidak informatif | Tambah data, cek `class_weight`, periksa korelasi fitur. |
| Akurasi uji terlalu tinggi (mis. 1.0) | Kebocoran data: foto tanaman yang sama di train dan test | Bagi data per tanaman/hari, bukan per frame. |
| `Label kelas tidak dapat dipetakan` | `label_encoder.pkl` tidak ada dan model tidak menyimpan label string | Sertakan `label_encoder.pkl`. |
| Hasil berbeda jauh dari saat training | Fitur cm/piksel tercampur, atau memakai segmentasi klasik | Latih dan inferensi sama-sama memakai `aggregated_raw` dari YOLO. |
