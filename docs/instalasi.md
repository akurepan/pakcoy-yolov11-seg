# Instalasi dan menjalankan lokal

## Prasyarat

- **Python 3.12** (diuji pada 3.12.8).
- Windows untuk `run.bat`; di Linux/macOS gunakan perintah `python` yang setara.
- Koneksi internet saat pertama kali memasang dependensi (±1,5 GB bila memakai
  `requirements-ml.txt`, karena `torch` ikut terpasang).
- Bobot model **sudah disertakan** di `models/best.pt`; tidak perlu mengunduh
  manual.

## Langkah

```bat
git clone https://github.com/akurepan/pakcoy-yolov11-seg.git
cd pakcoy-yolov11-seg

py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-ml.txt   :: untuk mode YOLO
run.bat
```

`requirements.txt` berisi dependensi inti (FastAPI, OpenCV headless, NumPy).
`requirements-ml.txt` menambahkan `scikit-learn` dan `ultralytics`; tanpa
dependensi ini aplikasi tetap jalan dalam mode klasik.

`run.bat` membuka `http://127.0.0.1:8000`. Tanpa `run.bat`:

```bat
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Memastikan mode YOLO aktif

Buka `http://127.0.0.1:8000`, atau periksa langsung:

```bat
curl http://127.0.0.1:8000/api/status
```

Yang perlu dilihat:

- `yolo.available: true`
- `yolo.weights: "best.pt"`
- `classifier.is_demo: true` selama artefak klasifikasi belum ada (lihat
  [klasifikasi.md](klasifikasi.md))

Di halaman, badge **Mode demo** akan hilang bila YOLO aktif. Bobot yang diganti
saat server berjalan terdeteksi otomatis, tetapi artefak **classifier** baru
terbaca setelah server dijalankan ulang.

## Menjalankan pengujian

```bat
.venv\Scripts\python -m unittest discover -s tests -v
```

Uji memakai `unittest` bawaan Python, tanpa dependensi tambahan. Cakupannya:

- preprocessing, segmentasi klasik, pengukuran, dan agregasi fitur,
- prediksi, kalibrasi satuan, dan rekomendasi tanpa fase,
- integritas artefak: zip dapat diekstrak dan `best.pt` di dalamnya identik
  dengan `models/best.pt`.

Dua uji yang memerlukan bobot akan **di-skip** bila `models/best.pt` tidak ada.
Di repositori ini bobot sudah disertakan, jadi seluruh uji berjalan (13 uji).

> Pesan Ultralytics ditulis ke stderr sehingga PowerShell bisa menampilkan teks
> merah dan `Exited with code 1`. Selama ringkasan berakhir `OK`, pengujian
> lulus. Untuk memastikan, jalankan `echo $LASTEXITCODE` setelah perintah:
> nilainya `0`.

## Pemecahan masalah lokal

| Gejala | Penyebab umum | Solusi |
| --- | --- | --- |
| `ModuleNotFoundError` | Perintah dijalankan di luar `.venv` | Gunakan `.venv\Scripts\python`. |
| Header menampilkan **Mode demo** | `models/best.pt` tidak ada atau `ultralytics` belum terpasang | Pastikan bobot ada dan jalankan `pip install -r requirements-ml.txt`, lalu muat ulang halaman. |
| "Bobot ditemukan, tetapi ultralytics belum terpasang" | Dependensi ML belum diinstal | `pip install -r requirements-ml.txt`. |
| Prediksi memakai "Aturan heuristik (demo)" | Artefak classifier belum ada | Ikuti [klasifikasi.md](klasifikasi.md). |
| YOLO tidak menemukan daun | Foto terlalu dekat, daun menutup, atau skala berbeda dari data latih | Jauhkan kamera agar satu tanaman utuh masuk bingkai, pakai latar kontras. |
| Tidak ada daun terdeteksi | Latar ramai, cahaya kurang, objek terlalu kecil | Gunakan latar kontras, foto lebih dekat, atau naikkan confidence. |
| Port 8000 sudah dipakai | Server lain masih berjalan | Hentikan proses lama, atau jalankan `uvicorn ... --port 8010`. |
| Respons 413 saat unggah | Berkas lebih dari 20 MB | Perkecil foto; batas ada di `app/config.py`. |
