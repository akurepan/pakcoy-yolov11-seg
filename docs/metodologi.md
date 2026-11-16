# Catatan metodologis dan batasan

Bagian ini penting untuk penulisan laporan dan sidang. Tujuannya mencegah klaim
yang melebihi bukti yang tersedia.

## Yang boleh diklaim

- Segmentasi instance daun memakai **YOLOv11n-seg** hasil training sendiri:
  satu kelas `daun-pakcoy`, `imgsz` 640, 100 epoch.
- Metrik validasi dari checkpoint: **mAP50 box 0,956** dan **mAP50 mask 0,953**
  (rincian di [artefak.md](artefak.md)).
- Pengukuran bentuk (panjang, lebar, luas, rasio) dihitung dari mask instance
  dengan OpenCV, bukan dari bounding box.
- Estimasi usia dan rekomendasi berasal dari **aturan pakar** per fase, bukan
  model prediktif.

## Yang belum boleh diklaim

- **Fase pertumbuhan sebagai keluaran model terlatih.** Classifier fase belum
  ada; selama `prediction.is_demo = true`, hasilnya aturan heuristik. Ikuti
  [klasifikasi.md](klasifikasi.md) untuk melatih dan menggantinya.
- **Akurasi klasifikasi fase** dalam bentuk apa pun, termasuk angka dari contoh
  `model_info.json` di dokumentasi. Angka contoh bukan hasil penelitian.
- **Kurva PR / confusion matrix** yang hilang dari zip lama. Jangan menggambar
  ulang secara manual lalu menyajikannya sebagai keluaran training.
- **Kalibrasi cm** sebelum piksel/cm diukur sendiri dengan penggaris.

## Asumsi pengukuran

- Foto memuat **satu tanaman**. Banyak tanaman dalam satu foto membuat agregasi
  fitur tidak lagi mewakili satu individu.
- Fitur klasifikasi selalu dihitung dalam piksel. Kalibrasi piksel/cm hanya
  memengaruhi satuan yang ditampilkan, agar distribusi fitur tetap konsisten
  dengan data latih.
- Kalibrasi tidak diisi otomatis. `DEFAULT_PIXELS_PER_CM` di `app/config.py`
  (bisa diganti lewat environment variable `PAKCOY_PIXELS_PER_CM`) hanya dipakai
  sebagai contoh di placeholder. Nilai `38,5` px/cm **belum tervalidasi**.
- Daun yang saling menutup rapat dapat menyatu atau terpecah saat segmentasi.
  Verifikasi visual tetap diperlukan.
- Lebar daun adalah lebar proyeksi tegak lurus sumbu daun. Bila daun terpotong
  objek lurus seperti penggaris, piksel yang tertutup tidak ada di citra. Untuk
  daun seperti ini, lebar direkonstruksi dengan asumsi simetri terhadap garis
  penghalang dan ditandai `≈` pada tabel. Panjang dan luas tetap dihitung dari
  piksel yang terlihat, jadi keduanya bisa lebih kecil dari ukuran asli.

## Batas model yang teramati

- **Foto close-up ekstrem gagal dideteksi.** `app/static/samples/day-42.jpg`
  menghasilkan 0 instance bahkan pada `conf=0.05`, sehingga `engine=auto` turun
  ke segmentasi klasik. Penyebabnya domain gap: daun memenuhi bingkai, saling
  menutup, dan latar berbeda dari data latih. Menurunkan ambang tidak menolong;
  perbaikannya adalah menambah data hari 35–45 dengan variasi jarak dan sudut.
- **Gambar abstrak dapat menghasilkan deteksi palsu.** Pada citra noise,
  segmentasi klasik bisa menganggap banyak region sebagai daun dan menyentuh
  batas 40 instance. Selalu periksa `warnings` dan citra anotasi.
- **Fallback menyembunyikan kegagalan YOLO.** Bila YOLO menemukan 0 daun,
  aplikasi otomatis memakai mode klasik dan menulis peringatan. Untuk klaim
  hasil penelitian, pastikan `engine.used` bernilai `yolo`.

## Rekomendasi pelaporan

1. Laporkan `engine.used` dan `prediction.source` untuk setiap hasil yang
   dimasukkan ke laporan.
2. Cantumkan versi Ultralytics dan tanggal training dari `/api/status`.
3. Simpan CSV/JSON mentah hasil analisis sebagai lampiran, bukan hanya tangkapan
   layar.
4. Jika classifier fase nanti dilatih, laporkan pembagian data (train/valid/test)
   dan pastikan tidak ada tanaman yang sama muncul di lebih dari satu split.
