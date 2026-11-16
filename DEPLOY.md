# Deploy Pakcoy Vision

Panduan langkah demi langkah untuk membuat aplikasi bisa diakses publik.
Status per 16 November 2026: **belum ada deployment aktif**, jadi belum ada URL
publik yang bisa dibagikan. Berkas konfigurasi sudah siap.

| Jalur | Mode | Bobot YOLO | Kapan dipakai |
| --- | --- | --- | --- |
| Hugging Face Spaces (Docker) | YOLOv11-seg + fallback klasik | Ya | Untuk demo hasil penelitian |
| Vercel | Klasik saja (demo UI) | Tidak | Untuk memastikan antarmuka jalan |
| Docker di VPS | YOLOv11-seg | Ya | Bila punya server sendiri |

Bobot `models/best.pt` sudah ikut repositori, jadi ketiga jalur tidak perlu
langkah `git add -f` lagi.

## 1. Hugging Face Spaces — mode YOLOv11-seg

### Lewat web

1. Buka <https://huggingface.co/new-space>.
2. Isi nama (mis. `pakcoy-vision`), pilih SDK **Docker**, visibility **Public**.
3. Tambahkan remote dan push:

```bat
git remote add hf https://huggingface.co/spaces/<username>/pakcoy-vision
git push hf main
```

Saat diminta password, pakai **token Hugging Face** (Settings → Access Tokens),
bukan password akun.

### Lewat CLI

```bat
.venv\Scripts\python -m pip install -U huggingface_hub
.venv\Scripts\hf auth login
.venv\Scripts\hf repo create pakcoy-vision --repo-type space --space-sdk docker --public
git remote add hf https://huggingface.co/spaces/<username>/pakcoy-vision
git push hf main
```

Front matter di awal `README.md` (`sdk: docker`, `app_port: 7860`) sudah dibaca
otomatis oleh Spaces. Build pertama sekitar 5–15 menit karena memasang torch.
Space gratis tidur setelah lama tidak diakses; kunjungan pertama butuh
30–60 detik untuk bangun.

URL hasil: `https://<username>-pakcoy-vision.hf.space`.

## 2. Vercel — hanya antarmuka mode demo

```bat
npx vercel --prod
```

Jawab: Root Directory `./`, framework preset **FastAPI**, tanpa environment
variable. Batas 250 MB per fungsi membuat `torch`/`ultralytics` tidak mungkin
ikut; `.vercelignore` juga membuang `models/*.pt`.

Yang perlu dicek setelah deploy:

- `/api/health` harus `200`.
- `/api/status` akan menampilkan `yolo.available = false` dan
  `classifier.is_demo = true`. Itu benar untuk Vercel, bukan bug.
- `/static/app.js` harus `200`. `vercel.json` mengarahkan semua request ke
  `/api`; bila aset ikut ter-rewrite, halaman akan tampil tanpa CSS/JS.
- Bila semua route mengembalikan `404`, periksa apakah fungsi menerima path
  asli (`/api/analyze`) atau path hasil rewrite (`/api`). Sesuaikan
  `vercel.json` bila perlu, lalu deploy ulang.

## 3. Docker / VPS

```bat
docker build -t pakcoy-vision .
docker run --rm -p 7860:7860 pakcoy-vision
```

Buka `http://127.0.0.1:7860`. Untuk server publik, letakkan di belakang reverse
proxy (Caddy/Nginx) yang menangani TLS, lalu teruskan ke port 7860. Batas unggah
20 MB di aplikasi perlu diselaraskan dengan batas proxy.

## 4. Checklist verifikasi

Jalankan setelah setiap deploy:

```bat
curl https://<domain>/api/health
curl https://<domain>/api/status
curl -X POST https://<domain>/api/analyze -F "file=@app/static/samples/day-20.jpg"
```

- [ ] `/api/health` mengembalikan `{"status":"ok",...}`.
- [ ] `/api/status` memuat `app.version` dan daftar sampel.
- [ ] Di HF/Docker: `yolo.available` bernilai `true` dan respons analisis
      memakai `engine.used = "yolo"`.
- [ ] Di Vercel: `yolo.available` bernilai `false` dan halaman menampilkan
      badge **Mode demo**.
- [ ] `classifier.is_demo` masih `true` selama artefak classifier belum ada.
      Jangan mengubah label ini sebelum model benar-benar tersedia.
- [ ] Aset `/static/styles.css` dan `/static/app.js` merespons `200`.
- [ ] URL deploy dicatat di `README.md` dan kolom **Website** repositori GitHub.

## Pemecahan masalah deploy

| Gejala | Penyebab umum | Solusi |
| --- | --- | --- |
| Build HF gagal saat `pip install` | Versi wheel tidak tersedia untuk platform | Cek log build, samakan versi di `requirements*.txt`. |
| HF Space menampilkan 503 saat dibuka | Space sedang tidur atau build belum selesai | Tunggu, buka ulang; cek tab **Logs**. |
| Semua route Vercel 404 | Rewrite meneruskan path yang salah atau `api/index.py` tidak terdeteksi | Uji `/api/health`; sesuaikan `rewrites`; pastikan `api/index.py` ada. |
| Halaman tanpa CSS/JS di Vercel | Request aset ikut di-rewrite ke fungsi | Kecualikan `/static/*` dari rewrite. |
| Respons Vercel 500 `No module named ...` | Dependensi tidak ada di `requirements.txt` | Tambahkan dependensi, tapi ingat batas 250 MB. |
| Analisis HF jatuh ke mode klasik | `models/best.pt` tidak ikut terkirim | Pastikan berkas ada di commit (`git ls-files models/`). |
