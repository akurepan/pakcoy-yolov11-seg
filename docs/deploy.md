# Deploy Pakcoy Vision

Panduan langkah demi langkah untuk membuat aplikasi bisa diakses publik.
Status per 16 November 2026: **belum ada deployment aktif**, jadi belum ada URL
publik yang bisa dibagikan. Berkas konfigurasi sudah siap.

| Jalur | Mode | Bobot YOLO | Klasifikasi fase | Kapan dipakai |
| --- | --- | --- | --- | --- |
| Hugging Face Spaces (Docker) | YOLOv11-seg + fallback klasik | Ya | Bisa (scikit-learn) | Untuk demo hasil penelitian |
| Vercel | Klasik saja (demo UI) | Tidak | Tidak | Untuk memastikan antarmuka jalan |
| Docker di VPS | YOLOv11-seg | Ya | Bisa | Bila punya server sendiri |

Bobot `models/best.pt` sudah ikut repositori, jadi tidak perlu `git add -f`.
Artefak classifier (bila sudah dilatih) juga ikut setelah ditambahkan ke git;
lihat [klasifikasi.md](klasifikasi.md).

## 1. Hugging Face Spaces — mode YOLOv11-seg

### 1.1 Buat akun dan token

1. Buat akun di <https://huggingface.co/join> bila belum punya, lalu verifikasi
   email.
2. Buka <https://huggingface.co/settings/tokens>.
3. Klik **Create new token**. Pilih tipe **Write** (atau **Fine-grained** dengan
   izin menulis pada Space yang akan dibuat).
4. Beri nama, misalnya `pakcoy-deploy`, lalu klik **Create token**.
5. Salin token sekali saja (formatnya `hf_...`). Token tidak bisa dilihat lagi
   setelah halaman ditutup.

> **Jangan menempelkan token di chat, WhatsApp, atau commit.** Simpan di
> password manager. Bila token bocor, langsung **Revoke** di halaman yang sama.

Cara paling aman bila ingin saya (agen) yang men-deploy: jalankan login di
komputer ini, lalu cukup bilang "sudah login". Token tersimpan lokal dan tidak
pernah ditampilkan di percakapan:

```bat
.venv\Scripts\python -m pip install -U huggingface_hub
.venv\Scripts\hf auth login
.venv\Scripts\hf whoami
```

`hf auth login` akan meminta token tanpa menampilkannya di layar.

### 1.2 Buat Space

Lewat web: buka <https://huggingface.co/new-space>, isi nama misalnya
`pakcoy-vision`, pilih SDK **Docker**, visibility **Public**, lalu **Create**.

Atau lewat CLI:

```bat
.venv\Scripts\hf repo create pakcoy-vision --repo-type space --space-sdk docker --public
```

### 1.3 Push repositori

```bat
git remote add hf https://huggingface.co/spaces/<username>/pakcoy-vision
git -c credential.helper= push hf main
```

Saat diminta:

- **Username**: username Hugging Face (bukan email).
- **Password**: token `hf_...` yang tadi dibuat.

Opsi `-c credential.helper=` mencegah Windows Credential Manager mengirim
kredensial GitHub yang tersimpan untuk `github.com`, yang sering membuat push
gagal dengan `Authentication failed`.

### 1.4 Tunggu build dan verifikasi

Build pertama 5–15 menit karena memasang `torch`. Pantau tab **Logs** di Space.
URL hasil: `https://<username>-pakcoy-vision.hf.space`.

```bat
curl https://<username>-pakcoy-vision.hf.space/api/health
curl https://<username>-pakcoy-vision.hf.space/api/status
```

`yolo.available` harus `true`. Bila `false`, lihat baris terakhir tabel
pemecahan masalah di bawah.

Space gratis tidur setelah lama tidak diakses; kunjungan pertama butuh
30–60 detik untuk bangun.

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

Classifier fase tidak aktif di Vercel karena `requirements.txt` tidak memuat
`scikit-learn`. Itu pilihan sadar untuk menjaga ukuran fungsi.

> **Batas unggahan Vercel 4,5 MB.** Body request di atas ±4,5 MB ditolak Vercel
> sebelum fungsi berjalan, dengan respons `413 FUNCTION_PAYLOAD_TOO_LARGE`
> berupa teks biasa (bukan JSON), sehingga halaman hanya menampilkan status
> error generik. Batas 20 MB di `app/config.py` hanya berlaku di HF/Docker.
> Bila memakai Vercel, perkecil foto di sisi browser sebelum mengirim.

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
- [ ] `classifier.is_demo` bernilai `true` hanya selama artefak classifier belum
      ada. Jangan mengubah label ini sebelum model benar-benar tersedia.
- [ ] Aset `/static/styles.css` dan `/static/app.js` merespons `200`.
- [ ] URL deploy dicatat di [README](../README.md) dan kolom **Website**
      repositori GitHub.

## Pemecahan masalah deploy

| Gejala | Penyebab umum | Solusi |
| --- | --- | --- |
| Build HF gagal saat `pip install` | Versi wheel tidak tersedia untuk platform | Cek log build, samakan versi di `requirements*.txt`. |
| HF Space menampilkan 503 saat dibuka | Space sedang tidur atau build belum selesai | Tunggu, buka ulang; cek tab **Logs**. |
| `Authentication failed` saat push | Credential Manager mengirim akun GitHub | Gunakan `git -c credential.helper= push hf main`, lalu isi token HF. |
| Push ditolak `permission denied` | Token tidak punya izin tulis | Buat token tipe **Write** atau fine-grained dengan akses tulis ke Space. |
| Semua route Vercel 404 | Rewrite meneruskan path yang salah atau `api/index.py` tidak terdeteksi | Uji `/api/health`; sesuaikan `rewrites`; pastikan `api/index.py` ada. |
| Halaman tanpa CSS/JS di Vercel | Request aset ikut di-rewrite ke fungsi | Kecualikan `/static/*` dari rewrite. |
| Respons Vercel 500 `No module named ...` | Dependensi tidak ada di `requirements.txt` | Tambahkan dependensi, tapi ingat batas 250 MB. |
| Analisis HF jatuh ke mode klasik | `models/best.pt` tidak ikut terkirim | Pastikan berkas ada di commit (`git ls-files models/`). |
