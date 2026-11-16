# Arah desain Pakcoy Vision

Dokumen ini mencatat arah visual yang dipakai halaman analisis pakcoy. Isinya
alasan keputusan, bukan daftar larangan. Rujukan aturan tetap `antislop`.

## Design Read

> Reading this as: alat analisis citra pertanian untuk mahasiswa dan peneliti
> pakcoy hidroponik, dengan bahasa visual laboratorium lapangan yang tenang,
> dial ENERGY 2 / RHYTHM 2 / MOTION 2.

- **ENERGY 2**: hijau tua dan lime memberi karakter, tetapi halaman tetap
  terbaca sebagai alat kerja, bukan kampanye pemasaran.
- **RHYTHM 2**: tiga komposisi berbeda (hero dua kolom, ruang kerja dua panel,
  detail bertingkat), dengan jarak bagian yang konsisten.
- **MOTION 2**: sapuan garis scan kontinu pada ilustrasi hero, gerakan daun
  yang sangat halus, dan chip yang mengambang pelan. Tidak ada parallax atau
  koreografi. Saat analisis berjalan, sapuan scan dipercepat sebagai indikator
  proses, dan `prefers-reduced-motion` menghentikan semua gerakan.

## Palet

| Token | Nilai | Alasan |
| --- | --- | --- |
| `--green-deep` | `#0f3d28` | Warna daun tua untuk teks utama dan tombol aksi; kontras 12,2:1 di atas putih. |
| `--green` | `#1f7a4d` | Warna daun untuk aksen, fokus, dan status siap. |
| `--lime` | `#c9f06a` | Satu aksen terang untuk penanda pindaian dan sorotan kata kunci. |
| `--amber` | `#b26a00` | Semantik peringatan dan fase Siap Panen, bukan dekorasi. |
| `--blue` | `#2563eb` | Semantik fase Penyemaian dan ikon informasi. |
| `--red` | `#b3261e` | Semantik galat saja. |

Netral putih, krem, dan abu kehijauan tidak dihitung sebagai warna inti. Tidak
ada gradien latar halaman. Gradien hanya dipakai untuk fungsi: bingkai ilustrasi
hero, sorotan kata kunci, dan isi bilah progres.

## Tipografi

`Segoe UI Variable Text` dengan fallback system-ui. Alasan: aplikasi berjalan
lokal di Windows, jadi font sistem memberi render tajam, beban unduh nol, dan
tetap netral untuk angka tabel. Tidak ada font mono dekoratif atau label
uppercase bertracking lebar pada judul bagian. Label metadata kecil (statistik
hero, kicker fase, header tabel) memakai uppercase dengan tracking 0,07-0,12em
agar terpisah dari nilai angka, bukan sebagai hiasan judul.

## Bentuk dan elevasi

- Radius mengikuti hierarki: kartu 24 px, tombol 12 px, badge 8 px, tab 7 px.
  Tidak semua elemen dibuat pil.
- Bayangan hanya dipakai pada bingkai ilustrasi hero dan chip melayang. Kartu
  memakai garis tipis, bukan bayangan, agar halaman tidak terasa melayang.
- Ikon yang dipakai adalah ikon fungsional (unggah, peringatan, informasi,
  centang) dengan `aria-hidden` ketika teksnya sudah menjelaskan aksi.

## Data dan kejujuran

- Angka model di hero (arsitektur, mAP50 mask, kelas) diambil dari checkpoint
  `models/best.pt` melalui `/api/status`. Sebelum status dimuat, nilainya `-`.
- Tidak ada testimoni, logo mitra, atau statistik pengguna.
- Klasifikasi fase masih heuristik dan selalu diberi label "Aturan heuristik
  (demo)" di antarmuka.
- Sampel foto berasal dari dataset penelitian, bukan gambar stok.

## Aksesibilitas

- Kontras teks terendah 5,1:1 (abu `#5c6c61` di atas latar krem).
- Target sentuh minimal 44 px untuk tombol, tab, ringkasan pengaturan, dan tautan
  footer.
- Fokus keyboard memakai outline hijau 2 px dengan offset 3 px.
- `prefers-reduced-motion` mematikan animasi dan smooth scroll.
- Tema terang dipilih karena alat dipakai di rumah kaca dan layar laptop siang
  hari; tidak ada toggle tema yang setengah jadi.

## Batas yang disengaja

Ilustrasi hero adalah gambar vektor daun dengan bingkai ukur dan garis pindai.
Ia tidak menampilkan angka palsu atau tangkapan layar produk yang tidak ada.
