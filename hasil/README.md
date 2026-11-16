# Folder hasil

Isi folder: `pakcoy_artifacts.zip` (bobot + berkas pendukung) dan
`checkpoint-skripsi-pakcoy.ipynb` (notebook training/evaluasi).

Provenance, isi zip, dan metrik validasi/testing ada di
[docs/artefak.md](../docs/artefak.md).

Verifikasi integritas zip:

```bat
.venv\Scripts\python -c "import zipfile; print(zipfile.ZipFile('hasil/pakcoy_artifacts.zip').testzip())"
```
