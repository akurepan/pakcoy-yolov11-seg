"""Aturan rekomendasi budidaya pakcoy hidroponik."""

from __future__ import annotations

from typing import Any

from .config import PHASE_TO_AGE


def _item(kind: str, title: str, detail: str) -> dict[str, str]:
    return {"type": kind, "title": title, "detail": detail}


PHASE_ACTIONS: dict[str, list[dict[str, str]]] = {
    "Penyemaian": [
        _item(
            "action",
            "Jaga kelembapan media semai",
            "Pertahankan rockwool lembap tetapi tidak tergenang. Air berlebih "
            "menghambat aerasi akar dan memicu busuk batang.",
        ),
        _item(
            "action",
            "Berikan nutrisi ringan",
            "Gunakan EC sekitar 0,6–1,0 mS/cm dan pH 5,8–6,2. Naikkan dosis "
            "secara bertahap saat daun baru mulai tumbuh.",
        ),
        _item(
            "action",
            "Cukupi cahaya tanpa panas berlebih",
            "Targetkan 12–16 jam cahaya per hari. Jaga jarak lampu agar suhu "
            "daun tidak terlalu tinggi.",
        ),
    ],
    "Vegetatif": [
        _item(
            "action",
            "Naikkan EC secara bertahap",
            "Arahkan EC 1,4–2,0 mS/cm dan pH 5,5–6,5. Perubahan drastis dapat "
            "membuat daun menggulung atau tepi daun terbakar.",
        ),
        _item(
            "action",
            "Pantau pH dan EC harian",
            "Catat nilai pagi dan sore. Selisih EC yang menurun cepat menandakan "
            "serapan nutrisi tinggi atau volume larutan kurang.",
        ),
        _item(
            "action",
            "Jaga sanitasi dan sirkulasi",
            "Bersihkan gulma dan sisa daun, serta pastikan aliran udara antar "
            "tanaman cukup untuk menekan penyakit daun.",
        ),
    ],
    "Siap Panen": [
        _item(
            "action",
            "Rencanakan panen pada pagi hari",
            "Panen saat suhu masih rendah untuk menjaga kesegaran dan kadar air "
            "daun. Rentang panen umum 31–45 HST.",
        ),
        _item(
            "action",
            "Bilas dengan air bersih sebelum panen",
            "Ganti larutan nutrisi dengan air bersih 1–2 hari sebelum panen untuk "
            "mengurangi residu nutrisi pada daun.",
        ),
        _item(
            "action",
            "Lakukan panen selektif",
            "Ambil daun luar yang sudah besar lebih dulu agar daun muda di bagian "
            "tengah tetap dapat tumbuh.",
        ),
    ],
}


def build_recommendations(
    phase: str | None,
    features: dict[str, float],
    context: dict[str, float],
    confidence: float,
    is_demo: bool,
    leaf_count: int,
) -> dict[str, Any]:
    items: list[dict[str, str]] = []

    if phase is None:
        items.append(
            _item(
                "warning",
                "Fase belum terdeteksi",
                "Tidak ada instance daun yang valid. Foto ulang dengan satu tanaman, "
                "latar kontras, dan pencahayaan merata.",
            )
        )
        summary = (
            "Fase pertumbuhan belum dapat ditentukan karena tidak ada daun yang "
            "terdeteksi pada foto."
        )
    else:
        actions = PHASE_ACTIONS.get(phase, [])
        items.extend(actions)
        age = PHASE_TO_AGE.get(phase, "-")
        focus = actions[0]["title"].lower() if actions else "pemantauan rutin"
        summary = (
            f"Tanaman terprediksi berada pada fase {phase} ({age}). "
            f"Fokus perawatan: {focus}."
        )

    total_area_ratio = float(context.get("total_area_ratio", 0.0))
    mean_length_ratio = float(context.get("mean_length_ratio", 0.0))
    mean_ratio = float(features.get("mean_ratio", 0.0))

    if leaf_count and leaf_count < 4:
        items.append(
            _item(
                "warning",
                "Jumlah daun masih sedikit",
                "Cek kecukupan cahaya, kepadatan tanam, dan kesehatan akar. "
                "Pastikan seluruh tajuk masuk ke dalam bingkai foto.",
            )
        )
    if leaf_count and total_area_ratio < 0.02:
        items.append(
            _item(
                "warning",
                "Tutupan daun rendah",
                "Luas daun relatif kecil terhadap bingkai. Periksa konsentrasi "
                "nutrisi, pH, dan suplai oksigen ke zona akar.",
            )
        )
    if leaf_count and mean_ratio > 2.6:
        items.append(
            _item(
                "warning",
                "Daun cenderung memanjang",
                "Rasio panjang–lebar tinggi dapat menandakan cahaya kurang merata "
                "atau tanaman mencari cahaya. Evaluasi jarak lampu dan posisi tanaman.",
            )
        )
    if leaf_count and mean_length_ratio > 0.45:
        items.append(
            _item(
                "info",
                "Daun relatif besar dalam bingkai",
                "Pastikan foto diambil dari jarak konsisten agar perbandingan antar "
                "pengamatan tetap valid.",
            )
        )
    if leaf_count >= 24:
        items.append(
            _item(
                "info",
                "Banyak instance terdeteksi",
                "Foto mungkin memuat lebih dari satu tanaman. Agregasi fitur "
                "diasumsikan untuk satu tanaman, jadi gunakan foto per tanaman "
                "untuk hasil yang lebih akurat.",
            )
        )
    if phase is not None and confidence < 0.55:
        items.append(
            _item(
                "warning",
                "Keyakinan prediksi rendah",
                "Gunakan foto dengan latar polos, satu tanaman, dan pencahayaan "
                "merata. Hindari bayangan tajam atau daun yang saling menutup.",
            )
        )
    if is_demo:
        items.append(
            _item(
                "info",
                "Klasifikasi fase masih demo",
                "Prediksi fase memakai aturan heuristik, bukan model MLP/Random "
                "Forest terlatih. Untuk hasil penelitian, letakkan artefak classifier "
                "di folder models/.",
            )
        )

    note = (
        "Saran dihasilkan dari aturan pakar dan fitur citra, bukan model MLP "
        "terlatih."
        if is_demo
        else "Saran menggabungkan prediksi model terlatih dengan aturan pakar."
    )

    return {"summary": summary, "items": items, "note": note}
