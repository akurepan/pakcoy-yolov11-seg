"use strict";

const PHASES = ["Penyemaian", "Vegetatif", "Siap Panen"];
const PHASE_CLASS = {
  Penyemaian: "phase-penyemaian",
  Vegetatif: "phase-vegetatif",
  "Siap Panen": "phase-siap-panen",
};
const MAX_UPLOAD_MB = 20;
const RING_CIRCUMFERENCE = 2 * Math.PI * 52;

const FALLBACK_SAMPLES = [
  {
    id: "day-05",
    file: "day-05.jpg",
    label: "Hari 5",
    hint: "Penyemaian",
    url: "/static/samples/day-05.jpg",
    thumb: "/static/samples/day-05-thumb.jpg",
  },
  {
    id: "day-20",
    file: "day-20.jpg",
    label: "Hari 20",
    hint: "Vegetatif",
    url: "/static/samples/day-20.jpg",
    thumb: "/static/samples/day-20-thumb.jpg",
  },
  {
    id: "day-30",
    file: "day-30.jpg",
    label: "Hari 30",
    hint: "Vegetatif akhir",
    url: "/static/samples/day-30.jpg",
    thumb: "/static/samples/day-30-thumb.jpg",
  },
  {
    id: "day-42",
    file: "day-42.jpg",
    label: "Hari 42",
    hint: "Siap panen",
    url: "/static/samples/day-42.jpg",
    thumb: "/static/samples/day-42-thumb.jpg",
  },
];

const LOADING_STAGES = [
  "Preprocessing citra",
  "Segmentasi instance daun",
  "Ekstraksi fitur OpenCV",
  "Klasifikasi fase",
  "Menyusun rekomendasi",
];

const ICONS = {
  action:
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m4 10.5 4 4 8-9" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  warning:
    '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 2.5 18 17H2L10 2.5Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M10 7v4.5M10 14.2v.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  info:
    '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M10 9v5M10 6.4v.1" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
};

const state = {
  status: null,
  file: null,
  previewUrl: null,
  result: null,
  activeView: "annotated",
  busy: false,
  loadingTimer: null,
};

const formatterCache = new Map();

function num(value, digits = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  const key = String(digits);
  if (!formatterCache.has(key)) {
    formatterCache.set(
      key,
      new Intl.NumberFormat("id-ID", { maximumFractionDigits: digits })
    );
  }
  return formatterCache.get(key).format(numeric);
}

function decimal(value, digits = 3) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return new Intl.NumberFormat("id-ID", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(numeric);
}

const pctFormatter = new Intl.NumberFormat("id-ID", {
  style: "percent",
  maximumFractionDigits: 0,
});

const $ = (id) => document.getElementById(id);

const els = {
  headerStatus: $("headerStatus"),
  statusText: $("statusText"),
  heroModel: $("heroModel"),
  heroMap: $("heroMap"),
  heroClasses: $("heroClasses"),
  fileInput: $("fileInput"),
  dropzone: $("dropzone"),
  dropEmpty: $("dropEmpty"),
  dropPreview: $("dropPreview"),
  previewImage: $("previewImage"),
  analyzeBtn: $("analyzeBtn"),
  uploadError: $("uploadError"),
  confInput: $("confInput"),
  confValue: $("confValue"),
  calibrationInput: $("calibrationInput"),
  calibrationHint: $("calibrationHint"),
  engineSelect: $("engineSelect"),
  engineBadge: $("engineBadge"),
  sampleList: $("sampleList"),
  resultCard: $("resultCard"),
  resultEmpty: $("resultEmpty"),
  resultLoading: $("resultLoading"),
  resultSummary: $("resultSummary"),
  loadingStage: $("loadingStage"),
  loadingBar: $("loadingBar"),
  phaseName: $("phaseName"),
  phaseAge: $("phaseAge"),
  phaseNote: $("phaseNote"),
  confidenceValue: $("confidenceValue"),
  confidenceTitle: $("confidenceTitle"),
  ringFg: $("ringFg"),
  probList: $("probList"),
  modelBadge: $("modelBadge"),
  engineMeta: $("engineMeta"),
  timeMeta: $("timeMeta"),
  summaryWarnings: $("summaryWarnings"),
  detailSection: $("detail"),
  detailCaption: $("detailCaption"),
  pipelineTrace: $("pipelineTrace"),
  resultImage: $("resultImage"),
  imageCaption: $("imageCaption"),
  visualPanel: $("visualPanel"),
  unitBadge: $("unitBadge"),
  aggGrid: $("aggGrid"),
  leafCountBadge: $("leafCountBadge"),
  leafTable: $("leafTable"),
  leafTableBody: $("leafTableBody"),
  leafTableNote: $("leafTableNote"),
  maskGallery: $("maskGallery"),
  thLengthUnit: $("thLengthUnit"),
  thWidthUnit: $("thWidthUnit"),
  thAreaUnit: $("thAreaUnit"),
  recPhaseBadge: $("recPhaseBadge"),
  recSummary: $("recSummary"),
  recList: $("recList"),
  recNote: $("recNote"),
  downloadCsvBtn: $("downloadCsvBtn"),
  downloadJsonBtn: $("downloadJsonBtn"),
  toast: $("toast"),
};

function init() {
  bindEvents();
  updateConfOutput();
  loadStatus();
}

function bindEvents() {
  els.fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) setFile(file);
  });

  els.dropzone.addEventListener("click", () => {
    if (!state.busy) els.fileInput.click();
  });

  ["dragenter", "dragover"].forEach((type) => {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!state.busy) els.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "dragend", "drop"].forEach((type) => {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      els.dropzone.classList.remove("dragover");
    });
  });

  els.dropzone.addEventListener("drop", (event) => {
    if (state.busy) return;
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) setFile(file, { autoAnalyze: true });
  });

  window.addEventListener("paste", (event) => {
    if (state.busy) return;
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          setFile(file);
          toast("Foto dari clipboard siap dianalisis.", "info");
          break;
        }
      }
    }
  });

  els.analyzeBtn.addEventListener("click", analyze);

  els.confInput.addEventListener("input", updateConfOutput);

  els.sampleList.addEventListener("click", (event) => {
    const button = event.target.closest(".sample-chip");
    if (!button) return;
    const samples = (state.status && state.status.samples) || FALLBACK_SAMPLES;
    const sample = samples.find((item) => item.id === button.dataset.sampleId);
    if (sample) loadSample(sample);
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setView(tab.dataset.view));
    tab.addEventListener("keydown", (event) => {
      const tabs = Array.from(document.querySelectorAll(".tab"));
      const index = tabs.indexOf(tab);
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
      if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
      if (event.key === "Home") next = tabs[0];
      if (event.key === "End") next = tabs[tabs.length - 1];
      if (next) {
        event.preventDefault();
        next.focus();
        setView(next.dataset.view);
      }
    });
  });

  els.downloadCsvBtn.addEventListener("click", downloadCsv);
  els.downloadJsonBtn.addEventListener("click", downloadJson);
}

function updateConfOutput() {
  const value = Number(els.confInput.value || 0.25);
  els.confValue.textContent = new Intl.NumberFormat("id-ID", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-cache" });
    if (!response.ok) throw new Error("Status server tidak tersedia.");
    const status = await response.json();
    state.status = status;
    renderStatus(status);
  } catch (error) {
    els.headerStatus.className = "header-status status-error";
    els.statusText.textContent = "Server tidak terjangkau";
    els.engineBadge.textContent = "Status tidak tersedia";
    renderSamples(FALLBACK_SAMPLES);
    toast("Status model tidak dapat dimuat. Pastikan server berjalan.", "error");
  }
}

function renderStatus(status) {
  const yoloReady = Boolean(status.yolo && status.yolo.available);
  const classifierDemo = Boolean(status.classifier && status.classifier.is_demo);
  const model = (status.yolo && status.yolo.model) || {};
  const map50 =
    model.metrics && model.metrics.mask ? model.metrics.mask.map50 : null;

  let headerClass = "header-status status-demo";
  let headerText = "Mode demo · segmentasi klasik";
  if (yoloReady && !classifierDemo) {
    headerClass = "header-status status-ready";
    headerText = "Pipeline terlatih siap";
  } else if (yoloReady && classifierDemo) {
    headerClass = "header-status status-ready";
    headerText = "YOLO siap · fase demo";
  }
  els.headerStatus.className = headerClass;
  els.statusText.textContent = headerText;

  const modelName = model.architecture || (status.yolo && status.yolo.weights) || "";
  els.engineBadge.textContent = yoloReady
    ? `YOLO · ${(status.yolo && status.yolo.weights) || "aktif"}`
    : "Mode demo";
  els.engineBadge.className = yoloReady ? "badge" : "badge badge-soft";
  if (yoloReady) {
    const parts = [modelName];
    if (map50 !== null && map50 !== undefined) {
      parts.push(`mAP50 mask ${decimal(map50)}`);
    }
    if (model.classes && model.classes.length) {
      parts.push(model.classes.join(", "));
    }
    els.engineBadge.title = parts.filter(Boolean).join(" · ");
  } else {
    els.engineBadge.removeAttribute("title");
  }

  if (els.heroModel) {
    els.heroModel.textContent = modelName || "-";
  }
  if (els.heroMap) {
    els.heroMap.textContent =
      map50 !== null && map50 !== undefined ? decimal(map50) : "-";
  }
  if (els.heroClasses) {
    els.heroClasses.textContent =
      model.classes && model.classes.length ? model.classes.join(", ") : "-";
  }

  const yoloOption = els.engineSelect.querySelector('option[value="yolo"]');
  if (yoloOption) {
    yoloOption.disabled = !yoloReady;
    yoloOption.textContent = yoloReady
      ? `${modelName || "YOLOv11-seg"} (model terlatih)`
      : "YOLOv11-seg (bobot belum tersedia)";
  }
  if (!yoloReady && els.engineSelect.value === "yolo") {
    els.engineSelect.value = "auto";
  }

  if (status.limits && Number.isFinite(Number(status.limits.default_conf))) {
    els.confInput.value = String(status.limits.default_conf);
    updateConfOutput();
  }

  const defaultCalibration = Number(
    status.limits && status.limits.default_pixels_per_cm,
  );
  if (defaultCalibration > 0) {
    const formatted = new Intl.NumberFormat("id-ID", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }).format(defaultCalibration);
    // Jangan isi otomatis: angka contoh belum tervalidasi dan tidak boleh
    // membuat hasil seolah sudah terkalibrasi.
    els.calibrationInput.placeholder = `mis. ${formatted}`;
    if (els.calibrationHint) {
      els.calibrationHint.textContent =
        "Kosongkan jika belum dikalibrasi, hasil akan tampil dalam piksel. " +
        `Isi hanya setelah mengukur dengan penggaris. Angka contoh ${formatted} ` +
        "px/cm belum tervalidasi. Model klasifikasi tetap memakai fitur piksel.";
    }
  }

  renderSamples(status.samples && status.samples.length ? status.samples : FALLBACK_SAMPLES);
}

function renderSamples(samples) {
  els.sampleList.replaceChildren();
  samples.forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-chip";
    button.dataset.sampleId = sample.id;
    button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-label", `Coba contoh ${sample.label} (${sample.hint})`);

    const image = document.createElement("img");
    image.className = "sample-thumb";
    image.src = sample.thumb || sample.url;
    image.alt = "";
    image.width = 56;
    image.height = 44;
    image.loading = "lazy";

    const meta = document.createElement("span");
    meta.className = "sample-meta";
    const title = document.createElement("strong");
    title.textContent = sample.label;
    const hint = document.createElement("span");
    hint.textContent = sample.hint;
    meta.append(title, hint);

    button.append(image, meta);
    els.sampleList.append(button);
  });
}

function setFile(file, options = {}) {
  if (!file || state.busy) return;

  const typeOk =
    file.type.startsWith("image/") ||
    /\.(jpe?g|png|webp)$/i.test(file.name || "");
  if (!typeOk) {
    showError("Format berkas tidak didukung. Gunakan JPG, PNG, atau WEBP.");
    return;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    showError(
      `Ukuran foto melebihi ${MAX_UPLOAD_MB} MB. Perkecil foto lalu coba lagi.`
    );
    return;
  }

  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);

  els.previewImage.onload = () => {
    els.previewImage.width = els.previewImage.naturalWidth;
    els.previewImage.height = els.previewImage.naturalHeight;
  };
  els.previewImage.src = state.previewUrl;
  els.previewImage.alt = "";
  els.dropEmpty.hidden = true;
  els.dropPreview.hidden = false;
  els.analyzeBtn.disabled = false;
  els.uploadError.hidden = true;
  els.fileInput.value = "";
  clearActiveSample();

  if (options.autoAnalyze) analyze();
}

async function loadSample(sample) {
  if (state.busy) return;
  try {
    const response = await fetch(sample.url, { cache: "no-cache" });
    if (!response.ok) throw new Error("Contoh foto gagal dimuat.");
    const blob = await response.blob();
    const file = new File([blob], sample.file || `${sample.id}.jpg`, {
      type: blob.type || "image/jpeg",
    });
    setFile(file);
    markActiveSample(sample.id);
    await analyze();
  } catch (error) {
    showError(error.message || "Contoh foto gagal dimuat.");
    toast(error.message || "Contoh foto gagal dimuat.", "error");
  }
}

function markActiveSample(sampleId) {
  document.querySelectorAll(".sample-chip").forEach((button) => {
    const active = button.dataset.sampleId === sampleId;
    button.classList.toggle("sample-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function clearActiveSample() {
  document.querySelectorAll(".sample-chip").forEach((button) => {
    button.classList.remove("sample-active");
    button.setAttribute("aria-pressed", "false");
  });
}

async function analyze() {
  if (!state.file || state.busy) return;

  state.busy = true;
  setBusy(true);
  showLoading();
  startLoadingProgress();

  const form = new FormData();
  form.append("file", state.file, state.file.name || "foto-pakcoy.jpg");
  form.append("conf", els.confInput.value || "0.25");
  form.append("iou", "0.5");
  form.append("engine", els.engineSelect.value || "auto");
  const calibration = els.calibrationInput.value.trim();
  if (calibration) form.append("pixels_per_cm", calibration.replace(",", "."));

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    let data = {};
    try {
      data = await response.json();
    } catch (parseError) {
      data = {};
    }
    if (!response.ok) {
      throw new Error(
        data.detail || `Analisis gagal (HTTP ${response.status}).`
      );
    }

    state.result = data;
    renderSummary(data);
    renderDetail(data);
    setView("annotated");
    toast(
      `Analisis selesai dalam ${num((data.timings && data.timings.total_ms) || 0, 0)} ms.`,
      "success"
    );
  } catch (error) {
    showError(error.message || "Analisis gagal.");
    showEmpty();
    toast(error.message || "Analisis gagal.", "error");
  } finally {
    stopLoadingProgress();
    state.busy = false;
    setBusy(false);
  }
}

function setBusy(busy) {
  const label = els.analyzeBtn.querySelector(".btn-label");
  const spinner = els.analyzeBtn.querySelector(".spinner");
  document.body.classList.toggle("is-analyzing", busy);
  els.analyzeBtn.disabled = busy || !state.file;
  if (label) label.textContent = busy ? "Menganalisis…" : "Analisis foto";
  if (spinner) spinner.hidden = !busy;
  els.resultCard.setAttribute("aria-busy", busy ? "true" : "false");
  els.engineSelect.disabled = busy;
  els.confInput.disabled = busy;
  els.calibrationInput.disabled = busy;
  els.fileInput.disabled = busy;
  els.dropzone.disabled = busy;
}

function showLoading() {
  els.resultEmpty.hidden = true;
  els.resultSummary.hidden = true;
  els.resultLoading.hidden = false;
  els.detailSection.hidden = true;
}

function showEmpty() {
  els.resultEmpty.hidden = false;
  els.resultLoading.hidden = true;
  els.resultSummary.hidden = true;
  els.detailSection.hidden = true;
  state.result = null;
}

function showError(message) {
  els.uploadError.textContent = message;
  els.uploadError.hidden = false;
  els.uploadError.focus();
}

function startLoadingProgress() {
  let index = 0;
  els.loadingStage.textContent = LOADING_STAGES[0];
  els.loadingBar.style.transform = "scaleX(0.1)";
  state.loadingTimer = setInterval(() => {
    index = Math.min(index + 1, LOADING_STAGES.length - 1);
    els.loadingStage.textContent = LOADING_STAGES[index];
    const progress = Math.min(92, 10 + index * 20) / 100;
    els.loadingBar.style.transform = `scaleX(${progress})`;
  }, 750);
}

function stopLoadingProgress() {
  if (state.loadingTimer) {
    clearInterval(state.loadingTimer);
    state.loadingTimer = null;
  }
  els.loadingBar.style.transform = "scaleX(1)";
}

function renderSummary(data) {
  const prediction = data.prediction || {};
  const phase = prediction.phase;
  const phaseClass = PHASE_CLASS[phase] || "phase-unknown";
  els.resultSummary.className = `result-summary ${phaseClass}`;

  els.phaseName.textContent = phase || "Belum terdeteksi";
  els.phaseAge.textContent = phase
    ? `Estimasi usia: ${prediction.estimated_age || "-"}`
    : "Estimasi usia belum tersedia";
  els.phaseNote.textContent = prediction.note || "";

  const confidence = Math.max(0, Math.min(1, Number(prediction.confidence) || 0));
  els.confidenceValue.textContent = pctFormatter.format(confidence);
  els.ringFg.style.strokeDashoffset = String(
    RING_CIRCUMFERENCE * (1 - confidence)
  );
  els.confidenceTitle.textContent = `Keyakinan prediksi ${pctFormatter.format(confidence)}`;

  els.modelBadge.textContent = prediction.model || "Model belum tersedia";
  els.modelBadge.className = prediction.is_demo ? "badge badge-soft" : "badge";

  const engine = data.engine || {};
  const yoloModel = (engine.yolo && engine.yolo.model) || {};
  const maskMap =
    yoloModel.metrics && yoloModel.metrics.mask
      ? yoloModel.metrics.mask.map50
      : null;
  let engineText = engine.label || "-";
  if (engine.used === "yolo" && yoloModel.architecture) {
    engineText = yoloModel.architecture;
    if (maskMap !== null && maskMap !== undefined) {
      engineText += ` · mAP50 ${decimal(maskMap)}`;
    }
  }
  els.engineMeta.textContent = engineText;
  els.engineMeta.title = engine.note || "";
  els.timeMeta.textContent = `${num(
    (data.timings && data.timings.total_ms) || 0,
    0
  )} ms · ${(data.leaves && data.leaves.length) || 0} daun`;

  renderProbabilities(prediction.probabilities || {});
  renderWarnings(data);

  els.resultEmpty.hidden = true;
  els.resultLoading.hidden = true;
  els.resultSummary.hidden = false;
}

function renderProbabilities(probabilities) {
  els.probList.replaceChildren();
  PHASES.forEach((phase) => {
    const value = Math.max(0, Math.min(1, Number(probabilities[phase]) || 0));
    const row = document.createElement("div");
    row.className = `prob-row ${PHASE_CLASS[phase] || ""}`;

    const name = document.createElement("span");
    name.className = "prob-name";
    name.textContent = phase;

    const track = document.createElement("div");
    track.className = "prob-track";
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-label", phase);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(Math.round(value * 100)));
    track.setAttribute("aria-valuetext", pctFormatter.format(value));

    const fill = document.createElement("span");
    fill.className = "prob-fill";
    fill.style.transform = `scaleX(${value})`;
    track.append(fill);

    const valueLabel = document.createElement("span");
    valueLabel.className = "prob-value";
    valueLabel.textContent = pctFormatter.format(value);

    row.append(name, track, valueLabel);
    els.probList.append(row);
  });
}

function renderWarnings(data) {
  const warnings = Array.isArray(data.warnings) ? [...data.warnings] : [];
  if (data.prediction && data.prediction.low_confidence && data.prediction.phase) {
    warnings.push(
      "Keyakinan prediksi rendah. Foto ulang dengan satu tanaman, latar kontras, dan cahaya merata."
    );
  }
  const unique = [...new Set(warnings)].filter(Boolean).slice(0, 5);
  els.summaryWarnings.replaceChildren();
  unique.forEach((text) => {
    const item = document.createElement("div");
    item.className = "warning-item";
    const icon = document.createElement("span");
    icon.innerHTML = ICONS.warning;
    icon.setAttribute("aria-hidden", "true");
    const body = document.createElement("span");
    body.textContent = text;
    item.append(icon, body);
    els.summaryWarnings.append(item);
  });
}

function renderDetail(data) {
  els.detailSection.hidden = false;
  const image = data.image || {};
  const engine = data.engine || {};
  els.detailCaption.textContent = `${image.filename || "foto"} · ${
    image.width || 0
  }×${image.height || 0} px · ${engine.label || "-"}`;

  renderPipelineTrace(data.timings || {});
  updateUnits(data);
  renderAggregates(data);
  renderLeafTable(data);
  renderRecommendations(data);
}

function renderPipelineTrace(timings) {
  const steps = [
    ["01", "Preprocessing", timings.preprocess_ms],
    ["02", "Segmentasi", timings.segment_ms],
    ["03", "Fitur OpenCV", timings.features_ms],
    ["04", "Klasifikasi", timings.classify_ms],
    ["05", "Total", timings.total_ms],
  ];
  els.pipelineTrace.replaceChildren();
  steps.forEach(([number, label, duration], index) => {
    const item = document.createElement("div");
    item.className = `trace-item${index === steps.length - 1 ? " trace-total" : ""}`;
    const numEl = document.createElement("span");
    numEl.className = "trace-num";
    numEl.textContent = number;
    const labelEl = document.createElement("span");
    labelEl.className = "trace-label";
    labelEl.textContent = label;
    const timeEl = document.createElement("span");
    timeEl.className = "trace-time";
    timeEl.textContent = `${num(duration || 0, 1)} ms`;
    item.append(numEl, labelEl, timeEl);
    els.pipelineTrace.append(item);
  });
}

function updateUnits(data) {
  const units = data.units || {};
  const lengthUnit = units.length || "px";
  const areaUnit = units.area || "px²";
  els.thLengthUnit.textContent = lengthUnit;
  els.thWidthUnit.textContent = lengthUnit;
  els.thAreaUnit.textContent = areaUnit;
  els.unitBadge.textContent = units.calibrated ? "terkalibrasi · cm" : "piksel";
}

function renderAggregates(data) {
  const aggregated = data.aggregated || {};
  const units = data.units || {};
  const lengthUnit = units.length || "px";
  const areaUnit = units.area || "px²";
  const items = [
    ["Jumlah daun", aggregated.leaf_count, "daun", 0],
    ["Panjang rata-rata", aggregated.mean_length, lengthUnit, 1],
    ["Lebar rata-rata", aggregated.mean_width, lengthUnit, 1],
    ["Panjang maksimum", aggregated.max_length, lengthUnit, 1],
    ["Lebar maksimum", aggregated.max_width, lengthUnit, 1],
    ["Luas total", aggregated.total_area, areaUnit, 1],
    ["Rasio L/W", aggregated.mean_ratio, "×", 2],
    ["Tutupan daun", aggregated.total_area_ratio, "%", 2],
  ];

  els.aggGrid.replaceChildren();
  items.forEach(([label, value, unit, digits]) => {
    const item = document.createElement("div");
    item.className = "agg-item";
    const labelEl = document.createElement("span");
    labelEl.className = "agg-label";
    labelEl.textContent = label;
    const valueEl = document.createElement("div");
    valueEl.className = "agg-value";
    valueEl.textContent = num(value, digits);
    const unitEl = document.createElement("span");
    unitEl.className = "agg-unit";
    unitEl.textContent = unit;
    valueEl.append(unitEl);
    item.append(labelEl, valueEl);
    els.aggGrid.append(item);
  });
}

function renderLeafTable(data) {
  const leaves = Array.isArray(data.leaves) ? data.leaves : [];
  const hasConfidence = leaves.some(
    (leaf) => leaf.confidence !== null && leaf.confidence !== undefined
  );
  els.leafTableBody.replaceChildren();
  els.leafCountBadge.textContent = `${leaves.length} daun`;
  els.leafTable.classList.toggle("hide-confidence", !hasConfidence);
  renderMaskGallery(leaves);
  renderLeafTableNote(leaves);

  if (!leaves.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = hasConfidence ? 6 : 5;
    cell.className = "table-empty";
    cell.textContent = "Tidak ada instance daun yang lolos filter area minimum.";
    row.append(cell);
    els.leafTableBody.append(row);
    return;
  }

  leaves.forEach((leaf) => {
    const row = document.createElement("tr");

    const idCell = document.createElement("td");
    const idWrap = document.createElement("span");
    idWrap.className = "leaf-id";
    const swatch = document.createElement("span");
    swatch.className = "leaf-swatch";
    swatch.style.background = leaf.color || "#1f7a4d";
    const idText = document.createElement("span");
    idText.textContent = `D${leaf.id}`;
    idWrap.append(swatch, idText);
    idCell.append(idWrap);

    const values = [
      [leaf.length, 1],
      [leaf.width, 1],
      [leaf.area, 1],
      [leaf.ratio, 2],
    ];
    row.append(idCell);
    values.forEach(([value, digits], position) => {
      const cell = document.createElement("td");
      cell.className = "num";
      cell.textContent = num(value, digits);
      if (position === 1 && leaf.occluded) {
        const unit = (data.units && data.units.length) || "px";
        const flag = document.createElement("span");
        flag.className = "width-flag";
        flag.textContent = "≈";
        flag.title =
          `Lebar diperkirakan dari bagian yang terlihat (${num(leaf.width_visible, 1)} ${unit}) karena terpotong objek lurus, misalnya penggaris.`;
        cell.append(flag);
      }
      row.append(cell);
    });

    const confidenceCell = document.createElement("td");
    confidenceCell.className = "num";
    confidenceCell.textContent =
      leaf.confidence === null || leaf.confidence === undefined
        ? "-"
        : pctFormatter.format(leaf.confidence);
    row.append(confidenceCell);

    els.leafTableBody.append(row);
  });
}

function renderLeafTableNote(leaves) {
  if (!els.leafTableNote) return;
  const occluded = leaves.filter((leaf) => leaf.occluded);
  els.leafTableNote.hidden = occluded.length === 0;
  if (!occluded.length) {
    els.leafTableNote.textContent = "";
    return;
  }
  const ids = occluded.map((leaf) => `D${leaf.id}`).join(", ");
  els.leafTableNote.textContent = `≈ ${ids}: lebar direkonstruksi dari bagian yang terlihat karena daun terpotong objek lurus. Panjang dan luas tetap dihitung dari piksel yang terlihat.`;
}

function renderMaskGallery(leaves) {
  els.maskGallery.replaceChildren();
  leaves.slice(0, 16).forEach((leaf) => {
    const figure = document.createElement("figure");
    figure.className = "mask-item";
    const image = document.createElement("img");
    image.src = leaf.mask;
    image.alt = `Mask instance daun D${leaf.id}`;
    image.width = 112;
    image.height = 112;
    image.loading = "lazy";
    const caption = document.createElement("figcaption");
    caption.textContent = `D${leaf.id}`;
    figure.append(image, caption);
    els.maskGallery.append(figure);
  });

  if (leaves.length > 16) {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = `Menampilkan 16 dari ${leaves.length} mask. Tabel memuat semua instance.`;
    els.maskGallery.append(note);
  }
}

function renderRecommendations(data) {
  const recommendations = data.recommendations || {};
  const phase = data.prediction && data.prediction.phase;
  const phaseClass = PHASE_CLASS[phase] || "phase-unknown";
  els.recPhaseBadge.className = `badge ${phaseClass}`;
  els.recPhaseBadge.textContent = phase || "Belum terdeteksi";
  els.recSummary.textContent = recommendations.summary || "Belum ada saran untuk foto ini.";
  els.recNote.textContent = recommendations.note || "";

  els.recList.replaceChildren();
  const items = Array.isArray(recommendations.items) ? recommendations.items : [];
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "rec-item rec-info";
    const icon = document.createElement("span");
    icon.className = "rec-icon";
    icon.innerHTML = ICONS.info;
    icon.setAttribute("aria-hidden", "true");
    const body = document.createElement("div");
    const title = document.createElement("h4");
    title.textContent = "Belum ada saran";
    const detail = document.createElement("p");
    detail.textContent = "Analisis foto terlebih dahulu untuk mendapatkan rekomendasi.";
    body.append(title, detail);
    empty.append(icon, body);
    els.recList.append(empty);
    return;
  }

  items.forEach((item) => {
    const type = ["action", "warning", "info"].includes(item.type)
      ? item.type
      : "info";
    const listItem = document.createElement("li");
    listItem.className = `rec-item rec-${type}`;
    const icon = document.createElement("span");
    icon.className = "rec-icon";
    icon.innerHTML = ICONS[type];
    icon.setAttribute("aria-hidden", "true");
    const body = document.createElement("div");
    const title = document.createElement("h4");
    title.textContent = item.title || "Saran";
    const detail = document.createElement("p");
    detail.textContent = item.detail || "";
    body.append(title, detail);
    listItem.append(icon, body);
    els.recList.append(listItem);
  });
}

function setView(view) {
  if (!state.result || !state.result.image) return;
  const image = state.result.image;
  const engine = state.result.engine || {};
  const sources = {
    original: {
      src: image.original,
      alt: "Foto asli tanaman pakcoy yang diunggah",
      caption: "Foto asli setelah resize ke 640 px, sebelum CLAHE.",
    },
    preprocessed: {
      src: image.preprocessed,
      alt: "Citra pakcoy setelah resize, CLAHE, dan Gaussian blur",
      caption: "Resize 640 px, CLAHE, dan Gaussian blur ringan.",
    },
    annotated: {
      src: image.annotated,
      alt: "Hasil segmentasi dengan kontur, kotak minimum, dan nomor daun",
      caption: `Hasil ${engine.label || "segmentasi"}: kontur, kotak minimum, dan nomor instance.`,
    },
  };
  const source = sources[view];
  if (!source) return;

  state.activeView = view;
  els.resultImage.src = source.src;
  els.resultImage.alt = source.alt;
  els.resultImage.width = image.width || 640;
  els.resultImage.height = image.height || 480;
  els.imageCaption.textContent = source.caption;
  els.visualPanel.setAttribute("aria-labelledby", `tab-${view}`);

  document.querySelectorAll(".tab").forEach((tab) => {
    const selected = tab.dataset.view === view;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
}

function downloadCsv() {
  const data = state.result;
  if (!data) return;
  const units = data.units || {};
  const lengthUnit = units.length || "px";
  const areaUnit = units.area === "cm²" ? "cm2" : "px";
  const headers = [
    "leaf_id",
    `length_${lengthUnit}`,
    `width_${lengthUnit}`,
    `width_visible_${lengthUnit}`,
    "width_reconstructed",
    `area_${areaUnit}`,
    "ratio_length_width",
    "confidence",
    "centroid_x",
    "centroid_y",
  ];
  const rows = (data.leaves || []).map((leaf) => [
    leaf.id,
    leaf.length,
    leaf.width,
    leaf.width_visible,
    leaf.occluded ? "yes" : "no",
    leaf.area,
    leaf.ratio,
    leaf.confidence === null || leaf.confidence === undefined ? "" : leaf.confidence,
    leaf.centroid ? leaf.centroid[0] : "",
    leaf.centroid ? leaf.centroid[1] : "",
  ]);
  const csv =
    "\ufeff" +
    [headers, ...rows]
      .map((row) => row.map(csvCell).join(","))
      .join("\r\n");
  downloadBlob(
    csv,
    `fitur-daun-${fileSlug(data.image && data.image.filename)}.csv`,
    "text/csv;charset=utf-8"
  );
  toast("Tabel fitur daun diunduh sebagai CSV.", "success");
}

function downloadJson() {
  const data = state.result;
  if (!data) return;
  const payload = {
    meta: {
      app: "Pakcoy Vision",
      exported_at: new Date().toISOString(),
      filename: data.image ? data.image.filename : null,
    },
    engine: data.engine,
    image: data.image
      ? { width: data.image.width, height: data.image.height }
      : null,
    leaves: (data.leaves || []).map((leaf) => {
      const { mask, ...rest } = leaf;
      return rest;
    }),
    aggregated: data.aggregated,
    prediction: data.prediction,
    recommendations: data.recommendations,
    timings: data.timings,
    units: data.units,
    warnings: data.warnings,
  };
  downloadBlob(
    JSON.stringify(payload, null, 2),
    `hasil-analisis-${fileSlug(data.image && data.image.filename)}.json`,
    "application/json"
  );
  toast("Hasil analisis diunduh sebagai JSON.", "success");
}

function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1200);
}

function csvCell(value) {
  const text = String(value === null || value === undefined ? "" : value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function fileSlug(name) {
  return (
    String(name || "pakcoy")
      .replace(/\.[^.]+$/, "")
      .replace(/[^a-z0-9]+/gi, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase() || "pakcoy"
  );
}

let toastTimer = null;

function toast(message, type = "info") {
  els.toast.textContent = message;
  els.toast.className = `toast toast-${type}`;
  els.toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    els.toast.hidden = true;
  }, 4200);
}

init();
