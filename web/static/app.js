// Niivue loaded via ESM CDN (no build step). To vendor offline, save the
// file at https://unpkg.com/@niivue/niivue/dist/index.js as
// web/static/lib/niivue.esm.js and change the import below.
import { Niivue, SLICE_TYPE, SHOW_RENDER } from "https://unpkg.com/@niivue/niivue/dist/index.js";

const $ = (sel) => document.querySelector(sel);
const fmt = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));

// Niivue instances. Created once; volumes are swapped via loadVolumes().
const nv3d = new Niivue({
  show3Dcrosshair: false,
  backColor: [0, 0, 0, 1],
  isColorbar: false,
});
nv3d.attachToCanvas($("#nv3d-canvas"));
nv3d.setSliceType(SLICE_TYPE.RENDER);

const nvMPR = new Niivue({
  show3Dcrosshair: true,
  backColor: [0, 0, 0, 1],
  isColorbar: false,
});
nvMPR.attachToCanvas($("#nvmpr-canvas"));
nvMPR.setSliceType(SLICE_TYPE.MULTIPLANAR);

// nvUnc created lazily (canvas is display:none at page load → 0×0 dimensions).
let nvUnc = null;
function getOrCreateNvUnc() {
  if (nvUnc) return nvUnc;
  nvUnc = new Niivue({ show3Dcrosshair: true, backColor: [0,0,0,1], isColorbar: false });
  nvUnc.attachToCanvas($("#nvunc-canvas"));
  nvUnc.setSliceType(SLICE_TYPE.MULTIPLANAR);
  // Default auto-layout sizes each tile to the volume's physical extent, so the
  // axial/coronal/sagittal panels come out different sizes on this wide canvas.
  // Force equal-sized tiles and drop the (dark, near-empty) 3D render tile so
  // the result is a clean, uniform row of the three planes.
  nvUnc.opts.multiplanarEqualSize = true;
  try { nvUnc.opts.multiplanarShowRender = SHOW_RENDER.NEVER; } catch (_) {}
  // Force a single row of the three planes (0=auto,1=col,2=grid,3=row).
  try { nvUnc.opts.multiplanarLayout = 3; } catch (_) {}
  return nvUnc;
}

let lastResp = null;             // last /api/predict response
let baseModality = "t1ce";
let _uncPollTimer = null;        // setInterval handle for uncertainty polling

// --- Meta -----------------------------------------------------------------
fetch("/api/meta").then(r => r.json()).then(meta => {
  if (!meta.ready) {
    $("#model-badge").textContent = "Model: not ready";
    $("#status").textContent = "Server is starting up...";
    $("#status").classList.add("error");
  } else {
    const name = meta.model_name || "Model";
    $("#model-badge").textContent = `Model: ${name}`;
  }
}).catch(() => {
  $("#model-badge").textContent = "Model: connection error";
});

// --- AAL region → plain-language description ------------------------------
// Maps an AAL3 base name (hemisphere stripped) to {plain, lobe, role}.
// Prefix-matched so all 170 labels resolve without a 170-row table.
const ANATOMY_MAP = [
  [/^Precentral/,        "Precentral gyrus",          "Frontal lobe",   "primary motor cortex — voluntary movement"],
  [/^Postcentral/,       "Postcentral gyrus",         "Parietal lobe",  "primary somatosensory cortex — touch & body sense"],
  [/^Frontal_Sup_Medial/,"Superior medial frontal",   "Frontal lobe",   "decision-making & social cognition"],
  [/^Frontal_Sup/,       "Superior frontal gyrus",    "Frontal lobe",   "working memory & planning"],
  [/^Frontal_Mid/,       "Middle frontal gyrus",      "Frontal lobe",   "attention & executive function"],
  [/^Frontal_Inf/,       "Inferior frontal gyrus",    "Frontal lobe",   "speech production (Broca's area, left)"],
  [/^Frontal_Med_Orb/,   "Medial orbitofrontal",      "Frontal lobe",   "reward & emotional regulation"],
  [/^OFC|^Frontal.*Orb/, "Orbitofrontal cortex",      "Frontal lobe",   "reward, decision-making & impulse control"],
  [/^Rectus/,            "Gyrus rectus",              "Frontal lobe",   "olfactory & limbic processing"],
  [/^Olfactory/,         "Olfactory cortex",          "Frontal lobe",   "sense of smell"],
  [/^Supp_Motor_Area/,   "Supplementary motor area",  "Frontal lobe",   "movement planning & sequencing"],
  [/^Rolandic_Oper/,     "Rolandic operculum",        "Frontal lobe",   "face/mouth motor control & taste"],
  [/^Paracentral_Lobule/,"Paracentral lobule",        "Frontal/Parietal","motor & sensory control of the legs"],
  [/^Insula/,            "Insular cortex",            "Insula",         "interoception, taste & emotion"],
  [/^Cingulate_Ant|^ACC/,"Anterior cingulate",        "Limbic system",  "emotion, error detection & motivation"],
  [/^Cingulate_Mid/,     "Mid cingulate",             "Limbic system",  "pain processing & action selection"],
  [/^Cingulate_Post/,    "Posterior cingulate",       "Limbic system",  "memory recall & self-reflection"],
  [/^Hippocampus/,       "Hippocampus",               "Limbic system",  "memory formation & spatial navigation"],
  [/^ParaHippocampal/,   "Parahippocampal gyrus",     "Limbic system",  "memory encoding & scene recognition"],
  [/^Amygdala/,          "Amygdala",                  "Limbic system",  "fear, emotion & threat detection"],
  [/^Calcarine/,         "Calcarine cortex",          "Occipital lobe", "primary visual cortex (V1) — core vision"],
  [/^Cuneus/,            "Cuneus",                    "Occipital lobe", "basic visual processing"],
  [/^Lingual/,           "Lingual gyrus",             "Occipital lobe", "vision — letters, words & faces"],
  [/^Occipital/,         "Occipital gyrus",           "Occipital lobe", "higher-order visual processing"],
  [/^Fusiform/,          "Fusiform gyrus",            "Temporal/Occipital","face & object recognition"],
  [/^Parietal_Sup/,      "Superior parietal lobule",  "Parietal lobe",  "spatial orientation & attention"],
  [/^Parietal_Inf/,      "Inferior parietal lobule",  "Parietal lobe",  "language, math & spatial reasoning"],
  [/^SupraMarginal/,     "Supramarginal gyrus",       "Parietal lobe",  "language perception & phonology"],
  [/^Angular/,           "Angular gyrus",             "Parietal lobe",  "reading, math & memory retrieval"],
  [/^Precuneus/,         "Precuneus",                 "Parietal lobe",  "self-awareness & visuospatial imagery"],
  [/^Heschl/,            "Heschl's gyrus",            "Temporal lobe",  "primary auditory cortex — hearing"],
  [/^Temporal_Sup/,      "Superior temporal gyrus",   "Temporal lobe",  "auditory processing & language (Wernicke)"],
  [/^Temporal_Mid/,      "Middle temporal gyrus",     "Temporal lobe",  "word meaning & visual motion"],
  [/^Temporal_Inf/,      "Inferior temporal gyrus",   "Temporal lobe",  "object & visual recognition"],
  [/^Temporal_Pole/,     "Temporal pole",             "Temporal lobe",  "social & emotional processing"],
  [/^Caudate/,           "Caudate nucleus",           "Basal ganglia",  "movement control & learning"],
  [/^Putamen/,           "Putamen",                   "Basal ganglia",  "movement regulation & motor skills"],
  [/^Pallidum/,          "Globus pallidus",           "Basal ganglia",  "movement inhibition & posture"],
  [/^N_Acc/,             "Nucleus accumbens",         "Basal ganglia",  "reward, pleasure & motivation"],
  [/^Thalamus|^Thal_/,   "Thalamus",                  "Diencephalon",   "sensory & motor relay hub"],
  [/^Cerebellum|^Vermis/,"Cerebellum",                "Cerebellum",     "balance, coordination & fine movement"],
  [/^VTA/,               "Ventral tegmental area",    "Midbrain",       "dopamine reward pathway"],
  [/^SN_/,               "Substantia nigra",          "Midbrain",       "dopamine production & movement"],
  [/^Red_N/,             "Red nucleus",               "Midbrain",       "motor coordination"],
  [/^LC_/,               "Locus coeruleus",           "Brainstem",      "arousal, attention & stress response"],
  [/^Raphe/,             "Raphe nuclei",              "Brainstem",      "serotonin regulation & mood"],
];

function regionInfo(name) {
  let hemi = "";
  if (/_L$/.test(name)) hemi = "Left ";
  else if (/_R$/.test(name)) hemi = "Right ";
  const base = name.replace(/_[LR]$/, "");
  for (const [re, plain, lobe, role] of ANATOMY_MAP) {
    if (re.test(base)) {
      return { title: `${hemi}${plain}`, lobe, role };
    }
  }
  return {
    title: `${hemi}${base.replace(/_/g, " ")}`,
    lobe: "Brain region",
    role: "labelled brain structure from the AAL3 atlas",
  };
}

// --- File picker: show chosen filename, mark row filled ------------------
document.querySelectorAll('.file-row input[type="file"]').forEach((inp) => {
  inp.addEventListener("change", () => {
    const row = inp.closest(".file-row");
    const nameEl = row.querySelector(".file-name");
    if (inp.files && inp.files.length) {
      nameEl.textContent = inp.files[0].name;
      row.classList.add("filled");
    } else {
      nameEl.textContent = "Choose a .nii / .nii.gz file";
      row.classList.remove("filled");
    }
  });
});

// --- Predict --------------------------------------------------------------
$("#upload-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const fd = new FormData(form);
  $("#predict-btn").disabled = true;
  $("#status").classList.remove("error");
  $("#status").textContent = "Uploading and running inference (~10 s)...";
  // Reset uncertainty panel from any previous run
  if (_uncPollTimer) { clearInterval(_uncPollTimer); _uncPollTimer = null; }
  $("#uncertainty-section").style.display = "none";
  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    if (!r.ok) {
      const txt = await r.text();
      throw new Error(`HTTP ${r.status}: ${txt}`);
    }
    const resp = await r.json();
    lastResp = resp;
    await renderResults(resp);
    $("#status").textContent = "Done.";
  } catch (e) {
    console.error(e);
    $("#status").textContent = `Error: ${e.message || e}`;
    $("#status").classList.add("error");
  } finally {
    $("#predict-btn").disabled = false;
  }
});

function confLabel(v) {
  if (v == null) return "Region not predicted";
  if (v >= 0.90) return "Very high certainty";
  if (v >= 0.80) return "High certainty";
  if (v >= 0.70) return "Moderate certainty";
  return "Low certainty — interpret carefully";
}

// --- Render volumes + cards + summary ------------------------------------
async function renderResults(resp) {
  const vol = resp.volumes_ml || {};
  $("#vol-wt").textContent = vol.WT == null ? "—" : `${fmt(vol.WT, 1)} mL`;
  $("#vol-tc").textContent = vol.TC == null ? "—" : `${fmt(vol.TC, 1)} mL`;
  $("#vol-et").textContent = vol.ET == null ? "—" : `${fmt(vol.ET, 1)} mL`;

  const conf = resp.confidence || {};
  const pct = (v) => (v == null ? "—" : `${Math.round(Number(v) * 100)}%`);
  $("#conf-wt").textContent = pct(conf.WT);
  $("#conf-tc").textContent = pct(conf.TC);
  $("#conf-et").textContent = pct(conf.ET);
  $("#conf-wt-desc").textContent = confLabel(conf.WT);
  $("#conf-tc-desc").textContent = confLabel(conf.TC);
  $("#conf-et-desc").textContent = confLabel(conf.ET);

  const flag = $("#conf-flag");
  const low = ["WT", "TC", "ET"].filter(r => conf[r] != null && conf[r] < 0.70);
  if (low.length) {
    flag.textContent = `⚠ ${low.join(" / ")} confidence below 0.70 — interpret these regions with caution.`;
    flag.classList.add("visible");
  } else {
    flag.textContent = "";
    flag.classList.remove("visible");
  }

  const wtRisk = (resp.risk && resp.risk.WT) || { level: "Unknown", color: "#94a3b8" };
  const badge = $("#risk-badge");
  badge.textContent = wtRisk.level || "Unknown";
  badge.style.background = wtRisk.color || "#94a3b8";
  $("#risk-pct").textContent = wtRisk.percentile == null
    ? "Population reference unavailable"
    : `${wtRisk.percentile}th percentile · larger than ${wtRisk.percentile}% of the cohort`;

  const ul = $("#anatomy-list");
  ul.innerHTML = "";
  if (!resp.anatomy_top || !resp.anatomy_top.length) {
    ul.innerHTML = '<li class="muted">Anatomy unavailable for this volume.</li>';
  } else {
    for (const a of resp.anatomy_top) {
      const info = regionInfo(a.name);
      const li = document.createElement("li");
      li.innerHTML = `
        <div class="an-text">
          <div class="an-title">${info.title}
            <span class="an-lobe">${info.lobe}</span>
          </div>
          <div class="an-role">${info.role || "—"}</div>
          <div class="an-code">${a.name}</div>
        </div>
        <div class="an-pct-wrap">
          <span class="pct">${a.pct}%</span>
          <div class="an-bar"><div class="an-bar-fill" style="width:${Math.min(100, a.pct)}%"></div></div>
        </div>`;
      ul.appendChild(li);
    }
  }

  $("#summary-text").classList.remove("muted");
  $("#summary-text").textContent = resp.summary || "";

  $("#modality-select").disabled = false;
  $("#opacity").disabled = false;
  $("#download-btn").disabled = false;

  await loadViewers(resp);
  startUncertaintyPoll(resp);
}

// --- Uncertainty polling -------------------------------------------------
function startUncertaintyPoll(resp) {
  if (_uncPollTimer) clearInterval(_uncPollTimer);
  const statusUrl = resp.uncertainty_url;
  if (!statusUrl) return;

  // Show the section immediately with "Computing…" badge
  const sec = $("#uncertainty-section");
  const badge = $("#uncertainty-badge");
  sec.style.display = "block";
  badge.textContent = "Computing…";
  badge.className = "unc-badge computing";

  _uncPollTimer = setInterval(async () => {
    try {
      const r = await fetch(statusUrl);
      const s = await r.json();
      if (s.ready && s.url) {
        clearInterval(_uncPollTimer);
        _uncPollTimer = null;
        badge.textContent = "Ready";
        badge.className = "unc-badge ready";
        await loadUncertaintyViewer(resp.nifti_urls[baseModality], s.url);
      }
    } catch (_) { /* ignore transient fetch errors */ }
  }, 3000);
}

async function loadUncertaintyViewer(baseUrl, uncUrl) {
  for (let i = 0; i < 40; i++) {
    if ($("#nvunc-canvas").clientWidth > 0) break;
    await new Promise(r => setTimeout(r, 50));
  }
  await new Promise(r => setTimeout(r, 50));
  const nv = getOrCreateNvUnc();
  await nv.loadVolumes([
    { url: baseUrl, colormap: "gray",    opacity: 1.0 },
    { url: uncUrl,  colormap: "inferno", opacity: 0.65, cal_min: 0 },
  ]);
  // The section starts display:none, so Niivue captured a stale/near-square
  // draw buffer; the 3:1 montage then fits into a narrow centered strip. Wait
  // for layout to settle, then re-run Niivue's own resize handler so the draw
  // buffer matches the full (wide) canvas and the tiles fill it.
  const CANVAS_H = 460; // displayed height in CSS px; tiles scale to fill it
  const syncSize = () => {
    const c = document.getElementById("nvunc-canvas");
    if (c) {
      const cssW = c.getBoundingClientRect().width || c.clientWidth;
      if (cssW > 0) {
        const dpr = window.devicePixelRatio || 1;
        // Force display height (independent of any CSS) and match the draw
        // buffer to the full width × height so the montage fills the canvas
        // instead of scaling to a small centered region.
        c.style.height = CANVAS_H + "px";
        c.width = Math.round(cssW * dpr);
        c.height = Math.round(CANVAS_H * dpr);
      }
    }
    try { window.dispatchEvent(new Event("resize")); } catch (_) {}
    try { nv.resizeListener(); } catch (_) {}
    try { nv.drawScene(); } catch (_) {}
  };
  requestAnimationFrame(() => requestAnimationFrame(syncSize));
  setTimeout(syncSize, 150);
  setTimeout(syncSize, 400);
}

// --- Niivue --------------------------------------------------------------
async function loadViewers(resp) {
  const baseUrl = resp.nifti_urls[baseModality];
  const segUrl  = resp.nifti_urls.seg;
  const opacity = Number($("#opacity").value) / 100;

  // BraTS label colormap (maximally distinct hues — yellow vs blue are
  // complementary so ED and ET never blend on the 3D render):
  //   0 = background (transparent)
  //   1 = NCR  → red    rgb(239, 68, 68)
  //   2 = ED   → green  rgb( 34,197, 94)
  //   3 = ET   → blue   rgb( 29, 78,216)
  const labelColormap = {
    R: [0, 239, 34, 29],
    G: [0, 68, 197, 78],
    B: [0, 68, 94, 216],
    A: [0, 255, 255, 255],
    I: [0, 1, 2, 3],
    labels: ["BG", "NCR", "ED", "ET"],
  };
  if (!nv3d.addColormap) {
    // older niivue versions
  } else {
    try { nv3d.addColormap("brats", labelColormap); } catch (_) {}
    try { nvMPR.addColormap("brats", labelColormap); } catch (_) {}
  }

  const volsRender = [
    { url: baseUrl, colormap: "gray", opacity: 1.0 },
    { url: segUrl,  colormap: "brats", opacity: opacity, cal_min: 0, cal_max: 3 },
  ];
  for (const nv of [nv3d, nvMPR]) {
    await nv.loadVolumes(volsRender);
    // Treat the segmentation as a DISCRETE label map. Without this Niivue
    // trilinearly interpolates the seg volume, blending ED(yellow)↔ET(blue)
    // into teal "light blue" and washing the small NCR(red) into orange.
    const seg = nv.volumes[1];
    let labelOk = false;
    if (seg && seg.setColormapLabel) {
      try {
        seg.setColormapLabel(labelColormap);
        nv.updateGLVolume();
        labelOk = true;
      } catch (_) {}
    }
    // Fallback for older Niivue: force nearest-neighbour so labels stay pure.
    if (!labelOk) { try { nv.setInterpolation(true); } catch (_) {} }
  }
}

$("#modality-select").addEventListener("change", async (ev) => {
  baseModality = ev.target.value;
  if (lastResp) await loadViewers(lastResp);
});

$("#opacity").addEventListener("input", (ev) => {
  const op = Number(ev.target.value) / 100;
  for (const nv of [nv3d, nvMPR]) {
    if (nv.volumes && nv.volumes.length > 1) {
      nv.setOpacity(1, op);
      nv.drawScene();
    }
  }
});

// --- Report download -----------------------------------------------------
$("#download-btn").addEventListener("click", async () => {
  if (!lastResp) return;
  const sid = lastResp.session_id;
  $("#status").classList.remove("error");
  $("#status").textContent = "Bundling report...";
  try {
    // Best-effort screenshot — a blank/failed capture must NOT block the zip.
    try {
      const blob = await canvasToBlob($("#nv3d-canvas"));
      if (blob && blob.size > 0) {
        await fetch(`/api/session/${sid}/screenshot`, { method: "POST", body: blob });
      }
    } catch (e) { console.warn("screenshot skipped:", e); }

    // Fetch the zip as a blob and trigger a real download via an anchor.
    const r = await fetch(`/api/session/${sid}/report`);
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const zip = await r.blob();
    const url = URL.createObjectURL(zip);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${sid.slice(0, 8)}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    $("#status").textContent = "Report downloaded.";
  } catch (e) {
    console.error(e);
    $("#status").textContent = `Download failed: ${e.message || e}`;
    $("#status").classList.add("error");
  }
});

function canvasToBlob(canvas) {
  return new Promise(resolve => {
    try {
      // Force a fresh frame so the WebGL buffer isn't empty at capture time.
      try { nv3d.drawScene(); } catch (_) {}
      requestAnimationFrame(() => {
        try { canvas.toBlob(b => resolve(b), "image/png"); }
        catch { resolve(null); }
      });
    } catch { resolve(null); }
  });
}
