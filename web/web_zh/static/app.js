// Niivue 透過 ESM CDN 載入（免建置步驟）。若要離線使用，將
// https://unpkg.com/@niivue/niivue/dist/index.js 存成
// web_zh/static/lib/niivue.esm.js 並修改下方 import。
import { Niivue, SLICE_TYPE, SHOW_RENDER } from "https://unpkg.com/@niivue/niivue/dist/index.js";

const $ = (sel) => document.querySelector(sel);
const fmt = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));

// Niivue 實例。建立一次，之後用 loadVolumes() 抽換體積。
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

// nvUnc 採延遲建立（canvas 在頁面載入時為 display:none → 尺寸 0×0）。
let nvUnc = null;
function getOrCreateNvUnc() {
  if (nvUnc) return nvUnc;
  nvUnc = new Niivue({ show3Dcrosshair: true, backColor: [0,0,0,1], isColorbar: false });
  nvUnc.attachToCanvas($("#nvunc-canvas"));
  nvUnc.setSliceType(SLICE_TYPE.MULTIPLANAR);
  // 預設自動佈局會依體積物理尺寸決定 tile 大小，導致三個切面在寬版 canvas 上
  // 大小不一。強制等寬 tile 並關閉（幾乎空白的）3D render tile，讓三個切面
  // 排成一排乾淨等寬。
  nvUnc.opts.multiplanarEqualSize = true;
  try { nvUnc.opts.multiplanarShowRender = SHOW_RENDER.NEVER; } catch (_) {}
  // 0=auto, 1=col, 2=grid, 3=row。
  try { nvUnc.opts.multiplanarLayout = 3; } catch (_) {}
  return nvUnc;
}

let lastResp = null;             // 最近一次 /api/predict 回應
let baseModality = "t1ce";
let _uncPollTimer = null;        // uncertainty 輪詢 setInterval handle

// --- Meta -----------------------------------------------------------------
fetch("/api/meta").then(r => r.json()).then(meta => {
  if (!meta.ready) {
    $("#model-badge").textContent = "Model：尚未就緒";
    $("#status").textContent = "伺服器啟動中…";
    $("#status").classList.add("error");
  } else {
    const name = meta.model_name || "Model";
    $("#model-badge").textContent = `Model：${name}`;
  }
}).catch(() => {
  $("#model-badge").textContent = "Model：連線錯誤";
});

// --- AAL 區域 → 白話說明 --------------------------------------------------
// 將 AAL3 基本名稱（去掉半球後綴）對應到 {白話名, 腦葉, 功能}。
// 以前綴比對，因此 170 個標籤不需逐筆列出即可解析。
const ANATOMY_MAP = [
  [/^Precentral/,        "中央前迴",        "額葉",       "初級運動皮質 — 自主運動"],
  [/^Postcentral/,       "中央後迴",        "頂葉",       "初級體感皮質 — 觸覺與身體感覺"],
  [/^Frontal_Sup_Medial/,"上內側額葉",      "額葉",       "決策與社會認知"],
  [/^Frontal_Sup/,       "上額迴",          "額葉",       "工作記憶與規劃"],
  [/^Frontal_Mid/,       "中額迴",          "額葉",       "注意力與執行功能"],
  [/^Frontal_Inf/,       "下額迴",          "額葉",       "語言產生（布若卡區，左側）"],
  [/^Frontal_Med_Orb/,   "內側眶額",        "額葉",       "獎賞與情緒調節"],
  [/^OFC|^Frontal.*Orb/, "眶額皮質",        "額葉",       "獎賞、決策與衝動控制"],
  [/^Rectus/,            "直迴",            "額葉",       "嗅覺與邊緣系統處理"],
  [/^Olfactory/,         "嗅覺皮質",        "額葉",       "嗅覺"],
  [/^Supp_Motor_Area/,   "輔助運動區",      "額葉",       "動作規劃與排序"],
  [/^Rolandic_Oper/,     "中央溝蓋部",      "額葉",       "臉部／口部運動控制與味覺"],
  [/^Paracentral_Lobule/,"旁中央小葉",      "額／頂葉",   "下肢的運動與感覺控制"],
  [/^Insula/,            "腦島皮質",        "腦島",       "內感受、味覺與情緒"],
  [/^Cingulate_Ant|^ACC/,"前扣帶迴",        "邊緣系統",   "情緒、錯誤偵測與動機"],
  [/^Cingulate_Mid/,     "中扣帶迴",        "邊緣系統",   "疼痛處理與動作選擇"],
  [/^Cingulate_Post/,    "後扣帶迴",        "邊緣系統",   "記憶回想與自我反思"],
  [/^Hippocampus/,       "海馬迴",          "邊緣系統",   "記憶形成與空間導航"],
  [/^ParaHippocampal/,   "海馬旁迴",        "邊緣系統",   "記憶編碼與場景辨識"],
  [/^Amygdala/,          "杏仁核",          "邊緣系統",   "恐懼、情緒與威脅偵測"],
  [/^Calcarine/,         "距狀溝皮質",      "枕葉",       "初級視覺皮質（V1）— 核心視覺"],
  [/^Cuneus/,            "楔葉",            "枕葉",       "基本視覺處理"],
  [/^Lingual/,           "舌迴",            "枕葉",       "視覺 — 字母、文字與臉孔"],
  [/^Occipital/,         "枕迴",            "枕葉",       "高階視覺處理"],
  [/^Fusiform/,          "梭狀迴",          "顳／枕葉",   "臉孔與物件辨識"],
  [/^Parietal_Sup/,      "上頂小葉",        "頂葉",       "空間定向與注意力"],
  [/^Parietal_Inf/,      "下頂小葉",        "頂葉",       "語言、數學與空間推理"],
  [/^SupraMarginal/,     "緣上迴",          "頂葉",       "語言感知與語音處理"],
  [/^Angular/,           "角迴",            "頂葉",       "閱讀、數學與記憶提取"],
  [/^Precuneus/,         "楔前葉",          "頂葉",       "自我覺察與視覺空間想像"],
  [/^Heschl/,            "赫氏迴",          "顳葉",       "初級聽覺皮質 — 聽覺"],
  [/^Temporal_Sup/,      "上顳迴",          "顳葉",       "聽覺處理與語言（韋尼克區）"],
  [/^Temporal_Mid/,      "中顳迴",          "顳葉",       "詞義與視覺動態"],
  [/^Temporal_Inf/,      "下顳迴",          "顳葉",       "物件與視覺辨識"],
  [/^Temporal_Pole/,     "顳極",            "顳葉",       "社會與情緒處理"],
  [/^Caudate/,           "尾狀核",          "基底核",     "運動控制與學習"],
  [/^Putamen/,           "殼核",            "基底核",     "運動調節與動作技能"],
  [/^Pallidum/,          "蒼白球",          "基底核",     "運動抑制與姿勢"],
  [/^N_Acc/,             "依核",            "基底核",     "獎賞、愉悅與動機"],
  [/^Thalamus|^Thal_/,   "視丘",            "間腦",       "感覺與運動的中繼樞紐"],
  [/^Cerebellum|^Vermis/,"小腦",            "小腦",       "平衡、協調與精細動作"],
  [/^VTA/,               "腹側被蓋區",      "中腦",       "多巴胺獎賞路徑"],
  [/^SN_/,               "黑質",            "中腦",       "多巴胺生成與運動"],
  [/^Red_N/,             "紅核",            "中腦",       "運動協調"],
  [/^LC_/,               "藍斑核",          "腦幹",       "覺醒、注意力與壓力反應"],
  [/^Raphe/,             "縫核",            "腦幹",       "血清素調節與情緒"],
];

function regionInfo(name) {
  let hemi = "";
  if (/_L$/.test(name)) hemi = "左 ";
  else if (/_R$/.test(name)) hemi = "右 ";
  const base = name.replace(/_[LR]$/, "");
  for (const [re, plain, lobe, role] of ANATOMY_MAP) {
    if (re.test(base)) {
      return { title: `${hemi}${plain}`, lobe, role };
    }
  }
  return {
    title: `${hemi}${base.replace(/_/g, " ")}`,
    lobe: "腦區域",
    role: "AAL3 圖譜標記的腦結構",
  };
}

// --- 檔案選擇：顯示已選檔名、標記該列已填 --------------------------------
document.querySelectorAll('.file-row input[type="file"]').forEach((inp) => {
  inp.addEventListener("change", () => {
    const row = inp.closest(".file-row");
    const nameEl = row.querySelector(".file-name");
    if (inp.files && inp.files.length) {
      nameEl.textContent = inp.files[0].name;
      row.classList.add("filled");
    } else {
      nameEl.textContent = "選擇 .nii / .nii.gz 檔案";
      row.classList.remove("filled");
    }
  });
});

// --- 預測 -----------------------------------------------------------------
$("#upload-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  const fd = new FormData(form);
  $("#predict-btn").disabled = true;
  $("#status").classList.remove("error");
  $("#status").textContent = "上傳並執行推論中（約 10 秒）…";
  // 重置前次的不確定性面板
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
    $("#status").textContent = "完成。";
  } catch (e) {
    console.error(e);
    $("#status").textContent = `錯誤：${e.message || e}`;
    $("#status").classList.add("error");
  } finally {
    $("#predict-btn").disabled = false;
  }
});

function confLabel(v) {
  if (v == null) return "未預測此區域";
  if (v >= 0.90) return "信心極高";
  if (v >= 0.80) return "信心高";
  if (v >= 0.70) return "信心中等";
  return "信心低 — 請謹慎判讀";
}

// --- 渲染體積 + 卡片 + 摘要 ----------------------------------------------
async function renderResults(resp) {
  const vol = resp.volumes_ml || {};
  $("#vol-wt").textContent = vol.WT == null ? "—" : `${fmt(vol.WT, 1)} mL`;
  $("#vol-tc").textContent = vol.TC == null ? "—" : `${fmt(vol.TC, 1)} mL`;
  $("#vol-et").textContent = vol.ET == null ? "—" : `${fmt(vol.ET, 1)} mL`;

  const vp = resp.volume_pct || {};
  const pctText = (entry) => {
    if (!entry || entry.pct == null) return "";
    const base = entry.of === "brain" ? "占腦" : "占全腫瘤";
    const dp = entry.of === "brain" ? 2 : 0;
    return `${base} ${fmt(entry.pct, dp)}%`;
  };
  $("#pct-wt").textContent = pctText(vp.WT);
  $("#pct-tc").textContent = pctText(vp.TC);
  $("#pct-et").textContent = pctText(vp.ET);

  const conf = resp.confidence || {};
  const pct = (v) => (v == null ? "—" : `${Math.round(Number(v) * 100)}%`);
  $("#conf-wt").textContent = pct(conf.WT);
  $("#conf-tc").textContent = pct(conf.TC);
  $("#conf-et").textContent = pct(conf.ET);
  $("#conf-wt-desc").textContent = confLabel(conf.WT);
  $("#conf-tc-desc").textContent = confLabel(conf.TC);
  $("#conf-et-desc").textContent = confLabel(conf.ET);

  const flag = $("#conf-flag");
  const nameZh = { WT: "全腫瘤", TC: "腫瘤核心", ET: "強化腫瘤" };
  const low = ["WT", "TC", "ET"].filter(r => conf[r] != null && conf[r] < 0.70);
  if (low.length) {
    flag.textContent = `⚠ ${low.map(r => nameZh[r]).join("／")} 信心低於 0.70 — 請謹慎判讀這些區域。`;
    flag.classList.add("visible");
  } else {
    flag.textContent = "";
    flag.classList.remove("visible");
  }

  // 風險等級英文 → 中文（顯示用）
  const LEVEL_ZH = { "Low": "低", "Medium": "中", "High": "高", "Very High": "極高", "Unknown": "未知" };
  const wtRisk = (resp.risk && resp.risk.WT) || { level: "Unknown", color: "#94a3b8" };
  const badge = $("#risk-badge");
  badge.textContent = LEVEL_ZH[wtRisk.level] || wtRisk.level || "未知";
  badge.style.background = wtRisk.color || "#94a3b8";
  $("#risk-pct").textContent = wtRisk.percentile == null
    ? "無族群參考資料"
    : `第 ${wtRisk.percentile} 百分位 · 大於 ${wtRisk.percentile}% 的世代`;

  renderMalignancy(resp.malignancy);

  const ul = $("#anatomy-list");
  ul.innerHTML = "";
  if (!resp.anatomy_top || !resp.anatomy_top.length) {
    ul.innerHTML = '<li class="muted">此影像無法取得解剖資訊。</li>';
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

  renderEnergy(resp.energy);
  loadStats();

  $("#modality-select").disabled = false;
  $("#opacity").disabled = false;
  $("#download-btn").disabled = false;

  await loadViewers(resp);
  startUncertaintyPoll(resp);
}

// --- 影像惡性度傾向（描述性，非診斷） -----------------------------------
function renderMalignancy(m) {
  const badge = $("#mal-badge");
  const idxEl = $("#mal-index");
  const ul = $("#mal-drivers");
  ul.innerHTML = "";
  if (!m) {
    badge.textContent = "—";
    badge.style.background = "#94a3b8";
    idxEl.textContent = "等待預測";
    return;
  }
  badge.textContent = m.label_zh || "—";
  badge.style.background = m.color || "#94a3b8";
  idxEl.textContent = m.index == null
    ? "未偵測到腫瘤"
    : `影像指數 ${fmt(m.index, 2)}（0＝最低 · 1＝最高）`;
  for (const d of (m.drivers_zh || m.drivers || [])) {
    const li = document.createElement("li");
    li.textContent = d;
    ul.appendChild(li);
  }
}

// --- 永續：單次足跡 + 累積儀表板 -----------------------------------------
const nf = (v, d = 0) => (v == null ? "—" : Number(v).toLocaleString(undefined, {
  minimumFractionDigits: d, maximumFractionDigits: d,
}));

function renderEnergy(e) {
  if (!e) {
    $("#en-energy").textContent = "—";
    $("#en-co2").textContent = "—";
    $("#en-cost").textContent = "—";
    $("#en-method").textContent = "等待預測";
    $("#en-scale").textContent = "";
    $("#en-scale").classList.remove("visible");
    return;
  }
  $("#en-energy").textContent = `${fmt(e.energy_wh, 3)} Wh`;
  $("#en-co2").textContent = `${fmt(e.co2_g, 2)} g`;
  $("#en-cost").textContent = `NT$ ${fmt(e.cost_twd, 4)}`;
  $("#en-method").textContent = e.measured
    ? `實測 · ${e.backend_name || "GPU"} · 平均 ${fmt(e.mean_power_w, 0)} W，歷時 ${fmt(e.duration_s, 1)} 秒（${e.samples} 次取樣）`
    : "估算值 — 此主機無 GPU 功率遙測";

  const s = e.scale;
  if (s) {
    const perCaseH = e.manual_minutes_saved / 60;
    $("#en-scale").textContent =
      `若一間醫院每天 ${nf(s.cases_per_day)} 例，一年約耗 ${nf(s.energy_kwh, 1)} kWh` +
      `（≈${nf(s.co2_kg, 1)} kg CO₂ ≈ 開車 ${nf(s.equiv_car_km)} 公里），` +
      `同時省下約 ${nf(s.manual_hours_saved)} 小時工時 — 假設每例完整人工描繪約 ` +
      `${nf(perCaseH)} 小時（文獻 1–4 h）。`;
    $("#en-scale").classList.add("visible");
  }
}

async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    const s = await r.json();
    // 自動單位：單次推論僅約 0.05 Wh，用 kWh/kg 會顯示成「0.000」直到上萬筆。
    // 小量級先用 Wh/g，量大才切換成 kWh/kg。
    const wh = s.total_energy_wh || 0;
    const g = s.total_co2_g || 0;
    $("#stat-count").textContent = nf(s.total_inferences);
    $("#stat-energy").textContent = wh >= 1000 ? `${fmt(wh / 1000, 3)} kWh` : `${fmt(wh, 2)} Wh`;
    $("#stat-co2").textContent = g >= 1000 ? `${fmt(g / 1000, 3)} kg` : `${fmt(g, 2)} g`;
    $("#stat-hours").textContent = nf(s.total_manual_hours_saved, 1);
    if (s.total_inferences > 0) {
      $("#stat-foot").textContent =
        `碳排約等於開車 ${nf(s.equiv_car_km, 1)} 公里。自 ${s.since || "—"} 起 · ` +
        `電網 ${s.grid_co2_kg_per_kwh} kg/kWh · 遙測：${s.telemetry}。`;
    } else {
      $("#stat-foot").textContent = "完成第一筆分析即開始記錄。";
    }
  } catch (_) { /* 儀表板為盡力而為 */ }
}
loadStats();

const _resetBtn = document.getElementById("reset-stats-btn");
if (_resetBtn) {
  _resetBtn.addEventListener("click", async () => {
    if (!confirm("確定要將累積統計歸零嗎？此動作無法復原。")) return;
    _resetBtn.disabled = true;
    try {
      await fetch("/api/stats/reset", { method: "POST" });
      await loadStats();
    } catch (_) { /* 忽略 */ }
    finally { _resetBtn.disabled = false; }
  });
}

// --- 不確定性輪詢 --------------------------------------------------------
function startUncertaintyPoll(resp) {
  if (_uncPollTimer) clearInterval(_uncPollTimer);
  const statusUrl = resp.uncertainty_url;
  if (!statusUrl) return;

  // 立即顯示區塊並標示「計算中」
  const sec = $("#uncertainty-section");
  const badge = $("#uncertainty-badge");
  sec.style.display = "block";
  badge.textContent = "計算中…";
  badge.className = "unc-badge computing";

  _uncPollTimer = setInterval(async () => {
    try {
      const r = await fetch(statusUrl);
      const s = await r.json();
      if (s.ready && s.url) {
        clearInterval(_uncPollTimer);
        _uncPollTimer = null;
        badge.textContent = "完成";
        badge.className = "unc-badge ready";
        await loadUncertaintyViewer(resp.nifti_urls[baseModality], s.url);
      }
    } catch (_) { /* 忽略暫時的 fetch 錯誤 */ }
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
  // 區塊起始為 display:none，Niivue 一開始捕捉的是過時／近正方的 draw buffer，
  // 導致 3:1 montage 被縮成中央窄條。等版面穩定後，重跑 Niivue 自己的 resize
  // handler，讓 draw buffer 對齊整個（寬版）canvas，tile 才會填滿。
  const CANVAS_H = 460; // CSS 像素顯示高度；tile 會等比縮放填滿
  const syncSize = () => {
    const c = document.getElementById("nvunc-canvas");
    if (c) {
      const cssW = c.getBoundingClientRect().width || c.clientWidth;
      if (cssW > 0) {
        const dpr = window.devicePixelRatio || 1;
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

  // BraTS 標籤色彩對應（採用最對比的色相 — 黃 vs 藍為互補色，
  // 因此 ED 與 ET 在 3D 渲染中不會混色）：
  //   0 = 背景（透明）
  //   1 = NCR  → 紅 rgb(239, 68, 68)
  //   2 = ED   → 綠 rgb( 34,197, 94)
  //   3 = ET   → 藍 rgb( 29, 78,216)
  const labelColormap = {
    R: [0, 239, 34, 29],
    G: [0, 68, 197, 78],
    B: [0, 68, 94, 216],
    A: [0, 255, 255, 255],
    I: [0, 1, 2, 3],
    labels: ["BG", "NCR", "ED", "ET"],
  };
  if (!nv3d.addColormap) {
    // 較舊版本的 niivue
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
    // 將分割視為「離散」標籤圖。否則 Niivue 會對 seg 體積做三線性內插，
    // 把 ED(綠)↔ET(藍) 混成藍綠色，並把小塊 NCR(紅) 洗成橘色。
    const seg = nv.volumes[1];
    let labelOk = false;
    if (seg && seg.setColormapLabel) {
      try {
        seg.setColormapLabel(labelColormap);
        nv.updateGLVolume();
        labelOk = true;
      } catch (_) {}
    }
    // 較舊 Niivue 的後備方案：強制最近鄰，讓標籤保持純色。
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

// --- 報告下載 -------------------------------------------------------------
$("#download-btn").addEventListener("click", async () => {
  if (!lastResp) return;
  const sid = lastResp.session_id;
  $("#status").classList.remove("error");
  $("#status").textContent = "產生 PDF 報告中…";
  try {
    // 盡力截圖 — 若擷取失敗或空白也不可阻擋 PDF 下載。
    // 3D 立體渲染截圖。
    try {
      const blob = await canvasToBlob($("#nv3d-canvas"), nv3d);
      if (blob && blob.size > 0) {
        await fetch(`/api/session/${sid}/screenshot`, { method: "POST", body: blob });
      }
    } catch (e) { console.warn("screenshot skipped:", e); }
    // 切面檢視器（橫切／冠狀／矢狀 MRI 切片）截圖。
    try {
      const blob = await canvasToBlob($("#nvmpr-canvas"), nvMPR);
      if (blob && blob.size > 0) {
        await fetch(`/api/session/${sid}/slices`, { method: "POST", body: blob });
      }
    } catch (e) { console.warn("slices skipped:", e); }
    // 不確定性圖截圖（僅在 MC-Dropout 完成並渲染後）。
    try {
      if (nvUnc && $("#nvunc-canvas") && $("#nvunc-canvas").clientWidth > 0) {
        const blob = await canvasToBlob($("#nvunc-canvas"), nvUnc);
        if (blob && blob.size > 0) {
          await fetch(`/api/session/${sid}/uncertainty_shot`, { method: "POST", body: blob });
        }
      }
    } catch (e) { console.warn("uncertainty shot skipped:", e); }

    // 以 blob 取得 PDF，再用 anchor 觸發實際下載。
    const r = await fetch(`/api/session/${sid}/report`);
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const pdf = await r.blob();
    const url = URL.createObjectURL(pdf);
    const a = document.createElement("a");
    a.href = url;
    const pid = (lastResp.patient_id || sid.slice(0, 8)).replace(/[^A-Za-z0-9_.\-]/g, "_");
    a.download = `Report_${pid}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    $("#status").textContent = "報告已下載。";
  } catch (e) {
    console.error(e);
    $("#status").textContent = `下載失敗：${e.message || e}`;
    $("#status").classList.add("error");
  }
});

function canvasToBlob(canvas, nv = nv3d) {
  return new Promise(resolve => {
    try {
      // 強制重繪一幀，避免擷取時 WebGL buffer 為空。
      try { nv.drawScene(); } catch (_) {}
      requestAnimationFrame(() => {
        try { canvas.toBlob(b => resolve(b), "image/png"); }
        catch { resolve(null); }
      });
    } catch { resolve(null); }
  });
}
