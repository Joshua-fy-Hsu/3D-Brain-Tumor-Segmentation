"""Single integrated PDF report, styled like a radiology report.

Composes one A4 PDF from a session's metrics + clinical summary + the 3D /
slice / uncertainty screenshots, in English ("en") or Traditional Chinese
("zh"). Layout mimics a clinical report: title band, patient demographics
table, TECHNIQUE / FINDINGS / IMPRESSION sections, an electronically-generated
footer and a radiologist review/signature line.

Text uses a CJK TrueType font (Microsoft JhengHei on Windows, Noto CJK /
PingFang elsewhere); its Latin glyphs are clean so English shares it. Without
a CJK font it falls back to core Helvetica (Latin only). Override the font
with REPORT_CJK_FONT.
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

from fpdf import FPDF
from PIL import Image, ImageChops, ImageStat

from web import anatomy_desc as AD

# fpdf2 subsets the CJK font on every output() and logs verbosely — silence it.
logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

# ── Font discovery (once at import) ─────────────────────────────────────────
_CJK_REG = next((p for p in [
    os.environ.get("REPORT_CJK_FONT") or "",
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\mingliu.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
] if p and os.path.exists(p)), None)
_CJK_BOLD = next((p for p in [
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
] if os.path.exists(p)), None)
_USE_CJK = _CJK_REG is not None
_FAM = "body" if _USE_CJK else "helvetica"

# ── Palette (clinical navy + green accent for sustainability) ───────────────
NAVY = (23, 55, 84)
INK = (31, 41, 55)
MUTE = (107, 114, 128)
LINE = (210, 218, 228)
ACCENT = (37, 99, 235)
RED = (185, 28, 28)
GREEN = (5, 120, 90)
BAR_NAVY = (234, 239, 245)
BAR_GREEN = (236, 250, 244)
BOXBG = (247, 249, 252)
WHITE = (255, 255, 255)

_REGION = {
    "en": {"WT": "Whole Tumor", "TC": "Tumor Core", "ET": "Enhancing Tumor"},
    "zh": {"WT": "全腫瘤", "TC": "腫瘤核心", "ET": "強化腫瘤"},
}
_RISK_ZH = {"Low": "低", "Medium": "中", "High": "高",
            "Very High": "極高", "Unknown": "未知"}

L = {
    "en": {
        "title": "Brain Tumor Segmentation Report",
        "subtitle": "AI-Assisted Volumetric MRI Analysis",
        "tag": "RESEARCH USE ONLY",
        "patient": "Patient ID", "rdate": "Report Date",
        "modality": "Modality", "modality_v": "MRI: T1, T1CE, T2, FLAIR",
        "method": "Method", "method_v": "AI automated 3D segmentation",
        "model": "Model", "status": "Status", "status_v": "Research / non-diagnostic",
        "technique_h": "TECHNIQUE",
        "technique": "Multiparametric brain MRI (T1, T1CE, T2, FLAIR) analysed with a "
                     "deep-learning model for automated 3D segmentation of tumor sub-regions "
                     "(enhancing tumor, tumor core, whole tumor). No intravenous contrast was "
                     "administered as part of this analysis.",
        "vol_h": "QUANTITATIVE FINDINGS - TUMOR VOLUMES",
        "vol_intro": "Volume of affected tissue per BraTS region.",
        "of_brain": "of brain", "of_wt": "of whole tumor",
        "vd": {"WT": "All abnormal tissue (NCR + ED + ET)",
               "TC": "Necrotic core + enhancing tissue",
               "ET": "Active, contrast-enhancing region"},
        "loc_h": "ANATOMICAL LOCALIZATION",
        "loc_intro": "Brain regions the whole-tumor mask overlaps most (AAL3 atlas). "
                     "Share = fraction of the tumor inside that region.",
        "burden_h": "TUMOR BURDEN vs REFERENCE POPULATION",
        "burden_intro": "Whole-tumor volume vs 1251 BraTS reference patients. "
                        "<33rd pct = Low, to 67th = Medium, to 90th = High, above = Very High.",
        "risk_level": "Risk level", "percentile": "Percentile", "pct_tail": "th percentile",
        "mal_h": "IMAGING MALIGNANCY TENDENCY",
        "mal_intro": "Descriptive indicator from predicted enhancement & necrosis "
                     "fractions. NOT a tumour grade/stage and not validated against "
                     "pathology - for reference only.",
        "mal_index": "Imaging index",
        "conf_h": "SEGMENTATION CONFIDENCE",
        "conf_intro": "Mean predicted probability inside each region. Higher = more certain.",
        "unc_h": "PREDICTIVE UNCERTAINTY",
        "unc_intro": "Voxel-level predictive uncertainty (MC-Dropout, T=10 forward passes). "
                     "Brighter = the model is less certain; useful for flagging regions that "
                     "warrant closer review.",
        "summary_h": "SUMMARY",
        "imaging_h": "IMAGING",
        "render3d": "3D volume rendering", "slices": "Orthogonal slices (axial / coronal / sagittal)",
        "energy_h": "ENVIRONMENTAL FOOTPRINT & SUSTAINABILITY",
        "energy_intro": "Measured GPU-card energy for this inference (excludes CPU and PSU losses).",
        "gpu_energy": "GPU energy", "carbon": "Carbon", "cost": "Electricity cost",
        "power": "GPU power", "measured": "measured", "estimated": "estimated (no telemetry)",
        "saved": "Manual time saved", "saved_unit": "h",
        "saved_desc": "vs full manual 3D delineation (literature 1-4 h/case)",
        "review": "Findings require verification by a qualified radiologist.",
        "sign": "Reviewed by: ______________________     Date: ________________",
        "footer": "Electronically generated. AI-assisted research tool - not a substitute "
                  "for radiological interpretation.",
        "na": "n/a",
    },
    "zh": {
        "title": "腦腫瘤分割報告",
        "subtitle": "AI 輔助腦部 MRI 體積分析",
        "tag": "研究用途",
        "patient": "病患代號", "rdate": "報告日期",
        "modality": "影像模態", "modality_v": "MRI：T1、T1CE、T2、FLAIR",
        "method": "分析方法", "method_v": "AI 自動 3D 分割",
        "model": "模型", "status": "狀態", "status_v": "研究用途，非診斷依據",
        "technique_h": "檢查方法",
        "technique": "多參數腦部 MRI（T1、T1CE、T2、FLAIR）經深度學習模型自動進行腫瘤次區域"
                     "（強化腫瘤、腫瘤核心、全腫瘤）之 3D 分割。本分析未額外施打顯影劑。",
        "vol_h": "量化結果 - 腫瘤體積",
        "vol_intro": "各 BraTS 區域受影響組織的體積。",
        "of_brain": "占腦", "of_wt": "占全腫瘤",
        "vd": {"WT": "所有異常組織（NCR＋ED＋ET）",
               "TC": "壞死核心＋強化組織",
               "ET": "活躍、顯影劑強化的區域"},
        "loc_h": "解剖定位",
        "loc_intro": "全腫瘤遮罩重疊最多的腦區（AAL3 圖譜）。占比＝腫瘤落在該區域內的比例。",
        "burden_h": "腫瘤負荷 - 與族群比較",
        "burden_intro": "全腫瘤體積與 1251 名 BraTS 參考病患比較。低於第 33 百分位為低、"
                        "至第 67 為中、至第 90 為高、以上為極高。",
        "risk_level": "風險等級", "percentile": "百分位", "pct_tail": " 百分位",
        "mal_h": "影像惡性度傾向",
        "mal_intro": "由預測的強化腫瘤與壞死比例推導之描述性指標。並非腫瘤分級／分期，"
                     "且未經病理驗證，僅供參考。",
        "mal_index": "影像指數",
        "conf_h": "分割信心",
        "conf_intro": "各區域內的平均預測機率。數值越高，代表模型越有把握。",
        "unc_h": "預測不確定性",
        "unc_intro": "體素層級的預測不確定性（MC Dropout，T=10 次前向推論）。越亮代表模型越不確定，"
                     "可用於標記需要進一步檢視的區域。",
        "summary_h": "總結",
        "imaging_h": "影像",
        "render3d": "3D 立體渲染", "slices": "正交切面（橫切／冠狀／矢狀）",
        "energy_h": "環境足跡與永續",
        "energy_intro": "本次推論的顯卡實測耗電（不含 CPU 與電源供應器損耗）。",
        "gpu_energy": "顯卡耗電", "carbon": "碳排放", "cost": "電費",
        "power": "顯卡功率", "measured": "實測", "estimated": "估算（無遙測）",
        "saved": "省下人工工時", "saved_unit": "小時",
        "saved_desc": "相較完整人工 3D 描繪（文獻每例 1-4 小時）",
        "review": "本結果需由合格放射科醫師確認。",
        "sign": "判讀醫師簽名：______________________     日期：________________",
        "footer": "本報告由系統自動產生。AI 輔助研究工具，不可取代放射科醫師之判讀。",
        "na": "無",
    },
}


def _n(v, d=2):
    try:
        return f"{float(v):,.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _c(s):
    return (str(s).replace("—", "-").replace("–", "-")
            .replace("≈", "~").replace("　", " "))


def _trim(path):
    """Crop near-black borders/empty panels off a screenshot so the brain
    fills the figure (the WebGL captures have large black margins, and an
    un-rendered montage panel is solid black). Returns a PIL image, or the
    original path on any failure (fpdf2 accepts either)."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return path
    try:
        gray = im.convert("L")
        mask = gray.point(lambda p: 255 if p > 18 else 0)
        bbox = mask.getbbox()
        if not bbox:
            return im
        pad = 6
        l, t, r, b = bbox
        crop = (max(0, l - pad), max(0, t - pad),
                min(im.width, r + pad), min(im.height, b + pad))
        return im.crop(crop)
    except Exception:
        return im


def _imsize(img):
    if hasattr(img, "size"):
        return img.size
    try:
        with Image.open(img) as im:
            return im.size
    except Exception:
        return (4, 3)


def _is_blank(img):
    """True if the image is near-uniform (e.g. a failed/empty WebGL capture)."""
    try:
        g = img.convert("L") if hasattr(img, "convert") else Image.open(img).convert("L")
        return ImageStat.Stat(g).stddev[0] < 4.0
    except Exception:
        return False


def _cert(v, lang):
    if v is None:
        return ""
    v = float(v)
    if lang == "zh":
        return ("信心極高" if v >= .9 else "信心高" if v >= .8
                else "信心中等" if v >= .7 else "信心低 - 請謹慎判讀")
    return ("Very high certainty" if v >= .9 else "High certainty" if v >= .8
            else "Moderate certainty" if v >= .7 else "Low - interpret carefully")


class _PDF(FPDF):
    footer_text = ""

    def __init__(self):
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(True, margin=16)
        self.set_margins(16, 14, 16)
        if _USE_CJK:
            self.add_font("body", "", _CJK_REG)
            self.add_font("body", "B", _CJK_BOLD or _CJK_REG)

    def f(self, size, bold=False):
        self.set_font(_FAM, "B" if bold else "", size)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(*LINE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(1)
        self.f(7)
        self.set_text_color(*MUTE)
        self.cell(self.epw - 18, 5, _c(self.footer_text))
        self.cell(18, 5, str(self.page_no()), align="R")

    # title band drawn manually on page 1
    def title_band(self, t):
        h = 19
        self.set_fill_color(*NAVY)
        self.rect(0, 0, self.w, h, "F")
        self.set_xy(self.l_margin, 4)
        self.f(16, True)
        self.set_text_color(*WHITE)
        self.cell(0, 7, _c(t["title"]))
        self.set_xy(self.l_margin, 11.5)
        self.f(9)
        self.set_text_color(220, 228, 238)
        self.cell(0, 4.5, _c(t["subtitle"]))
        # right tag
        self.f(8, True)
        self.set_text_color(*WHITE)
        self.set_xy(self.w - self.r_margin - 50, 7)
        self.cell(50, 5, _c(t["tag"]), align="R")
        self.set_y(h + 4)
        self.set_text_color(*INK)

    def demographics(self, fields):
        x0, y0, w = self.l_margin, self.get_y(), self.epw
        cols, ch = 2, 11.5
        rows = (len(fields) + 1) // 2
        cw = w / cols
        h = rows * ch
        self.set_draw_color(*LINE)
        self.set_fill_color(*BOXBG)
        self.rect(x0, y0, w, h, "DF")
        for r in range(1, rows):
            self.line(x0, y0 + r * ch, x0 + w, y0 + r * ch)
        self.line(x0 + cw, y0, x0 + cw, y0 + h)
        for i, (lab, val) in enumerate(fields):
            r, c = i // 2, i % 2
            cx, cy = x0 + c * cw + 3, y0 + r * ch + 1.8
            self.set_xy(cx, cy)
            self.f(7)
            self.set_text_color(*MUTE)
            self.cell(cw - 6, 3.6, _c(lab.upper()))
            self.set_xy(cx, cy + 3.9)
            self.f(10, True)
            self.set_text_color(*INK)
            self.cell(cw - 6, 5, _c(val))
        self.set_y(y0 + h + 3)

    def section(self, title, intro=None, tone="navy"):
        self.ln(2.5)
        x0, y, w, bh = self.l_margin, self.get_y(), self.epw, 8.4
        bar = BAR_GREEN if tone == "green" else BAR_NAVY
        col = GREEN if tone == "green" else NAVY
        self.set_fill_color(*bar)
        self.rect(x0, y, w, bh, "F")
        self.set_fill_color(*col)
        self.rect(x0, y, 2.0, bh, "F")           # left accent bar
        self.set_xy(x0 + 4, y)
        self.f(13, True)
        self.set_text_color(*col)
        self.cell(0, bh, _c(title.upper()))
        self.set_y(y + bh + 2.2)
        if intro:
            self.f(8.5)
            self.set_text_color(*MUTE)
            self.multi_cell(self.epw, 4.5, _c(intro))
            self.ln(0.5)
        self.set_text_color(*INK)

    def row3(self, name, desc, value):
        self.f(10.5, True)
        self.set_text_color(*INK)
        self.cell(44, 7, _c(name))
        self.f(8.5)
        self.set_text_color(*MUTE)
        self.cell(self.epw - 44 - 30, 7, _c(desc))
        self.f(11, True)
        self.set_text_color(*INK)
        self.cell(30, 7, _c(value), align="R", new_x="LMARGIN", new_y="NEXT")

    def anat_row(self, d, pct):
        self.f(10.5, True)
        self.set_text_color(*INK)
        self.cell(self.epw - 20, 5.8, _c(f"{d['title']}   ·   {d['lobe']}"))
        self.f(10.5, True)
        self.cell(20, 5.8, f"{pct}%", align="R", new_x="LMARGIN", new_y="NEXT")
        self.f(8.5)
        self.set_text_color(*MUTE)
        self.cell(0, 4.8, _c(f"{d['role']}    ({d['code']})"),
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(1.3)

    def need(self, h):
        """Start a new page if less than `h` mm remains, so a section header
        is never orphaned at the bottom with its image on the next page."""
        if self.get_y() + h > self.h - 16:
            self.add_page()

    def draw_image(self, img, caption, w):
        w = min(w, self.epw)
        try:
            x = self.l_margin + (self.epw - w) / 2
            self.image(img, x=x, w=w)
        except Exception:
            return
        self.f(8.5)
        self.set_text_color(*MUTE)
        self.cell(0, 5, _c(caption), align="C", new_x="LMARGIN", new_y="NEXT")

    def figure(self, path, caption, w):
        self.draw_image(_trim(path), caption, w)


def build_pdf(metrics: dict, summary_text: str, lang: str = "en",
              screenshot_path: Optional[str] = None,
              slices_path: Optional[str] = None,
              uncertainty_path: Optional[str] = None) -> bytes:
    lang = lang if lang in L else "en"
    t = L[lang]
    rn = _REGION[lang]
    pdf = _PDF()
    pdf.footer_text = t["footer"]
    pdf.add_page()
    pdf.title_band(t)

    # ── Demographics ────────────────────────────────────────────────────
    patient = metrics.get("patient_id") or t["na"]
    model = metrics.get("model_name") or metrics.get("run_name", "")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.demographics([
        (t["patient"], str(patient)), (t["rdate"], now),
        (t["modality"], t["modality_v"]), (t["method"], t["method_v"]),
        (t["model"], str(model)), (t["status"], t["status_v"]),
    ])

    # ── Summary (lead) ──────────────────────────────────────────────────
    if summary_text:
        pdf.section(t["summary_h"])
        pdf.f(10)
        pdf.set_text_color(*INK)
        pdf.multi_cell(pdf.epw, 5.7, _c(summary_text.strip()))

    # ── Volumes ─────────────────────────────────────────────────────────
    vols = metrics.get("volumes_ml", {}) or {}
    vpct = metrics.get("volume_pct", {}) or {}
    pdf.section(t["vol_h"], t["vol_intro"])
    for r in ("WT", "TC", "ET"):
        v = vols.get(r)
        desc = t["vd"][r]
        entry = vpct.get(r) or {}
        if entry.get("pct") is not None:
            base = t["of_brain"] if entry.get("of") == "brain" else t["of_wt"]
            dp = 2 if entry.get("of") == "brain" else 0
            if lang == "zh":
                desc = f"{desc}  ·  {base} {_n(entry['pct'], dp)}%"
            else:
                desc = f"{desc}  ·  {_n(entry['pct'], dp)}% {base}"
        pdf.row3(rn[r], desc, f"{_n(v, 1)} mL" if v is not None else t["na"])

    # ── Localization ────────────────────────────────────────────────────
    anatomy = metrics.get("anatomy_top") or []
    if anatomy:
        pdf.section(t["loc_h"], t["loc_intro"])
        for a in anatomy:
            d = AD.describe(str(a.get("name", "")), lang)
            pdf.anat_row(d, a.get("pct", 0))

    # ── Burden vs population ────────────────────────────────────────────
    risk = (metrics.get("risk") or {}).get("WT") or {}
    if risk:
        pdf.section(t["burden_h"], t["burden_intro"])
        lvl = risk.get("level", "Unknown")
        if lang == "zh":
            lvl = _RISK_ZH.get(lvl, lvl)
        pdf.row3(t["risk_level"], "", str(lvl))
        pct = risk.get("percentile")
        pdf.row3(t["percentile"], "",
                 f"{pct}{t['pct_tail']}" if pct is not None else t["na"])

    # ── Imaging malignancy tendency (descriptive, non-diagnostic) ───────
    mal = metrics.get("malignancy") or {}
    if mal and mal.get("category") not in (None, "unknown"):
        pdf.section(t["mal_h"], t["mal_intro"])
        label = mal.get("label_zh") if lang == "zh" else mal.get("label")
        idx = mal.get("index")
        pdf.row3(label or "", t["mal_index"],
                 _n(idx, 2) if idx is not None else t["na"])
        drivers = (mal.get("drivers_zh") if lang == "zh" else mal.get("drivers")) or []
        for d in drivers:
            pdf.f(8.5)
            pdf.set_text_color(*MUTE)
            pdf.multi_cell(pdf.epw, 4.5, _c("- " + d))
        disc = mal.get("disclaimer_zh") if lang == "zh" else mal.get("disclaimer")
        if disc:
            pdf.ln(0.5)
            pdf.f(8.5)
            pdf.set_text_color(*MUTE)
            pdf.multi_cell(pdf.epw, 4.5, _c(disc))
        pdf.set_text_color(*INK)

    # ── Confidence ──────────────────────────────────────────────────────
    conf = metrics.get("confidence", {}) or {}
    pdf.section(t["conf_h"], t["conf_intro"])
    for r in ("WT", "TC", "ET"):
        c = conf.get(r)
        pdf.row3(rn[r], _cert(c, lang),
                 f"{round(float(c) * 100)}%" if c is not None else t["na"])

    # ── Predictive uncertainty (image) ──────────────────────────────────
    # Skip silently if the capture came back blank (failed WebGL grab) so we
    # never print an orphan header above an empty page.
    if uncertainty_path and os.path.exists(uncertainty_path):
        unc_img = _trim(uncertainty_path)
        if not _is_blank(unc_img):
            pdf.need(86)
            pdf.section(t["unc_h"], t["unc_intro"])
            pdf.draw_image(unc_img, t["unc_h"], min(pdf.epw, 170))

    # ── Imaging figures ─────────────────────────────────────────────────
    figs = [(screenshot_path, t["render3d"]), (slices_path, t["slices"])]
    figs = [(p, cap) for p, cap in figs if p and os.path.exists(p)]
    if figs:
        pdf.need(95)
        pdf.section(t["imaging_h"])
        if len(figs) == 2:
            colw = (pdf.epw - 6) / 2
            max_h = 58.0          # cap height so the square 3D render isn't huge
            y0 = pdf.get_y()
            maxh = 0.0
            for i, (p, cap) in enumerate(figs):
                img = _trim(p)
                iw, ih = _imsize(img)
                dw, dh = colw, colw * ih / iw
                if dh > max_h:     # too tall: shrink to the height cap, keep centred
                    dh, dw = max_h, max_h * iw / ih
                cx = pdf.l_margin + i * (colw + 6) + (colw - dw) / 2
                try:
                    pdf.image(img, x=cx, y=y0, w=dw)
                    maxh = max(maxh, dh)
                except Exception:
                    continue
            pdf.set_y(y0 + maxh + 1.5)
            pdf.f(8.5)
            pdf.set_text_color(*MUTE)
            pdf.cell(colw + 6, 5, _c(figs[0][1]), align="C")
            pdf.cell(0, 5, _c(figs[1][1]), align="C", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.figure(figs[0][0], figs[0][1], min(pdf.epw, 110))

    # ── Environmental footprint ─────────────────────────────────────────
    energy = metrics.get("energy")
    if energy:
        pdf.section(t["energy_h"], t["energy_intro"], tone="green")
        method = t["measured"] if energy.get("measured") else t["estimated"]
        pdf.row3(t["gpu_energy"], "", f"{_n(energy.get('energy_wh'), 3)} Wh")
        pdf.row3(t["carbon"], "", f"{_n(energy.get('co2_g'), 2)} g")
        pdf.row3(t["cost"], "", f"NT$ {_n(energy.get('cost_twd'), 4)}")
        pdf.row3(t["power"], f"{method} ({energy.get('backend_name', 'GPU')})",
                 f"{_n(energy.get('mean_power_w'), 0)} W")
        per_h = energy.get("manual_minutes_saved", 0) / 60.0
        pdf.row3(t["saved"], t["saved_desc"], f"{_n(per_h, 1)} {t['saved_unit']}")
        sc = energy.get("scale") or {}
        if sc:
            if lang == "zh":
                line = (f"若醫院每天 {_n(sc.get('cases_per_day'), 0)} 例，一年約耗 "
                        f"{_n(sc.get('energy_kwh'), 1)} kWh（約 {_n(sc.get('co2_kg'), 1)} kg CO2、"
                        f"約開車 {_n(sc.get('equiv_car_km'), 0)} 公里），同時省下約 "
                        f"{_n(sc.get('manual_hours_saved'), 0)} 小時人工描繪"
                        f"（假設每例約 {_n(per_h, 0)} 小時，文獻 1-4 h）。")
            else:
                line = (f"At {_n(sc.get('cases_per_day'), 0)} scans/day a hospital would use "
                        f"~{_n(sc.get('energy_kwh'), 1)} kWh/yr (~{_n(sc.get('co2_kg'), 1)} kg CO2 "
                        f"~ {_n(sc.get('equiv_car_km'), 0)} km driving) while saving "
                        f"~{_n(sc.get('manual_hours_saved'), 0)} h of manual segmentation "
                        f"(assumes ~{_n(per_h, 0)} h/case, literature 1-4 h).")
            pdf.ln(1)
            pdf.set_fill_color(*BAR_GREEN)
            pdf.f(9)
            pdf.set_text_color(*GREEN)
            pdf.multi_cell(pdf.epw, 5.3, _c(line), fill=True)
            pdf.set_text_color(*INK)

    # ── Review / signature ──────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.f(8.5, True)
    pdf.set_text_color(*RED)
    pdf.multi_cell(pdf.epw, 4.8, _c(t["review"]))
    pdf.ln(1)
    pdf.f(9.5)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6, _c(t["sign"]), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
