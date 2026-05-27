"""TumorSeg 競賽簡報用 — 把現成的 qualitative_success.png 中文化。

不需要重新跑推論：在現有 PNG 上覆蓋中文標題與案例說明。

Output: docs/report_figures/qualitative_success_zh.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "docs" / "figures" / "qualitative_success.png"
OUT  = ROOT / "docs" / "report_figures" / "qualitative_success_zh.png"

FONT_BOLD = "C:/Windows/Fonts/msjhbd.ttc"  # Microsoft JhengHei Bold

img = Image.open(SRC).convert("RGB")
W, H = img.size  # 1585 x 1117
draw = ImageDraw.Draw(img)

font_title = ImageFont.truetype(FONT_BOLD, int(H * 0.030))
font_case  = ImageFont.truetype(FONT_BOLD, int(H * 0.024))


def cover(x0_frac, y0_frac, x1_frac, y1_frac):
    draw.rectangle([int(W * x0_frac), int(H * y0_frac),
                    int(W * x1_frac), int(H * y1_frac)],
                   fill="white")


def centered_text(text, xc_frac, y_top, font, color="#222"):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((int(W * xc_frac) - w // 2, y_top), text, font=font, fill=color)


# === Row 1: case label + column titles ===
cover(0, 0.0, 1.0, 0.13)

centered_text(
    "患者 BraTS2021_01418  |  平均 Dice 0.92  |  腫瘤體積 187,569 voxels",
    0.50, int(H * 0.010), font_case
)

col_x = [0.178, 0.500, 0.823]
col_titles = ["T1CE (輸入)", "標準答案", "TumorSeg 預測"]
for cx, t in zip(col_x, col_titles):
    centered_text(t, cx, int(H * 0.080), font_title)

# === Case 2 label between the two rows ===
# It sits roughly y 49% – 53%
cover(0, 0.480, 1.0, 0.540)
centered_text(
    "患者 BraTS2021_01593  |  平均 Dice 0.99  |  腫瘤體積 77,802 voxels",
    0.50, int(H * 0.495), font_case
)

img.save(OUT, dpi=(200, 200))
print(f"wrote {OUT}")
