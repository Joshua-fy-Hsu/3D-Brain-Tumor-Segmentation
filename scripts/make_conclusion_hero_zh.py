"""TumorSeg 競賽簡報用 — 結論代表性案例 (中文化)

把現成的 conclusion_hero.png 上的英文標題覆蓋成中文。
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "docs" / "figures" / "conclusion_hero.png"
OUT  = ROOT / "docs" / "report_figures" / "conclusion_hero_zh.png"

FONT_BOLD = "C:/Windows/Fonts/msjhbd.ttc"

img = Image.open(SRC).convert("RGB")
W, H = img.size  # 2451 x 762
draw = ImageDraw.Draw(img)

font_sup   = ImageFont.truetype(FONT_BOLD, int(H * 0.045))
font_title = ImageFont.truetype(FONT_BOLD, int(H * 0.060))


def cover(x0f, y0f, x1f, y1f):
    draw.rectangle([int(W * x0f), int(H * y0f),
                    int(W * x1f), int(H * y1f)], fill="white")


def centered(text, xc_f, y_top, font, color="#222"):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((int(W * xc_f) - w // 2, y_top), text, font=font, fill=color)


# 1) Suptitle band (top ~5%)
cover(0, 0, 1.0, 0.07)
centered(
    "代表性案例 BraTS2021_01418 — TumorSeg 預測與專家標註高度一致，不確定性熱圖標出邊界風險區",
    0.50, int(H * 0.015), font_sup
)

# 2) Column titles band — generous cover to wipe original English titles
cover(0, 0.07, 1.0, 0.22)

col_x = [0.130, 0.380, 0.625, 0.870]
titles = ["T1CE (輸入)", "標準答案", "TumorSeg 預測", "不確定性"]
for cx, t in zip(col_x, titles):
    centered(t, cx, int(H * 0.110), font_title)

img.save(OUT, dpi=(200, 200))
print(f"wrote {OUT}")
