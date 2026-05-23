"""Opening motivation slide — brain-tumor burden in Taiwan (two panels).

Left  : annual new malignant brain-tumor cases (5-year-period total / 5).
Right : observed survival 1->10 years after diagnosis.

Source: Taiwan Cancer Registry (HPA) interactive query, cancer site = 腦 (brain,
malignant), all ages, both sexes, nationwide. Incidence is the 5-year-period
total divided by 5; survival is the 2012-2021 cohort observed survival rate.

Output: docs/report_figures/taiwan_epidemiology.png
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "docs" / "report_figures"

BLUE = "#3b6fb5"
RED = "#dc2929"

# ---- left: incidence ----
PERIODS = ["1979–83", "1984–88", "1989–93", "1994–98", "1999–2003",
           "2004–08", "2009–13", "2014–18", "2019–23"]
TOTALS = [1156, 1564, 2189, 2549, 2959, 3086, 3540, 3727, 3752]
PER_YEAR = [round(t / 5) for t in TOTALS]
inc_colors = [BLUE] * (len(PER_YEAR) - 1) + [RED]

# ---- right: observed survival (year 0 = diagnosis = 100%) ----
SURV_YR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SURV = [100, 67.4, 47.2, 38.0, 33.1, 30.2, 27.6, 26.1, 24.4, 23.0, 22.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.7),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# === left panel ===
bars = ax1.bar(PERIODS, PER_YEAR, color=inc_colors, width=0.7,
               edgecolor="white", linewidth=1.0)
ax1.set_ylabel("New cases per year", fontsize=11.5)
ax1.set_ylim(0, 880)
for b, v in zip(bars, PER_YEAR):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 14,
             f"{v}", ha="center", va="bottom",
             fontsize=8.8, fontweight="bold", color="#1f2937")
ax1.text(0.6, 775, "×3.2 over 40 years", ha="left", va="bottom",
         fontsize=11.5, fontweight="bold", color=RED)
ax1.set_title("Diagnoses keep rising  (~750 a year)",
              fontsize=12.5, fontweight="bold", pad=10)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(axis="x", labelsize=9, rotation=40)
for lbl in ax1.get_xticklabels():
    lbl.set_ha("right")
ax1.tick_params(axis="y", labelsize=10)

# === right panel ===
ax2.plot(SURV_YR, SURV, color=BLUE, lw=2.6, marker="o",
         markersize=5, markerfacecolor=BLUE, markeredgecolor="white")
ax2.fill_between(SURV_YR, SURV, color=BLUE, alpha=0.10)
# highlight 5-year point
ax2.plot(5, 30.2, "o", markersize=10, color=RED, zorder=5)
ax2.annotate("5-year\n30%", xy=(5, 30.2), xytext=(5.7, 47),
             fontsize=11.5, fontweight="bold", color=RED,
             ha="left", va="center")
ax2.set_xlim(0, 10.3)
ax2.set_ylim(0, 105)
ax2.set_xticks(range(0, 11, 2))
ax2.set_xlabel("Years after diagnosis", fontsize=11.5)
ax2.set_ylabel("Patients surviving (%)", fontsize=11.5)
ax2.set_title("Most do not survive five years",
              fontsize=12.5, fontweight="bold", pad=10)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(labelsize=10)

fig.text(0.5, 0.005,
         "Source: Taiwan Cancer Registry (HPA) · malignant brain tumors · "
         "incidence = annual avg per 5-year period · survival = 2012–2021 cohort, observed",
         ha="center", va="bottom", fontsize=9, color="#6b7280")

fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.28, left=0.07, right=0.97)
fig.savefig(OUT / "taiwan_epidemiology.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print(f"wrote {OUT / 'taiwan_epidemiology.png'}")
