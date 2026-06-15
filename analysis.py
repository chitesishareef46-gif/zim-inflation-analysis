"""
Zimbabwe Inflation Analysis & Forecasting
==========================================
Author  : Shareef Chitesi
Degree  : BSc Honours Applied Mathematics & Computational Science
          Midlands State University — Year 2
Project : Statistical analysis and time series forecasting of
          Zimbabwe's inflation rate (2010–2024) using Python.
Data    : World Bank / ZIMSTAT / RBZ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE    = "#1B4F72"
RED     = "#C0392B"
ORANGE  = "#E67E22"
GREEN   = "#1E8449"
GREY    = "#808B96"
LIGHT   = "#D6EAF8"

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "figure.dpi"      : 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA  (annual CPI inflation %, World Bank / RBZ / ZIMSTAT)
# ─────────────────────────────────────────────────────────────────────────────
data = {
    "Year": list(range(2010, 2026)),
    "Inflation_pct": [
        3.2,    # 2010 — post dollarisation stability
        3.5,    # 2011
        3.7,    # 2012
        1.6,    # 2013
        -0.2,   # 2014 — deflation
        -2.4,   # 2015 — deflation
        -1.6,   # 2016
        0.9,    # 2017
        10.6,   # 2018 — RTGS dollar introduced
        255.3,  # 2019 — ZWL introduced, hyperinflation
        557.2,  # 2020 — peak hyperinflation
        98.5,   # 2021 — some stabilisation
        104.7,  # 2022
        87.4,   # 2023
        47.6,   # 2024 — ZiG introduced
        30.0,   # 2025 — estimate (ZIMSTAT April 2026: 1.06% monthly)
    ],
    "Currency": [
        "USD","USD","USD","USD","USD","USD","USD","USD",
        "RTGS/ZWL","ZWL","ZWL","ZWL","ZWL","ZWL","ZiG","ZiG"
    ],
    "Event": [
        "Post-dollarisation","","","","Deflation","Deflation","","",
        "RTGS introduced","ZWL introduced","Peak hyperinflation",
        "Stabilisation","","","ZiG introduced","ZiG stabilising"
    ]
}

df = pd.DataFrame(data)
df["log_inflation"] = np.where(df["Inflation_pct"] > 0,
                               np.log(df["Inflation_pct"] + 1), 0)

print("=" * 60)
print("  ZIMBABWE INFLATION ANALYSIS — Shareef Chitesi")
print("=" * 60)
print(df[["Year","Inflation_pct","Currency"]].to_string(index=False))
print()
print("── Summary Statistics ──────────────────────────────────")
print(f"  Mean inflation (2010–2025) : {df.Inflation_pct.mean():.1f}%")
print(f"  Median                     : {df.Inflation_pct.median():.1f}%")
print(f"  Peak                       : {df.Inflation_pct.max():.1f}% ({df.loc[df.Inflation_pct.idxmax(),'Year']})")
print(f"  Lowest                     : {df.Inflation_pct.min():.1f}% ({df.loc[df.Inflation_pct.idxmin(),'Year']})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIMPLE FORECASTING  (linear regression on log scale for 2026–2028)
# ─────────────────────────────────────────────────────────────────────────────
# Use last 5 years (stabilisation period) for forecast
recent = df[df["Year"] >= 2021].copy()
x = recent["Year"].values
y = recent["Inflation_pct"].values

# Fit linear trend on recent data
coeffs = np.polyfit(x, y, 1)
trend  = np.poly1d(coeffs)

forecast_years  = np.array([2026, 2027, 2028])
forecast_values = np.clip(trend(forecast_years), 5, None)

# Confidence band (±20% of forecast)
upper = forecast_values * 1.20
lower = np.clip(forecast_values * 0.80, 0, None)

print("── Inflation Forecast (Linear Trend on 2021–2025) ─────")
for yr, fv, lo, hi in zip(forecast_years, forecast_values, lower, upper):
    print(f"  {yr}: {fv:.1f}%  (range: {lo:.1f}% – {hi:.1f}%)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PLOT 1 — Full historical inflation (bar + line)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

colors = [RED if v > 100 else ORANGE if v > 20 else
          GREEN if v < 0 else BLUE for v in df["Inflation_pct"]]

bars = ax.bar(df["Year"], df["Inflation_pct"], color=colors,
              alpha=0.85, width=0.7, zorder=3)
ax.plot(df["Year"], df["Inflation_pct"], color=BLUE,
        linewidth=1.5, marker="o", markersize=4, zorder=4)

# Annotate key events
annotations = {
    2019: ("ZWL\nintroduced", 0.5),
    2020: ("Peak\nhyperinflation", 0.5),
    2024: ("ZiG\nintroduced", 0.5),
}
for yr, (txt, offset) in annotations.items():
    val = df.loc[df["Year"] == yr, "Inflation_pct"].values[0]
    ax.annotate(txt, xy=(yr, val), xytext=(yr + offset, val + 30),
                fontsize=7.5, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

ax.set_title("Zimbabwe Annual Inflation Rate (2010–2025)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Inflation Rate (%)", fontsize=11)
ax.set_xticks(df["Year"])
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}%"))

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=RED,    label="Hyperinflation (>100%)"),
    Patch(facecolor=ORANGE, label="High inflation (20–100%)"),
    Patch(facecolor=BLUE,   label="Moderate inflation"),
    Patch(facecolor=GREEN,  label="Deflation"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/01_historical_inflation.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 1 saved: Historical Inflation")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOT 2 — Log-scale inflation (better visibility across eras)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
pos_df  = df[df["Inflation_pct"] > 0]
neg_df  = df[df["Inflation_pct"] <= 0]

ax.semilogy(pos_df["Year"], pos_df["Inflation_pct"],
            color=BLUE, linewidth=2, marker="o", markersize=6, label="Inflation (log scale)")
ax.scatter(neg_df["Year"], [0.1]*len(neg_df), color=GREEN,
           zorder=5, s=60, label="Deflation years")

ax.axhline(y=10, color=ORANGE, linestyle="--", alpha=0.6, label="10% threshold")
ax.axhline(y=100, color=RED,   linestyle="--", alpha=0.6, label="100% threshold")

ax.set_title("Zimbabwe Inflation — Log Scale View (2010–2025)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Inflation Rate (%, log scale)", fontsize=11)
ax.set_xticks(df["Year"])
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}%"))

plt.tight_layout()
plt.savefig("outputs/02_log_scale_inflation.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 2 saved: Log-Scale Inflation")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOT 3 — Recent stabilisation + forecast (2021–2028)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

recent_plot = df[df["Year"] >= 2021]
ax.plot(recent_plot["Year"], recent_plot["Inflation_pct"],
        color=BLUE, linewidth=2.5, marker="o", markersize=7,
        label="Actual inflation")

ax.plot(forecast_years, forecast_values,
        color=ORANGE, linewidth=2.5, marker="s", markersize=7,
        linestyle="--", label="Forecast (linear trend)")

ax.fill_between(forecast_years, lower, upper,
                color=ORANGE, alpha=0.15, label="Confidence band (±20%)")

# Shade forecast region
ax.axvspan(2025.5, 2028.5, color=LIGHT, alpha=0.4, label="Forecast period")

for yr, fv in zip(forecast_years, forecast_values):
    ax.annotate(f"{fv:.1f}%", xy=(yr, fv), xytext=(0, 12),
                textcoords="offset points", ha="center",
                fontsize=9, color=ORANGE, fontweight="bold")

ax.set_title("Zimbabwe Inflation: Stabilisation & Forecast (2021–2028)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Inflation Rate (%)", fontsize=11)
ax.set_xticks(list(recent_plot["Year"]) + list(forecast_years))
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("outputs/03_forecast.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 3 saved: Forecast")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOT 4 — Era comparison (box/summary by currency era)
# ─────────────────────────────────────────────────────────────────────────────
era_map = {
    "USD" : "Dollarisation\n(2010–2018)",
    "RTGS/ZWL": "Dollarisation\n(2010–2018)",
    "ZWL" : "ZWL Era\n(2019–2023)",
    "ZiG" : "ZiG Era\n(2024–2025)",
}
df["Era"] = df["Currency"].map(era_map)
era_order = ["Dollarisation\n(2010–2018)", "ZWL Era\n(2019–2023)", "ZiG Era\n(2024–2025)"]
era_colors = [GREEN, RED, BLUE]

fig, ax = plt.subplots(figsize=(9, 5))
era_means = df.groupby("Era")["Inflation_pct"].mean().reindex(era_order)
era_maxes = df.groupby("Era")["Inflation_pct"].max().reindex(era_order)
era_mins  = df.groupby("Era")["Inflation_pct"].min().reindex(era_order)

x_pos = np.arange(len(era_order))
bars = ax.bar(x_pos, era_means, color=era_colors, alpha=0.8,
              width=0.5, zorder=3, label="Mean inflation")
ax.errorbar(x_pos, era_means,
            yerr=[era_means - era_mins, era_maxes - era_means],
            fmt="none", color="black", capsize=6, linewidth=1.5, zorder=4)

for i, (mean, mx) in enumerate(zip(era_means, era_maxes)):
    ax.text(i, mean + 10, f"Mean: {mean:.1f}%\nPeak: {mx:.1f}%",
            ha="center", va="bottom", fontsize=8.5, color="black")

ax.set_xticks(x_pos)
ax.set_xticklabels(era_order, fontsize=10)
ax.set_title("Zimbabwe Inflation by Currency Era — Mean & Range",
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Inflation Rate (%)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}%"))

plt.tight_layout()
plt.savefig("outputs/04_era_comparison.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 4 saved: Era Comparison")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  EXPORT DATA TO EXCEL (for Power BI import)
# ─────────────────────────────────────────────────────────────────────────────
with pd.ExcelWriter("outputs/zimbabwe_inflation_data.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Historical Data", index=False)

    forecast_df = pd.DataFrame({
        "Year"          : forecast_years,
        "Forecast_pct"  : forecast_values,
        "Lower_bound"   : lower,
        "Upper_bound"   : upper,
        "Type"          : ["Forecast"] * 3
    })
    forecast_df.to_excel(writer, sheet_name="Forecast", index=False)

    summary = pd.DataFrame({
        "Metric": ["Mean Inflation", "Median", "Peak Year", "Peak Value",
                   "Lowest Year", "Lowest Value", "Dollarisation Mean",
                   "ZWL Era Mean", "ZiG Era Mean"],
        "Value": [
            f"{df.Inflation_pct.mean():.1f}%",
            f"{df.Inflation_pct.median():.1f}%",
            str(df.loc[df.Inflation_pct.idxmax(), 'Year']),
            f"{df.Inflation_pct.max():.1f}%",
            str(df.loc[df.Inflation_pct.idxmin(), 'Year']),
            f"{df.Inflation_pct.min():.1f}%",
            f"{df[df['Era']=='Dollarisation\\n(2010–2018)'].Inflation_pct.mean():.1f}%",
            f"{df[df['Era']=='ZWL Era\\n(2019–2023)'].Inflation_pct.mean():.1f}%",
            f"{df[df['Era']=='ZiG Era\\n(2024–2025)'].Inflation_pct.mean():.1f}%",
        ]
    })
    summary.to_excel(writer, sheet_name="Summary Statistics", index=False)

print("  ✓ Excel data exported: zimbabwe_inflation_data.xlsx")
print()
print("=" * 60)
print("  All outputs saved to /outputs/ folder")
print("  Import zimbabwe_inflation_data.xlsx into Power BI")
print("=" * 60)
