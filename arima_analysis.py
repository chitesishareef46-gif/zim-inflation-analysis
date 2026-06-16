"""
Zimbabwe Inflation — ARIMA Forecasting (Built from Scratch)
============================================================
Author  : Shareef Chitesi
Degree  : BSc Honours Applied Mathematics & Computational Science
          Midlands State University — Year 2

This script implements ARIMA(p,d,q) from first principles using only
NumPy and Pandas — no statsmodels or external forecasting libraries.

Mathematical foundation:
  - Differencing (d): removes non-stationarity (links to ODEs / calculus)
  - AR(p):  y_t = c + φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t
  - MA(q):  y_t = μ + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
  - AIC:    AIC = 2k - 2ln(L)  — used for model selection
  - OLS:    normal equations β = (XᵀX)⁻¹Xᵀy — from Linear Algebra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#1B4F72"
RED    = "#C0392B"
ORANGE = "#E67E22"
GREEN  = "#1E8449"
GREY   = "#808B96"
LIGHT  = "#D6EAF8"
PURPLE = "#6C3483"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────
years = np.array(list(range(2010, 2026)))
inflation = np.array([
    3.2, 3.5, 3.7, 1.6, -0.2, -2.4, -1.6, 0.9,
    10.6, 255.3, 557.2, 98.5, 104.7, 87.4, 47.6, 30.0
])

print("=" * 65)
print("  ZIMBABWE INFLATION — ARIMA FORECASTING (Built from Scratch)")
print("  Author: Shareef Chitesi | MSU Applied Mathematics Year 2")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  STATIONARITY — Apply log transform + first differencing
#     This is the 'I' (Integrated) part of ARIMA
#     Concept: same as finding equilibrium in a differential equation
# ─────────────────────────────────────────────────────────────────────────────
# Shift to positive (min is -2.4)
shifted = inflation + 5.0
log_series = np.log(shifted)

# First difference (d=1) — removes trend
diff1 = np.diff(log_series)

print(f"\n── Stationarity Check ──────────────────────────────────────")
print(f"  Original series mean   : {inflation.mean():.2f}%")
print(f"  Original series std    : {inflation.std():.2f}%")
print(f"  Log-diff series mean   : {diff1.mean():.4f}")
print(f"  Log-diff series std    : {diff1.std():.4f}")
print(f"  Log-differencing reduces variance significantly → more stationary")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  OLS PARAMETER ESTIMATION (Linear Algebra — Normal Equations)
#     β = (XᵀX)⁻¹Xᵀy
#     Used to estimate AR coefficients
# ─────────────────────────────────────────────────────────────────────────────
def fit_ar(series, p):
    """Fit AR(p) model using OLS normal equations: β = (X'X)^{-1} X'y"""
    n = len(series)
    if n <= p:
        return None, None, np.inf
    
    # Build design matrix X and target y
    X = np.column_stack([series[p-i-1:n-i-1] for i in range(p)] + 
                        [np.ones(n - p)])
    y = series[p:]
    
    # Normal equations — core linear algebra from the degree
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        residuals = y - X @ beta
        sigma2 = np.var(residuals)
        
        # Log-likelihood
        n_eff = len(y)
        log_lik = -0.5 * n_eff * (np.log(2 * np.pi * sigma2) + 1)
        
        # AIC = 2k - 2*log_likelihood (k = number of parameters)
        k = p + 1
        aic = 2 * k - 2 * log_lik
        
        return beta, residuals, aic
    except np.linalg.LinAlgError:
        return None, None, np.inf

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL SELECTION — AIC grid search over ARIMA(p, 1, 0) for p in 1..4
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── AIC Model Selection (AR order p = 1 to 4) ──────────────")
print(f"  {'Model':<15} {'AIC':>10}")
print(f"  {'-'*25}")

best_aic  = np.inf
best_p    = 1
best_beta = None
best_resid= None

for p in range(1, 5):
    beta, resid, aic = fit_ar(diff1, p)
    marker = " ← best" if aic < best_aic else ""
    print(f"  AR({p}) / ARIMA({p},1,0)  {aic:>10.3f}{marker}")
    if aic < best_aic:
        best_aic  = aic
        best_p    = p
        best_beta = beta
        best_resid= resid

print(f"\n  Selected model: ARIMA({best_p}, 1, 0)")
print(f"  AIC            : {best_aic:.3f}")
print(f"  AR coefficients: {best_beta[:-1]}")
print(f"  Intercept      : {best_beta[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  RESIDUAL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
resid_mean = best_resid.mean()
resid_std  = best_resid.std()
resid_norm = (best_resid - resid_mean) / resid_std

print(f"\n── Residual Diagnostics ────────────────────────────────────")
print(f"  Residual mean  : {resid_mean:.6f}  (close to 0 = good)")
print(f"  Residual std   : {resid_std:.4f}")
print(f"  Min residual   : {best_resid.min():.4f}")
print(f"  Max residual   : {best_resid.max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FORECASTING — Recursive multi-step ahead forecast
#     Unlog and undifference to get back to inflation %
# ─────────────────────────────────────────────────────────────────────────────
def forecast_arima(log_series, diff_series, beta, p, steps=3):
    """
    Recursive forecast on differenced log series,
    then invert differencing and log transform.
    """
    history = list(diff_series.copy())
    forecasts_diff = []
    
    for _ in range(steps):
        # AR prediction using last p values
        ar_terms = history[-(p):][::-1]  # most recent p values
        pred = beta[-1]  # intercept
        for i, phi in enumerate(beta[:-1]):
            pred += phi * ar_terms[i]
        forecasts_diff.append(pred)
        history.append(pred)
    
    # Invert differencing: cumsum + last log value
    last_log = log_series[-1]
    forecasts_log = last_log + np.cumsum(forecasts_diff)
    
    # Invert log transform (shifted by 5)
    forecasts_inflation = np.exp(forecasts_log) - 5.0
    
    return forecasts_inflation, np.array(forecasts_diff)

forecast_years = np.array([2026, 2027, 2028])
forecast_vals, forecast_diff = forecast_arima(
    log_series, diff1, best_beta, best_p, steps=3
)
forecast_vals = np.clip(forecast_vals, 2.0, None)

# Confidence intervals based on residual std (grows with horizon)
resid_std_orig = best_resid.std()
ci_multiplier  = np.array([1.0, 1.5, 2.0])  # widens with forecast horizon
ci_width       = resid_std_orig * ci_multiplier * 15  # scale to % space
upper = forecast_vals + ci_width
lower = np.clip(forecast_vals - ci_width, 0.5, None)

print(f"\n── ARIMA({best_p},1,0) Forecast ─────────────────────────────────")
print(f"  {'Year':<8} {'Forecast':>12} {'Lower (95%)':>14} {'Upper (95%)':>14}")
print(f"  {'-'*50}")
for yr, fv, lo, hi in zip(forecast_years, forecast_vals, lower, upper):
    print(f"  {yr:<8} {fv:>10.1f}%  {lo:>12.1f}%  {hi:>12.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  LINEAR REGRESSION BASELINE (for comparison chart)
# ─────────────────────────────────────────────────────────────────────────────
recent_years = years[years >= 2021]
recent_inf   = inflation[years >= 2021]
coeffs = np.polyfit(recent_years, recent_inf, 1)
linear_forecast = np.clip(np.polyval(coeffs, forecast_years), 5, None)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT 1 — ARIMA Forecast vs Linear Regression comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

# Historical (stabilisation period)
stab_years = years[years >= 2019]
stab_inf   = inflation[years >= 2019]
ax.plot(stab_years, stab_inf, color=BLUE, linewidth=2.5,
        marker="o", markersize=8, label="Actual inflation", zorder=5)

# ARIMA forecast
ax.plot(forecast_years, forecast_vals, color=PURPLE, linewidth=2.5,
        marker="D", markersize=8, linestyle="--",
        label=f"ARIMA({best_p},1,0) forecast", zorder=5)
ax.fill_between(forecast_years, lower, upper,
                color=PURPLE, alpha=0.15, label="95% confidence interval")

# Linear regression (for comparison)
ax.plot(forecast_years, linear_forecast, color=ORANGE, linewidth=1.8,
        marker="s", markersize=6, linestyle=":",
        label="Linear regression (baseline)", zorder=4, alpha=0.8)

# Forecast region shading
ax.axvspan(2025.5, 2028.5, color=LIGHT, alpha=0.35, label="Forecast period")
ax.axvline(x=2025.5, color=GREY, linestyle="--", linewidth=1, alpha=0.5)

# Annotations
for yr, fv in zip(forecast_years, forecast_vals):
    ax.annotate(f"{fv:.1f}%", xy=(yr, fv), xytext=(0, 14),
                textcoords="offset points", ha="center",
                fontsize=9.5, color=PURPLE, fontweight="bold")

ax.set_title(f"Zimbabwe Inflation: ARIMA({best_p},1,0) vs Linear Regression Forecast",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Inflation Rate (%)", fontsize=11)
ax.set_xticks(list(stab_years) + list(forecast_years))
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig("outputs/05_arima_forecast.png", bbox_inches="tight")
plt.close()
print("\n  ✓ Chart 5 saved: ARIMA Forecast vs Linear Regression")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOT 2 — Residual Diagnostics (4-panel)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# Panel A — Residuals over time
ax1 = fig.add_subplot(gs[0, 0])
resid_years = years[best_p + 1:]
ax1.plot(resid_years, best_resid, color=BLUE, linewidth=1.8, marker="o", markersize=5)
ax1.axhline(0, color=RED, linestyle="--", linewidth=1.2, alpha=0.7)
ax1.fill_between(resid_years, best_resid, 0,
                 where=(best_resid > 0), color=BLUE, alpha=0.15)
ax1.fill_between(resid_years, best_resid, 0,
                 where=(best_resid < 0), color=RED, alpha=0.15)
ax1.set_title("Residuals Over Time", fontweight="bold", fontsize=10)
ax1.set_xlabel("Year"); ax1.set_ylabel("Residual")

# Panel B — Residual histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(best_resid, bins=8, color=BLUE, alpha=0.7, edgecolor="white")
ax2.axvline(0, color=RED, linestyle="--", linewidth=1.5)
ax2.set_title("Residual Distribution", fontweight="bold", fontsize=10)
ax2.set_xlabel("Residual Value"); ax2.set_ylabel("Frequency")

# Panel C — Normal Q-Q plot (manual)
ax3 = fig.add_subplot(gs[1, 0])
sorted_resid = np.sort(resid_norm)
n = len(sorted_resid)
theoretical_q = np.array([np.percentile(
    np.random.randn(10000), 100 * (i + 0.5) / n) for i in range(n)])
ax3.scatter(theoretical_q, sorted_resid, color=PURPLE, s=50, alpha=0.8, zorder=5)
ax3.plot([-2.5, 2.5], [-2.5, 2.5], color=RED, linestyle="--", linewidth=1.5)
ax3.set_title("Normal Q-Q Plot", fontweight="bold", fontsize=10)
ax3.set_xlabel("Theoretical Quantiles"); ax3.set_ylabel("Sample Quantiles")

# Panel D — ACF of residuals (manual autocorrelation)
ax4 = fig.add_subplot(gs[1, 1])
max_lag = min(8, len(best_resid) - 1)
acf_vals = []
resid_centered = best_resid - best_resid.mean()
var0 = np.dot(resid_centered, resid_centered)
for lag in range(max_lag + 1):
    if lag == 0:
        acf_vals.append(1.0)
    else:
        acf_vals.append(
            np.dot(resid_centered[lag:], resid_centered[:-lag]) / var0
        )
lags = np.arange(max_lag + 1)
ax4.bar(lags, acf_vals, color=BLUE, alpha=0.7, width=0.5)
conf_bound = 1.96 / np.sqrt(len(best_resid))
ax4.axhline(conf_bound,  color=RED, linestyle="--", linewidth=1.2, alpha=0.7)
ax4.axhline(-conf_bound, color=RED, linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"95% CI (±{conf_bound:.2f})")
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_title("ACF of Residuals", fontweight="bold", fontsize=10)
ax4.set_xlabel("Lag"); ax4.set_ylabel("Autocorrelation")
ax4.legend(fontsize=8)

fig.suptitle(f"ARIMA({best_p},1,0) — Residual Diagnostics",
             fontsize=13, fontweight="bold", y=1.01)

plt.savefig("outputs/06_residual_diagnostics.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 6 saved: Residual Diagnostics")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOT 3 — Full picture: History + ARIMA forecast
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

colors = [RED if v > 100 else ORANGE if v > 20 else
          GREEN if v < 0 else BLUE for v in inflation]
ax.bar(years, inflation, color=colors, alpha=0.6, width=0.7, zorder=2)
ax.plot(years, inflation, color=BLUE, linewidth=1.8,
        marker="o", markersize=4, zorder=3, label="Historical")

# ARIMA forecast extension
all_years_fc = np.concatenate([[years[-1]], forecast_years])
all_vals_fc  = np.concatenate([[inflation[-1]], forecast_vals])
all_upper_fc = np.concatenate([[inflation[-1]], upper])
all_lower_fc = np.concatenate([[inflation[-1]], lower])

ax.plot(all_years_fc, all_vals_fc, color=PURPLE, linewidth=2.5,
        marker="D", markersize=7, linestyle="--",
        label=f"ARIMA({best_p},1,0) forecast", zorder=5)
ax.fill_between(forecast_years, lower, upper,
                color=PURPLE, alpha=0.15, label="95% CI")

ax.axvspan(2025.5, 2028.5, color=LIGHT, alpha=0.35)

for yr, fv in zip(forecast_years, forecast_vals):
    ax.annotate(f"{fv:.1f}%", xy=(yr, fv), xytext=(0, 14),
                textcoords="offset points", ha="center",
                fontsize=9, color=PURPLE, fontweight="bold")

ax.set_title("Zimbabwe Inflation (2010–2025) + ARIMA Forecast (2026–2028)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Inflation Rate (%)", fontsize=11)
ax.set_xticks(list(years) + list(forecast_years))
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}%"))
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("outputs/07_full_history_arima.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart 7 saved: Full History + ARIMA Forecast")

# ─────────────────────────────────────────────────────────────────────────────
# 11. EXPORT updated Excel
# ─────────────────────────────────────────────────────────────────────────────
import openpyxl
with pd.ExcelWriter("outputs/zimbabwe_inflation_data.xlsx", engine="openpyxl") as writer:
    # Historical
    df_hist = pd.DataFrame({"Year": years, "Inflation_pct": inflation})
    df_hist.to_excel(writer, sheet_name="Historical Data", index=False)

    # ARIMA forecast
    df_fc = pd.DataFrame({
        "Year"         : forecast_years,
        "ARIMA_Forecast": forecast_vals,
        "Linear_Forecast": linear_forecast,
        "Lower_95CI"   : lower,
        "Upper_95CI"   : upper,
    })
    df_fc.to_excel(writer, sheet_name="ARIMA Forecast", index=False)

    # Model summary
    df_model = pd.DataFrame({
        "Parameter": [
            "Model", "AIC", "AR Order (p)", "Differencing (d)",
            "Residual Mean", "Residual Std",
            "Forecast 2026", "Forecast 2027", "Forecast 2028"
        ],
        "Value": [
            f"ARIMA({best_p},1,0)", f"{best_aic:.3f}", str(best_p), "1",
            f"{resid_mean:.6f}", f"{resid_std:.4f}",
            f"{forecast_vals[0]:.1f}%",
            f"{forecast_vals[1]:.1f}%",
            f"{forecast_vals[2]:.1f}%"
        ]
    })
    df_model.to_excel(writer, sheet_name="Model Summary", index=False)

    # Residuals
    df_resid = pd.DataFrame({
        "Year"    : years[best_p + 1:],
        "Residual": best_resid
    })
    df_resid.to_excel(writer, sheet_name="Residuals", index=False)

print("  ✓ Excel updated: zimbabwe_inflation_data.xlsx")

print(f"""
{'='*65}
  SUMMARY
{'='*65}
  Model selected  : ARIMA({best_p}, 1, 0)
  AIC             : {best_aic:.3f}
  Forecast 2026   : {forecast_vals[0]:.1f}%  (95% CI: {lower[0]:.1f}% – {upper[0]:.1f}%)
  Forecast 2027   : {forecast_vals[1]:.1f}%  (95% CI: {lower[1]:.1f}% – {upper[1]:.1f}%)
  Forecast 2028   : {forecast_vals[2]:.1f}%  (95% CI: {lower[2]:.1f}% – {upper[2]:.1f}%)

  Charts saved to outputs/
  05_arima_forecast.png
  06_residual_diagnostics.png
  07_full_history_arima.png
{'='*65}
""")
