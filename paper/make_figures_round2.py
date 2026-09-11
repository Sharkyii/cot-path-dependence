"""Generates fig_continuous_regression.pdf and fig_sges_comparison.pdf
from the real round-2 scale-up and Phase 4 SG-ES results, for §4.5 and
§4.6 of the paper. Vector PDF output, no rasterization needed at this
point count (45 and 19 points).

Note on the SG-ES figure: no raw token counts were logged per condition,
only fraction_of_trace_used (a proxy already used throughout the paper's
prose and tables). The "compute cost" panel plots that fraction directly
rather than inventing absolute token numbers that were never measured.
"""
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = pathlib.Path(__file__).resolve().parent / "latex" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-friendly palette
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#E69F00"
GREY = "#555555"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------
# Figure 1: continuous state-distance regression (paper Figure 4, §4.5)
# ---------------------------------------------------------------------

round2_files = sorted(glob.glob(str(REPO_ROOT / "results" / "candidate_b_fullpilot_round2" / "*.json")))
pairs = []
for f in round2_files:
    for e in json.load(open(f)):
        excess = e["observed_pdi"] - e["null_mean"]
        pairs.append((e["state_distance"], excess))

x = np.array([p[0] for p in pairs])
y = np.array([p[1] for p in pairs])
n = len(x)

# 2026-09-12 revision (REVISION_PLAN.md Tier 2): plot EXCESS PDI (observed
# minus that pair's own permutation null mean), not raw PDI. Raw PDI's
# plug-in estimator is biased upward at N=10 for any non-unanimous pair,
# so it is not comparable to zero -- excess PDI is. This is also what
# main.tex's corrected regression (cluster-robust SEs) is fit on.
X_aug = np.column_stack([np.ones(n), x])
beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
intercept, slope = beta[0], beta[1]
residuals = y - X_aug @ beta

# HC1 heteroskedasticity-robust (cluster-robust in spirit) SEs, matching
# scripts/final_analysis.py's regression_with_cluster_robust_se().
k = 2
XX_inv = np.linalg.inv(X_aug.T @ X_aug)
meat = np.zeros((k, k))
for i in range(n):
    meat += residuals[i] ** 2 * np.outer(X_aug[i], X_aug[i])
meat *= n / (n - k)
var_matrix = XX_inv @ meat @ XX_inv
se = np.sqrt(np.diag(var_matrix))
tcrit = stats.t.ppf(0.975, n - k)
intercept_se, slope_se = se[0], se[1]
intercept_p = 2 * (1 - stats.t.cdf(abs(intercept / intercept_se), n - k))
slope_p = 2 * (1 - stats.t.cdf(abs(slope / slope_se), n - k))

x_line = np.linspace(0, x.max() * 1.05, 200)
y_line = intercept + slope * x_line
# Approximate mean-fit CI band using the same HC1 variance at each x
# (var(y_hat) = [1, x] V [1, x]^T), good enough for the visual band.
se_line = np.array([
    np.sqrt(np.array([1, xv]) @ var_matrix @ np.array([1, xv]))
    for xv in x_line
])
ci_lo = y_line - tcrit * se_line
ci_hi = y_line + tcrit * se_line

# The direct-measurement points: pairs at exactly d=0 (matched on
# confidence and entropy too, not by design). Marked distinctly per the
# corrected fig:contreg caption in main.tex -- these are what the
# intercept is supposed to estimate, and unlike the fitted line, they
# are observed, not extrapolated.
zero_d_mask = x < 1e-9

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.scatter(x[~zero_d_mask], y[~zero_d_mask], s=28, color=BLUE, alpha=0.75,
           edgecolor="white", linewidth=0.5, zorder=3, label="Matched pairs ($d>0$)")
ax.scatter(x[zero_d_mask], y[zero_d_mask], s=60, color=TEAL, alpha=0.9, marker="D",
           edgecolor="black", linewidth=0.6, zorder=4,
           label=f"Direct measurement at $d=0$ ($n={zero_d_mask.sum()}$, all zero)")
ax.plot(x_line, y_line, color=ORANGE, linewidth=1.8, zorder=2, label="Fitted regression")
ax.fill_between(x_line, ci_lo, ci_hi, color=ORANGE, alpha=0.18, zorder=1, label="95% CI (mean fit)")
ax.axhline(0, color=GREY, linewidth=0.6, linestyle=":", zorder=0)

ax.set_xlabel(r"State-space distance $d(\phi_A, \phi_B)$")
ax.set_ylabel("Excess Path Dependence Index (observed $-$ null mean)")
ax.set_xlim(left=0)

stat_text = (
    f"$n = {n}$\n"
    f"intercept $= {intercept:.4f}$, $p = {intercept_p:.2f}$\n"
    f"slope $= {slope:.4f}$, $p = {slope_p:.2f}$\n"
    f"$d{{=}}0$ pairs: {zero_d_mask.sum()} of {n}, all excess $=0$"
)
ax.text(
    0.97, 0.95, stat_text, transform=ax.transAxes,
    ha="right", va="top", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GREY, linewidth=0.6, alpha=0.9),
)
ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.92,
          edgecolor=GREY, facecolor="white")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_continuous_regression.pdf")
plt.close(fig)
print(f"Wrote fig_continuous_regression.pdf: n={n}, intercept={intercept:.4f} (p={intercept_p:.4f}), "
      f"slope={slope:.4f} (p={slope_p:.4f}), zero-d pairs={zero_d_mask.sum()}")

# ---------------------------------------------------------------------
# Figure 2: SG-ES vs baselines (paper Figure 5, §4.6)
# ---------------------------------------------------------------------

phase4 = json.load(open(REPO_ROOT / "results" / "candidate_b_phase4_sges" / "phase4_results.json"))
n_p4 = len(phase4)

conditions = [
    ("full_generation", "Full\nGeneration"),
    ("forced_extraction_0.5", "Naive\n(fixed 50%)"),
    ("sg_es", "SG-ES"),
]
colors = [GREY, TEAL, ORANGE]

acc_mean, acc_se = [], []
frac_mean, frac_se = [], []
for key, _ in conditions:
    correct = np.array([1.0 if e[key]["correct"] else 0.0 for e in phase4])
    frac = np.array([e[key]["fraction_of_trace_used"] for e in phase4])
    p_hat = correct.mean()
    acc_mean.append(p_hat * 100)
    acc_se.append(100 * np.sqrt(p_hat * (1 - p_hat) / n_p4))
    frac_mean.append(frac.mean() * 100)
    frac_se.append(100 * frac.std(ddof=1) / np.sqrt(n_p4))

labels = [c[1] for c in conditions]
xpos = np.arange(len(conditions))

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))

ax0 = axes[0]
ax0.bar(xpos, acc_mean, yerr=acc_se, capsize=4, color=colors, width=0.6,
        edgecolor="black", linewidth=0.5)
ax0.set_ylabel("Accuracy (%)")
ax0.set_title("Accuracy (Pass@1)")
ax0.set_xticks(xpos)
ax0.set_xticklabels(labels, fontsize=8.5)
ax0.set_ylim(0, 100)
for xi, v in zip(xpos, acc_mean):
    ax0.text(xi, v + 3, f"{v:.1f}", ha="center", fontsize=8.5)

ax1 = axes[1]
ax1.bar(xpos, frac_mean, yerr=frac_se, capsize=4, color=colors, width=0.6,
        edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Mean fraction of trace used (%)")
ax1.set_title("Compute cost proxy")
ax1.set_xticks(xpos)
ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_ylim(0, 110)
for xi, v in zip(xpos, frac_mean):
    ax1.text(xi, v + 3, f"{v:.1f}", ha="center", fontsize=8.5)

fig.suptitle(f"n = {n_p4} MATH-500 problems, SE across problems shown", fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_sges_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Wrote fig_sges_comparison.pdf: n={n_p4}")
print("accuracy means:", dict(zip(labels, acc_mean)))
print("fraction means:", dict(zip(labels, frac_mean)))
