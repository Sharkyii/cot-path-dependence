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
        pairs.append((e["state_distance"], e["observed_pdi"]))

x = np.array([p[0] for p in pairs])
y = np.array([p[1] for p in pairs])
n = len(x)

res = stats.linregress(x, y)
slope, intercept = res.slope, res.intercept

xbar = x.mean()
Sxx = np.sum((x - xbar) ** 2)
resid = y - (intercept + slope * x)
dof = n - 2
s_resid = np.sqrt(np.sum(resid ** 2) / dof)
tcrit = stats.t.ppf(0.975, dof)

x_line = np.linspace(0, x.max() * 1.05, 200)
y_line = intercept + slope * x_line
se_line = s_resid * np.sqrt(1.0 / n + (x_line - xbar) ** 2 / Sxx)
ci_lo = y_line - tcrit * se_line
ci_hi = y_line + tcrit * se_line

slope_ci_lo = slope - tcrit * res.stderr
slope_ci_hi = slope + tcrit * res.stderr

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.scatter(x, y, s=28, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3, label="Matched pairs")
ax.plot(x_line, y_line, color=ORANGE, linewidth=1.8, zorder=2, label="Fitted regression")
ax.fill_between(x_line, ci_lo, ci_hi, color=ORANGE, alpha=0.18, zorder=1, label="95% CI (mean fit)")
ax.axhline(0, color=GREY, linewidth=0.6, linestyle=":", zorder=0)

ax.set_xlabel(r"State-space distance $d(\phi_A, \phi_B)$")
ax.set_ylabel("Path Dependence Index (PDI)")
ax.set_ylim(0, 1)
ax.set_xlim(left=0)

stat_text = (
    f"$n = {n}$\n"
    f"$\\hat\\beta = {slope:.4f}$\n"
    f"$p = {res.pvalue:.2f}$\n"
    f"95% CI $[{slope_ci_lo:.3f}, {slope_ci_hi:.3f}]$"
)
ax.text(
    0.97, 0.95, stat_text, transform=ax.transAxes,
    ha="right", va="top", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GREY, linewidth=0.6, alpha=0.9),
)
ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_continuous_regression.pdf")
plt.close(fig)
print(f"Wrote fig_continuous_regression.pdf: n={n}, slope={slope:.4f}, p={res.pvalue:.4f}, "
      f"95% CI=[{slope_ci_lo:.4f}, {slope_ci_hi:.4f}]")

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
