import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = f"{REPO_ROOT}/results/candidate_b_fullpilot"
OUT = f"{REPO_ROOT}/paper_build/figs"

# ---------- Load main 21-pair dataset ----------
pairs = []
for f in sorted(glob.glob(f"{RESULTS}/level*_*of3.json")):
    d = json.load(open(f))
    for e in d:
        pairs.append(e)

assert len(pairs) == 21, len(pairs)

# ---------- Figure 1: PDI distribution across all 21 pairs, sorted ----------
fig, ax = plt.subplots(figsize=(6.5, 3.2))
sorted_pairs = sorted(pairs, key=lambda e: e["observed_pdi"])
pdis = [e["observed_pdi"] for e in sorted_pairs]
sig = [e["p_value"] < 0.05 for e in sorted_pairs]
bonferroni_sig = [e["p_value"] < 0.05/21 for e in sorted_pairs]
colors = []
for s, bs in zip(sig, bonferroni_sig):
    if bs:
        colors.append("#c0392b")
    elif s:
        colors.append("#e67e22")
    else:
        colors.append("#7f8c8d")
x = np.arange(len(pairs))
ax.bar(x, pdis, color=colors, width=0.7)
ax.set_xlabel("Matched pair (sorted by PDI)")
ax.set_ylabel("Path Dependence Index (PDI)")
ax.set_xticks([])
ax.set_title("Distribution of PDI across all 21 L1-matched pairs", fontsize=11)
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#7f8c8d", label="not significant (p ≥ 0.05)"),
    Patch(facecolor="#e67e22", label="nominal p < 0.05"),
    Patch(facecolor="#c0392b", label="survives Bonferroni (p < 0.05/21)"),
]
ax.legend(handles=legend_elems, fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_pdi_distribution.png", dpi=200)
plt.close(fig)

# ---------- Figure 2: Per-stratum Fisher combined p-values ----------
from scipy.stats import combine_pvalues
from collections import defaultdict

by_stratum = defaultdict(list)
for e in pairs:
    by_stratum[e["difficulty"]].append(e)

strata = ["1", "2", "3"]
labels = ["Level 1", "Level 2", "Level 3", "Overall (21)"]
pvals_combined = []
for s in strata:
    pv = [e["p_value"] for e in by_stratum[s]]
    _, p = combine_pvalues(pv, method="fisher")
    pvals_combined.append(p)
_, p_overall = combine_pvalues([e["p_value"] for e in pairs], method="fisher")
pvals_combined.append(p_overall)

fig, ax = plt.subplots(figsize=(5.5, 3.2))
bar_colors = ["#2980b9", "#2980b9", "#2980b9", "#8e44ad"]
bars = ax.bar(labels, pvals_combined, color=bar_colors, width=0.55)
ax.axhline(0.05, color="#c0392b", linestyle="--", linewidth=1, label="α = 0.05")
ax.set_ylabel("Fisher's combined p-value")
ax.set_ylim(0, 1.05)
ax.set_title("Combined significance by difficulty stratum", fontsize=11)
for b, p in zip(bars, pvals_combined):
    ax.text(b.get_x() + b.get_width()/2, p + 0.02, f"{p:.3f}", ha="center", fontsize=9)
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_stratum_pvalues.png", dpi=200)
plt.close(fig)

# ---------- Figure 3: Decisiveness follow-up (L1 -> L2 -> L3) ----------
# Hand-entered from the pair_results (verified against saved JSON below)
followup = {
    "d1_prob1": {
        "L1": {"A": {"answer": 75, "empty": 25}, "B": {"answer": 95, "empty": 5}},
        "L2": {"A": {"answer": 100, "empty": 0}, "B": {"answer": 100, "empty": 0}},
        "L3": {"A": {"answer": 100, "empty": 0}, "B": {"answer": 100, "empty": 0}},
    },
    "d2_prob1": {
        "L1": {"A": {"answer": 72.5, "empty": 27.5}, "B": {"answer": 15, "empty": 85}},
        "L2": {"A": {"answer": 100, "empty": 0}, "B": {"answer": 100, "empty": 0}},
        "L3": {"A": {"answer": 100, "empty": 0}, "B": {"answer": 100, "empty": 0}},
    },
}

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
levels = ["L1", "L2", "L3"]
for ax, (prob, data) in zip(axes, followup.items()):
    x = np.arange(len(levels))
    width = 0.35
    a_vals = [data[l]["A"]["answer"] for l in levels]
    b_vals = [data[l]["B"]["answer"] for l in levels]
    ax.bar(x - width/2, a_vals, width, label="Side A", color="#27ae60")
    ax.bar(x + width/2, b_vals, width, label="Side B", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(["L1\n(answer only)", "L2\n(+confidence)", "L3\n(+entropy)"])
    ax.set_title(prob, fontsize=11, style="italic")
    ax.set_ylim(0, 110)
axes[0].set_ylabel("% branches with a\nstated (non-empty) answer")
axes[0].legend(fontsize=9, frameon=False, loc="lower right")
fig.suptitle("Decisiveness gap closes under stricter matching (exploratory, n=2)", fontsize=11)
fig.tight_layout(rect=[0.03, 0, 1, 0.92])
fig.savefig(f"{OUT}/fig3_decisiveness_followup.png", dpi=200)
plt.close(fig)

print("done")
for f in ["fig1_pdi_distribution.png", "fig2_stratum_pvalues.png", "fig3_decisiveness_followup.png"]:
    import os
    print(f, os.path.getsize(f"{OUT}/{f}"), "bytes")
