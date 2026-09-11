"""Generates fig_per_problem_pdi.pdf and fig_d3prob5_confidence_error.pdf
for the paper's new SS4.6 (Two divergence mechanisms). Uses the same
normalized-answer pipeline as scripts/problem_clustered_permutation_test.py
so numbers match the paper's reported figures exactly.
"""
import glob
import json
import pathlib
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = pathlib.Path(__file__).resolve().parent / "latex" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from early_stop.parsers import normalize_math_answer
from early_stop.path_dependence import answer_distribution, pdi

BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#E69F00"
GREY = "#999999"
DARKGREY = "#555555"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

RESULTS_DIR = REPO_ROOT / "results" / "candidate_b_fullpilot"


def load_pairs():
    pairs = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        if "followup" in f:
            continue
        for e in json.load(open(f)):
            if "side_a_raw_answers" in e:
                a, b = e["side_a_raw_answers"], e["side_b_raw_answers"]
            elif "branches_a" in e:
                a, b = e["branches_a"], e["branches_b"]
            else:
                continue
            a = [normalize_math_answer(x) if x else "EMPTY" for x in a]
            b = [normalize_math_answer(x) if x else "EMPTY" for x in b]
            pairs.append({"problem_id": e["problem_id"], "a": a, "b": b, "raw": e})
    return pairs


pairs = load_pairs()

# ---------------------------------------------------------------------
# Figure: per-problem mean PDI, sorted, anomalous 3 highlighted (Figure 6)
# ---------------------------------------------------------------------

by_problem = defaultdict(list)
for p in pairs:
    val = pdi(answer_distribution(p["a"]), answer_distribution(p["b"]))
    by_problem[p["problem_id"]].append(val)

items = sorted(by_problem.items(), key=lambda kv: -float(np.mean(kv[1])))
labels = [pid.replace("_prob", "\\_prob").replace("_", "\\_") for pid, _ in items]
labels = [pid for pid, _ in items]  # keep raw ids, escape at plot time via matplotlib text (no LaTeX render needed)
means = [float(np.mean(v)) for _, v in items]
ns = [len(v) for _, v in items]

anomalous = {"d1_prob1", "d2_prob1", "d3_prob5"}
colors = [ORANGE if pid in anomalous else GREY for pid, _ in items]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
xpos = np.arange(len(items))
ax.bar(xpos, means, color=colors, edgecolor="black", linewidth=0.4, width=0.7)
ax.set_xticks(xpos)
ax.set_xticklabels([pid.replace("_", "\\_") if False else pid for pid in labels], rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("Mean PDI across a problem's sampled pairs")
ax.set_xlabel("Problem")

for xi, (pid, n) in zip(xpos, [(pid, len(v)) for pid, v in items]):
    ax.text(xi, means[xpos.tolist().index(xi)] + 0.003, f"n={n}", ha="center", fontsize=6.5, color=DARKGREY)

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=ORANGE, edgecolor="black", label="Anomalous (carries aggregate signal)"),
    Patch(facecolor=GREY, edgecolor="black", label="Flat null (PDI = 0 in every pair)"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=8, frameon=False)
ax.set_title("14 of 17 tested problems are flat null. 3 carry the aggregate signal.", fontsize=10)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_per_problem_pdi.pdf")
plt.close(fig)
print("Wrote fig_per_problem_pdi.pdf")
for pid, m, n in zip(labels, means, ns):
    print(f"  {pid}: mean={m:.4f} n={n}")

# ---------------------------------------------------------------------
# Figure: d3_prob5 confidence vs error rate, both sides, all 4 pairs (Figure 7)
# ---------------------------------------------------------------------

d3p5 = [p for p in pairs if p["problem_id"] == "d3_prob5"]
assert len(d3p5) == 4, f"expected 4 pairs, got {len(d3p5)}"

rows = []
for i, p in enumerate(d3p5, start=1):
    a_state = p["raw"]["side_a_state"]
    b_state = p["raw"]["side_b_state"]
    shared_answer = normalize_math_answer(a_state["answer"])
    a_err = sum(1 for x in p["a"] if x != shared_answer) / len(p["a"]) * 100
    b_err = sum(1 for x in p["b"] if x != shared_answer) / len(p["b"]) * 100
    rows.append({
        "pair": i, "a_conf": a_state["confidence"], "b_conf": b_state["confidence"],
        "a_err": a_err, "b_err": b_err,
    })

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.5, 4.0), sharey=True)
ypos = np.arange(len(rows))[::-1]  # pair 1 at top

for y, r in zip(ypos, rows):
    lo, hi = sorted([r["a_conf"], r["b_conf"]])
    ax0.plot([lo, hi], [y, y], color=GREY, linewidth=1.2, zorder=1)
    a_color = ORANGE if r["a_err"] > 0 else TEAL
    b_color = ORANGE if r["b_err"] > 0 else TEAL
    ax0.scatter(r["a_conf"], y, color=a_color, s=90, edgecolor="black", linewidth=0.7, zorder=3, marker="o")
    ax0.scatter(r["b_conf"], y, color=b_color, s=90, edgecolor="black", linewidth=0.7, zorder=3, marker="s")

    lo, hi = sorted([r["a_err"], r["b_err"]])
    ax1.plot([lo, hi], [y, y], color=GREY, linewidth=1.2, zorder=1)
    ax1.scatter(r["a_err"], y, color=a_color, s=90, edgecolor="black", linewidth=0.7, zorder=3, marker="o")
    ax1.scatter(r["b_err"], y, color=b_color, s=90, edgecolor="black", linewidth=0.7, zorder=3, marker="s")

ax0.set_yticks(ypos)
ax0.set_yticklabels([f"Pair {r['pair']}" for r in rows])
ax0.set_xlabel("Confidence at matched checkpoint")
ax0.set_xlim(0, 1)
ax0.set_title("Confidence", fontsize=10)

ax1.set_xlabel("Error rate on continuation (%)")
ax1.set_xlim(-2, 35)
ax1.set_title("Error rate", fontsize=10)

legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markeredgecolor="black", markersize=9, label="Side A"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=GREY, markeredgecolor="black", markersize=9, label="Side B"),
    Patch(facecolor=TEAL, edgecolor="black", label="Stayed correct (0% error)"),
    Patch(facecolor=ORANGE, edgecolor="black", label="Scattered into errors"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.suptitle("d3_prob5: the lower-confidence side of each pair scatters into errors", fontsize=10.5)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUT_DIR / "fig_d3prob5_confidence_error.pdf", bbox_inches="tight")
plt.close(fig)
print("Wrote fig_d3prob5_confidence_error.pdf")
for r in rows:
    print(f"  pair {r['pair']}: A conf={r['a_conf']:.3f} err={r['a_err']:.1f}%  B conf={r['b_conf']:.3f} err={r['b_err']:.1f}%")
