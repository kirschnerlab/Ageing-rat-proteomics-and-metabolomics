"""Supplementary Figure S11 (study balance and recovered blood-associated signals), in the style of Figs 4-5, from
blood_signal_scores.csv. Design cells are labelled by labeling set and preparation batch, without reporter chemistry.
Writes outputs/supp_s11/supp_s11_study_design_blood_signal.{pdf,png}."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'shared')]
import paths
from style import *
SRC = paths.BLOOD_SIGNAL_SCORES
OUT = paths.out('supp_s11')
Q = pd.read_csv(SRC)
assert len(Q) == 31 and Q.label.is_unique
AGES = [9, 18, 21, 25]
SET = {"D": "Set 1", "H": "Set 2"}
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(183 * MM, 62 * MM))
ax = fig.add_axes([0.1, 0.3, 0.16, 0.54])
ct = pd.crosstab(Q.age, [Q.series.map(SET), Q.batch]).reindex(AGES)
ax.imshow(ct.to_numpy(), cmap="Purples", vmin=0, vmax=3.2, aspect="auto")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = int(ct.iloc[i, j])
        ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="white" if v >= 2 else INK)
ax.axvline(1.5, c="white", lw=2)
ax.set_xticks(range(ct.shape[1]), [b.replace("Batch ", "") for _, b in ct.columns], fontsize=6.3)
for j0, s in ((0.5, "Set 1"), (2.5, "Set 2")):
    ax.text(j0, 4.25, s, ha="center", va="top", fontsize=6.5)
ax.text(1.5, 4.85, "Preparation batch within labeling set", ha="center", va="top", fontsize=6.3, color=MUTED)
ax.set_yticks(range(len(AGES)), AGES)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_ylabel("Age (months)")
title(ax, "a", "Rats per design cell", dx=-0.5)
for k, (col, letter, ttl) in enumerate([("rbc_score_z", "b", "Erythrocyte-associated signal"), ("plasma_score_z", "c", "Plasma-associated signal")]):
    ax = fig.add_axes([0.37 + k * 0.33, 0.2, 0.25, 0.66])
    for j, a in enumerate(AGES):
        v = Q.loc[Q.age == a, col].to_numpy()
        ax.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=12, c=AGEC[a], edgecolor="white", lw=0.3, zorder=3)
        ax.plot([j - 0.25, j + 0.25], [v.mean()] * 2, c=INK, lw=0.9, zorder=4)
    ax.axhline(0, c=PALE, lw=0.6)
    ax.set_xticks(range(4), AGES)
    ax.set(xlabel="Age (months)", ylabel="Composite z score")
    if letter == "b":
        r = Q.loc[Q[col].idxmax()]
        ax.annotate(r.label.split("_")[0], (AGES.index(r.age), r[col]), xytext=(6, -3), textcoords="offset points", fontsize=6.3)
    title(ax, letter, ttl, dx=-0.22)
fig.savefig(OUT / "supp_s11_study_design_blood_signal.pdf")
fig.savefig(OUT / "supp_s11_study_design_blood_signal.png", dpi=400)
print(ct)
