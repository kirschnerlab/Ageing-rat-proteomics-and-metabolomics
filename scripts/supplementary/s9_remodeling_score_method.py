"""Supplementary Figure S9: how the remodeling score is calculated.

The score is recomputed here exactly as in scripts/shared/omics_data.py (_state) from the same matrix
(protein_abundance_log2.csv, 11,485 records x 31 rats), and the script stops unless it equals omics_data.STATE, the
score used in Figure 5.
  A  part of the protein matrix as measured (log2 relative to the 9-month mean), rats grouped by age
  B  the same proteins after subtracting each age group's mean and scaling each protein to SD 1
  C  two proteins before and after that step: HS3ST3A/B (changes with age, little spread within age) and KIM-1
  D  each protein's correlation with the score (PC1 of the matrix in B, all 11,485 records)
  E  four proteins against the score, two at each end
Writes outputs/supp_s9/supp_s9_remodeling_score.pdf and .png (300 dpi) and supp_s9_numbers.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "shared")]
import paths                                         # noqa: E402
from style import *                                  # noqa: E402,F401  (shared figure look)
import omics_data as D                               # noqa: E402
import loaders as C                                  # noqa: E402
import matplotlib.colors as mcolors                  # noqa: E402
OUT = paths.out("supp_s9")
AGES = (9, 18, 21, 25)
nums = {}

# ---------------------------------------------------------------- the score, as in omics_data.py
ok = np.isfinite(D.P.loc["Havcr1"].to_numpy(float))
sid, age = D.SID[ok], D.AGE[ok]
full = pd.read_csv(C.OLD)
Y = full[[C.LABEL[s] for s in sid]].to_numpy(float)          # 11,485 records x 31 rats, log2 vs 9-month mean
R = Y.copy()
for a in AGES:
    R[:, age == a] -= R[:, age == a].mean(1, keepdims=True)
Z = R / R.std(1, ddof=1, keepdims=True)
U, S, _ = np.linalg.svd(Z.T, full_matrices=False)
s = U[:, 0] * S[0]
if np.corrcoef(s, D.P.loc["Ptprc"].to_numpy(float)[ok])[0, 1] < 0:
    s = -s
score = s / s.std(ddof=1)
assert np.allclose(score, D.STATE[ok], atol=1e-9), "score differs from omics_data.STATE"
var1 = float(S[0] ** 2 / (S ** 2).sum())
r_all = (Z @ (score - score.mean())) / ((len(score) - 1) * score.std(ddof=1))        # correlation, since rows have SD 1
nums.update(n_records=int(len(Y)), n_rats=int(ok.sum()), pc1_variance_fraction=round(var1, 4))

# gene-level rows (one record per gene, as in Figure 5) for the examples
G = D.P.to_numpy(float)[:, ok]
fin = np.isfinite(G).all(1)
G, gn = G[fin], D.P.index[fin]
GR = G.copy()
for a in AGES:
    GR[:, age == a] -= GR[:, age == a].mean(1, keepdims=True)
GZ = GR / GR.std(1, ddof=1, keepdims=True)
g_r = pd.Series((GZ @ (score - score.mean())) / ((len(score) - 1) * score.std(ddof=1)), index=gn)
g_age = pd.Series(G[:, age == 25].mean(1) - G[:, age == 9].mean(1), index=gn)        # 25 vs 9 months, log2
named = ~gn.str.contains(r"LOC|ENSRNOG|RGD|^Ig[hkl]|^RT1|\.|;|_", regex=True)

# rat order: by age, then by score within age (as in Figure 5A)
order = [i for a in AGES for i in sorted(np.flatnonzero(age == a), key=lambda j: score[j])]
bounds = np.cumsum([np.sum(age == a) for a in AGES])

# protein rows for A/B: 40 most positively and 40 most negatively correlated with the score, and 40 that change most
# with age while barely tracking the score
N = 40
pos = g_r[named].sort_values(ascending=False).index[:N]
neg = g_r[named].sort_values().index[:N]
snr = g_age.abs() / pd.Series(GR.std(1, ddof=1), index=gn)              # age change relative to within-age spread
ageonly = snr[named & (g_r.abs() < 0.2)].sort_values(ascending=False).index[:N]
blocks = [("Higher in\nremodeled\nkidneys", pos), ("Lower in\nremodeled\nkidneys", neg), ("Change\nwith age\nonly", ageonly)]
rows = [g for _, b in blocks for g in b]
idx = {g: i for i, g in enumerate(gn)}
A = G[[idx[g] for g in rows]][:, order]
B = GZ[[idx[g] for g in rows]][:, order]
nums["A_B rows"] = {lab.replace("\n", " "): [g.upper() for g in b[:5]] for lab, b in blocks}

W, H = 183, 224
fig = plt.figure(figsize=(W * MM, H * MM))


def axes(x0, y0, w, h):
    return fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H])


def heat(ax, M, cmap, lim, ttl, letter, cbar_label, cx):
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", lw=1.6)
    for k in (N, 2 * N):
        ax.axhline(k - 0.5, color="white", lw=1.6)
    st = 0
    for a, b in zip(AGES, bounds):
        ax.text((st + b - 1) / 2, len(M) - 0.2, f"{a} mo", ha="center", va="top", fontsize=6, color=INK)
        st = b
    title(ax, letter, ttl, dx=-0.08 if letter == "b" else -0.3)
    cax = axes(cx, 104, 30, 2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(-lim, lim), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=5.8); cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_visible(False)


cmap = plt.colormaps["RdBu_r"]
axA = axes(30, 14, 64, 80)
heat(axA, A, cmap, 1.5, "Protein abundance as measured", "a", "log2 vs 9-month mean", 47)
for k, (lab, _) in enumerate(blocks):
    axA.text(-0.04, (k * N + (k + 1) * N - 1) / 2, lab, transform=axA.get_yaxis_transform(), ha="right", va="center",
             fontsize=5.9, color=INK, linespacing=1.05)
axA.text(0, -0.065, "Columns: rats, ordered within each age by remodeling score.  Rows: 120 example proteins.",
         transform=axA.transAxes, fontsize=5.6, va="top", color=MUTED)
axB = axes(112, 14, 64, 80)
heat(axB, B, cmap, 2.5, "After subtracting each age group's mean", "b", "SD within age group", 129)
fig.text(103 / W, 1 - 54 / H, "→", fontsize=14, ha="center", va="center", color=INK)

# ---------------------------------------------------------------- C: two proteins before and after
rng = np.random.default_rng(3)
EX = [("Hs3st3a1", "HS3ST3A/B", "rises with age, little spread within age"), ("Havcr1", "KIM-1", "spreads widely within age")]
for k, (g, lab, note) in enumerate(EX):
    v = G[idx[g]]; d = GR[idx[g]]
    span = v.max() - v.min()
    x0 = 18 + k * 90
    fig.text(x0 / W, 1 - 127 / H, f"{lab}: {note}", fontsize=6.4, color=INK, fontweight="bold")
    for j, (vals, ylab, sub) in enumerate(((v, "log2 vs 9-month mean", "as measured"),
                                         (d, "log2 minus age-group mean", "minus age-group mean"))):
        ax = axes(x0 + j * 36, 132, 26, 26)
        for i, a in enumerate(AGES):
            x = vals[age == a]
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(x)), x, s=7, c=AGEC[a], edgecolor="white", lw=0.3, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [x.mean()] * 2, c=INK, lw=0.8, zorder=4)
        ax.axhline(0, color=PALE, lw=0.5)
        ax.set_xticks(range(4), AGES, fontsize=5.5); ax.tick_params(axis="y", labelsize=5.5)
        ax.set_xlabel("Age (months)", fontsize=5.8)
        ax.set_ylabel(ylab, fontsize=5.4, labelpad=1)
        ax.set_title(sub, fontsize=6, pad=2, loc="left", color=MUTED)
        pad = 0.08 * span
        if j == 0:
            ax.set_ylim(v.min() - pad, v.max() + pad)
        else:                                            # same span, centred on 0, so the two plots compare directly
            ax.set_ylim(-span / 2 - pad, span / 2 + pad)
    nums[f"C {lab}"] = {"25 vs 9 (log2)": round(float(v[age == 25].mean() - v[age == 9].mean()), 3),
                        "SD within 25-month rats": round(float(d[age == 25].std(ddof=1)), 3),
                        "r with score": round(float(g_r[g]), 3)}
fig.text(8 / W, 1 - 121 / H, "c", fontsize=9.5, fontweight="bold")
fig.text(12 / W, 1 - 121 / H, "Two proteins before and after subtracting the age-group mean", fontsize=7.5, color=INK)

# ---------------------------------------------------------------- D: distribution of correlations with the score
axD = axes(18, 181, 50, 26)
axD.hist(r_all, bins=np.linspace(-1, 1, 61), color="#bfc3c8", lw=0)
for g, lab in (("Ptprc", "CD45"), ("Havcr1", "KIM-1"), ("Slc22a6", "OAT1"), ("Hs3st3a1", "HS3ST3A/B")):
    x = float(g_r[g])
    axD.axvline(x, color=UP if x > 0.3 else DOWN if x < -0.3 else MUTED, lw=0.8)
    axD.text(x, axD.get_ylim()[1] * 0.98, lab, rotation=90, ha="right", va="top", fontsize=5.6)
axD.set_xlim(-1, 1); axD.set_xlabel("Correlation with remodeling score (r)", fontsize=6)
axD.set_ylabel("Proteins", fontsize=6); axD.tick_params(labelsize=5.5)
fig.text(8 / W, 1 - 173 / H, "d", fontsize=9.5, fontweight="bold")
fig.text(12 / W, 1 - 173 / H, f"All {len(r_all):,} proteins", fontsize=7.5, color=INK)
nums["D share |r| >= 0.5"] = round(float(np.mean(np.abs(r_all) >= 0.5)), 3)

# ---------------------------------------------------------------- E: proteins against the score
cand_neg = [g for g in ("Slc22a6", "Slc34a1", "Slc5a2", "Lrp2") if g in idx]
neg2 = sorted(cand_neg, key=lambda g: g_r[g])[:2]
EXE = [("Ptprc", "CD45"), ("Havcr1", "KIM-1")] + [(g, {"Slc22a6": "OAT1", "Slc34a1": "NaPi-IIa", "Slc5a2": "SGLT2", "Lrp2": "megalin"}[g]) for g in neg2]
for k, (g, lab) in enumerate(EXE):
    ax = axes(86 + k * 25, 181, 19, 26)
    y = GZ[idx[g]]
    for a in AGES:
        m = age == a
        ax.scatter(score[m], y[m], s=6, c=AGEC[a], edgecolor="white", lw=0.3, zorder=3)
    b1, b0 = np.polyfit(score, y, 1)
    xx = np.array([score.min(), score.max()]); ax.plot(xx, b0 + b1 * xx, c=UP if b1 > 0 else DOWN, lw=0.9)
    ax.axhline(0, color=PALE, lw=0.5); ax.axvline(0, color=PALE, lw=0.5)
    ax.set_title(f"{lab}\nr = {g_r[g]:+.2f}".replace("-", "−"), fontsize=6, pad=2, loc="left")
    ax.set_xlabel("Score (SD)", fontsize=5.8); ax.tick_params(labelsize=5.3)
    ax.set_ylim(-3.2, 3.2); ax.set_yticks([-2, 0, 2])
    if k == 0:
        ax.set_ylabel("Protein, SD within age", fontsize=5.6)
    else:
        ax.set_yticklabels([])
    nums[f"E {lab} r"] = round(float(g_r[g]), 3)
fig.text(78 / W, 1 - 173 / H, "e", fontsize=9.5, fontweight="bold")
fig.text(82 / W, 1 - 173 / H, "Each rat's protein level within its age group against its score", fontsize=7.5, color=INK)
h_ = [plt.Line2D([], [], marker="o", ls="", color=AGEC[a], ms=3.5, label=f"{a} months") for a in AGES]
fig.legend(handles=h_, loc="center", bbox_to_anchor=(0.5, 1 - 218 / H), ncol=4, frameon=False, fontsize=6.2)

fig.savefig(OUT / "supp_s9_remodeling_score.pdf")
fig.savefig(OUT / "supp_s9_remodeling_score.png", dpi=300)
(OUT / "supp_s9_numbers.json").write_text(json.dumps(nums, indent=2), encoding="utf-8")
print(json.dumps(nums, indent=1))
