"""Supplementary Figure S3: member proteins of six fixed programs (program_members.csv) in every rat, with gene labels.
Values are log2 relative to each protein's 9-month mean; rats are ordered by age, then by KIM-1 as in Fig 4b; rows are
ordered by the 25-month mean. Programs of up to 24 measured proteins are fully labelled; larger ones label the six
largest increases and four largest decreases at 25 months.
Writes outputs/supp_s3/supp_s3_program_heatmaps.{pdf,png}."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'shared')]
import paths
from style import *
import omics_data as D
import matplotlib.colors as mcolors
OUT = paths.out('supp_s3')
W, H = 183, 116
fig = plt.figure(figsize=(W * MM, H * MM))


def axes(x0, y0, w, h):
    return fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H])


memb = pd.read_csv(paths.PROGRAM_MEMBERS)
PROGS = [("a", "Interferon response", "Interferon_response"), ("b", "Myeloid", "Macrophage_myeloid"), ("c", "Collagen IV", "Collagen_IV"),
         ("d", "Oxidative phosphorylation", "OXPHOS_structural"), ("e", "Fatty-acid oxidation", "FAO_beta_oxidation"),
         ("f", "Proximal-tubule markers", "Proximal_tubule_markers")]
kim = D.prot("Havcr1")
ok = np.isfinite(kim)
cols = [i for a in (9, 18, 21, 25) for i in sorted(np.flatnonzero(ok & (D.AGE == a)), key=lambda j: kim[j])]
ages_c = D.AGE[cols]
bounds = np.cumsum([np.sum(ages_c == a) for a in (9, 18, 21, 25)])
cmap = plt.colormaps["RdBu_r"]
POS = [(10, 8), (70, 8), (130, 8), (10, 64), (70, 64), (130, 64)]
nlab = {}
for (letter, name, key), (x0, y0) in zip(PROGS, POS):
    genes = [g for g in dict.fromkeys(memb.loc[memb.program.eq(key), "gene"]) if g in D.P.index]
    A = D.P.loc[genes].to_numpy(float)
    X = A[:, cols] - np.nanmean(A[:, ok & (D.AGE == 9)], axis=1, keepdims=True)
    o = np.argsort(-np.nanmean(X[:, ages_c == 25], axis=1))
    X, g_sorted = X[o], [genes[i] for i in o]
    ax = axes(x0, y0, 34, 40)
    ax.imshow(np.clip(X, -1.5, 1.5), aspect="auto", cmap=cmap, vmin=-1.5, vmax=1.5, interpolation="nearest")
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, c="white", lw=1.2)
    starts = np.r_[0, bounds[:-1]]
    ax.set_xticks((starts + bounds - 1) / 2, [9, 18, 21, 25], fontsize=6)
    ax.set_yticks([]); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Age (months)", fontsize=6.2, labelpad=2)
    n = len(g_sorted)
    pick = list(range(n)) if n <= 24 else list(range(6)) + list(range(n - 4, n))
    gap = 6.0 / (40 * MM * 72 / n)              # minimum label spacing, in rows
    ys = np.array(pick, float)
    for _ in range(200):                        # push overlapping labels apart
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                d = (gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= d; ys[i] += d; moved = True
        ys = np.clip(ys, -0.5, n - 0.5)
        if not moved:
            break
    nc = X.shape[1]
    for r, yl in zip(pick, ys):
        ax.plot([nc - 0.5, nc + 0.9, nc + 2.2], [r, r, yl], c="#9a9a9a", lw=0.35, clip_on=False)
        ax.text(nc + 2.6, yl, g_sorted[r].upper(), fontsize=5.2, va="center", ha="left", clip_on=False)
    title(ax, letter, f"{name} ({n})", dx=-0.1)
    nlab[name] = (n, len(pick))
cax = axes(176, 12, 2.5, 84)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(-1.5, 1.5), cmap=cmap), cax=cax)
cb.set_label("log2 vs 9-month mean (clipped at ±1.5)", fontsize=6); cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_visible(False)
fig.savefig(OUT / "supp_s3_program_heatmaps.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(OUT / "supp_s3_program_heatmaps.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
print(nlab)
