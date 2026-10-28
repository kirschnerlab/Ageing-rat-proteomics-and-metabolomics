"""Supplementary Figure S6: protein (this study) and RNA (54 other rats) trajectories for 23 example genes, in the style
of Figs 4-5. Protein values come from the protein matrix (one representative record per gene, relative to the 9-month
mean); RNA values from trajectory_examples.csv (relative to 6 months). CYP2C11 is not shown.
Writes outputs/supp_s6/supp_s6_protein_rna_trajectories.{pdf,png}."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'shared')]
import paths
from style import *
import omics_data as D
TR = pd.read_csv(paths.RNA_TRAJECTORY_EXAMPLES)
OUT = paths.out('supp_s6')
PROT_C, RNA_C = "#c8531f", "#2a78d6"
GENES = [("Havcr1", "HAVCR1 (KIM-1)"), ("Lcn2", "LCN2 (NGAL)"), ("Timp1", "TIMP1"), ("Cd68", "CD68"), ("Psmb8", "PSMB8"), ("Psmb9", "PSMB9"),
         ("Mcpt1", "MCPT1"), ("Cpa3", "CPA3"), ("Slc22a6", "SLC22A6 (OAT1)"), ("Slco1a1", "SLCO1A1"), ("Ndufa9", "NDUFA9"), ("Acadm", "ACADM"),
         ("Esrrb", "ESRRB"), ("C3", "C3"), ("C9", "C9"), ("C8b", "C8B"), ("Col4a1", "COL4A1"), ("Col4a3", "COL4A3"), ("Col4a5", "COL4A5"),
         ("Col1a1", "COL1A1"), ("Umod", "UMOD"), ("Tnfrsf11b", "TNFRSF11B (OPG)"), ("Alb", "ALB")]
assert set(g for g, _ in GENES) == set(TR.gene) - {"Cyp2c11"}
# the source protein layer must match the protein matrix used everywhere else
chk = []
for g, _ in GENES:
    v = D.prot(g); v = v - np.nanmean(v[D.AGE == 9])
    s = TR[(TR.gene == g) & (TR.layer == "protein")].set_index("sample").value
    m = pd.Series(v, index=[f"{x}_{a}mo" for x, a in zip(D.SID, D.AGE)]).reindex(s.index)
    chk.append(np.nanmax(np.abs(m - s)))
print("max |protein(source) - protein(matrix)|:", round(float(np.nanmax(chk)), 4))
rng = np.random.default_rng(5)
fig = plt.figure(figsize=(183 * MM, 222 * MM))
gs = fig.add_gridspec(6, 4, hspace=0.75, wspace=0.42, left=0.07, right=0.985, top=0.965, bottom=0.05)
for k, (g, lab) in enumerate(GENES):
    ax = fig.add_subplot(gs[k // 4, k % 4])
    v = D.prot(g); v = v - np.nanmean(v[D.AGE == 9])
    layers = [("Protein", D.AGE, v, PROT_C)]
    r = TR[(TR.gene == g) & (TR.layer == "RNA")]
    if len(r):
        layers.append(("RNA", r.age.to_numpy(int), r.value.to_numpy(float), RNA_C))
    for name, a, y, c in layers:
        okk = np.isfinite(y)
        a, y = a[okk], y[okk]
        ax.scatter(a + rng.uniform(-0.45, 0.45, len(a)), y, s=3.5, c=c, alpha=0.35, lw=0)
        mu = pd.Series(y).groupby(a).mean()
        ax.plot(mu.index, mu.values, "o-", ms=2.2, lw=1, c=c, label=name)
    ax.axhline(0, c=PALE, lw=0.5)
    ax.set_xticks([6, 12, 18, 24]); ax.set_xlim(4, 29)
    ax.set_title(lab, fontsize=6.8, pad=3, color=INK)
    ax.tick_params(labelsize=5.8)
    if k % 4 == 0:
        ax.set_ylabel("log2 vs youngest", fontsize=6.2)
    if k >= len(GENES) - 4:
        ax.set_xlabel("Age (months)", fontsize=6.2)
    if k == 0:
        ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.2, borderaxespad=0.1)
fig.savefig(OUT / "supp_s6_protein_rna_trajectories.pdf"); fig.savefig(OUT / "supp_s6_protein_rna_trajectories.png", dpi=400)
