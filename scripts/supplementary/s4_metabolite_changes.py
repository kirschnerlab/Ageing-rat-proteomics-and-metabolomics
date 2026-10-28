"""Supplementary Figure S4: panel A is the selected-metabolite forest (supp_s4a_selected_metabolites.pdf, drawn by
figure4_supp_s4a_s5_s7.py); panel B shows every metabolite, for perspective on the metabolome as a whole.

B: 25- vs 9-month change against -log10 P for all metabolites, from the same statistics table as panel A
(metabolite_age_statistics.csv: HC3, BH q across all 269 features). As in panel A and the text, NAD+ is the sum of its
two chromatographic peaks (HC3 contrast on the summed signal, nominal P) and the cyclic-ADP-ribose features, which track
the NAD+ signal, are not shown. The 13 panel-A metabolites are outlined and labelled.
Writes outputs/supp_s4/supp_s4_metabolite_changes.pdf and .png (300 dpi) and supp_s4_numbers.json."""
import io, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "shared")]
import paths                                         # noqa: E402
from style import *                                  # noqa: E402,F401
import omics_data as D                               # noqa: E402
from pypdf import PdfReader, PdfWriter, PageObject, Transformation   # noqa: E402
OUT = paths.out("supp_s4")
SRC_A = paths.out(paths.OUT_FIG4) / "supp_s4a_selected_metabolites.pdf"     # run figure4_supp_s4a_s5_s7.py first

S = D.MSTAT.copy()
name = S.reported_name.astype(str)
drop = name.str.contains("NAD", case=True) | name.str.contains("cyclic ADP|cADPR|Cyclic ADP", case=False)
print("not shown as separate points:", sorted(name[drop]))
S = S[~drop]
e, lo, hi, p = D.hc3_25v9(D.met("NAD"))
nad = pd.DataFrame({"reported_name": ["NAD+ (peaks summed)"], "log2FC_25_vs_9": [e], "p_HC3_25_vs_9": [p], "q_HC3_25_vs_9": [np.nan]},
                   index=["MET_NAD_SUM"])
S = pd.concat([S[["reported_name", "log2FC_25_vs_9", "p_HC3_25_vs_9", "q_HC3_25_vs_9"]], nad])
PANEL_A = {"MET_ROW_0173": "NMN", "MET_NAD_SUM": NADP, "MET_ROW_0110": "ADP-ribose", "MET_ROW_0169": "PLP",
           "MET_ROW_0050": "Pantothenate", "MET_ROW_0095": "Creatine", "MET_ROW_0047": "Alanine", "MET_ROW_0208": "Asparagine",
           "MET_ROW_0126": "γ-Glu-Glu", "MET_ROW_0153": "N6-Acetyl-lysine", "MET_ROW_0048": "Neu5Ac",
           "MET_ROW_0261": "TCDCA", "MET_ROW_0186": "Glycocholate"}
assert all(k in S.index for k in PANEL_A), [k for k in PANEL_A if k not in S.index]
x, y = S.log2FC_25_vs_9.to_numpy(float), -np.log10(S.p_HC3_25_vs_9.to_numpy(float))
sig = (S.q_HC3_25_vs_9 < 0.05).to_numpy()
col = np.where(sig & (x > 0), UP, np.where(sig & (x < 0), DOWN, "#c4c7cb"))

fig = plt.figure(figsize=(5.4, 3.0))
ax = fig.add_axes([0.11, 0.15, 0.86, 0.72])
ax.scatter(x, y, s=9, c=col, lw=0, zorder=2)
# labels stacked in one column on each side (lower at 25 months on the left, higher on the right), in the order of the points
pts = [(lab, float(S.loc[f, "log2FC_25_vs_9"]), float(-np.log10(S.loc[f, "p_HC3_25_vs_9"]))) for f, lab in PANEL_A.items()]
for side, xcol, ha_ in ((-1, -3.0, "left"), (1, 2.3, "left")):
    group = sorted([p_ for p_ in pts if np.sign(p_[1]) == side], key=lambda t: -t[2])
    top = 4.9 if side < 0 else 4.7
    for i, (lab, xi, yi) in enumerate(group):
        ax.scatter(xi, yi, s=16, facecolor="none", edgecolor=INK, lw=0.7, zorder=3)
        ax.annotate(lab, (xi, yi), xytext=(xcol, top - i * 0.5), fontsize=5.6, ha=ha_, va="center", color=INK,
                    arrowprops=dict(arrowstyle="-", lw=0.35, color="#999", shrinkA=0, shrinkB=2))
ax.axvline(0, color=PALE, lw=0.6)
ax.set_xlabel("log2 change, 25 vs 9 months")
ax.set_ylabel("−log10 P")
n_up, n_dn = int((sig & (x > 0)).sum()), int((sig & (x < 0)).sum())
ax.text(0.99, 0.97, f"{len(S)} metabolites\n{n_up} higher, {n_dn} lower at 25 months (q < 0.05)", transform=ax.transAxes,
        ha="right", va="top", fontsize=6, color=MUTED)
ax.set_title("All measured metabolites", loc="left", pad=6, fontsize=7.5, color=INK)
buf = io.BytesIO(); fig.savefig(buf, format="pdf"); plt.close(fig)
B = PdfReader(buf).pages[0]
A = PdfReader(str(SRC_A)).pages[0]
wa, ha = float(A.mediabox.width), float(A.mediabox.height)
wb, hb = float(B.mediabox.width), float(B.mediabox.height)
Wp, Hp = max(wa, wb) + 14, ha + hb + 10
page = PageObject.create_blank_page(width=Wp, height=Hp)
page.merge_transformed_page(A, Transformation().translate(14, hb + 10))
page.merge_transformed_page(B, Transformation().translate(14, 0))
lab = plt.figure(figsize=(Wp / 72, Hp / 72)); la = lab.add_axes([0, 0, 1, 1]); la.set_xlim(0, Wp); la.set_ylim(0, Hp); la.axis("off")
la.text(4, Hp - 4, "a", fontsize=9.5, fontweight="bold", va="top")
la.text(4, hb + 2, "b", fontsize=9.5, fontweight="bold", va="top")
lb = io.BytesIO(); lab.savefig(lb, format="pdf", transparent=True); plt.close(lab)
page.merge_page(PdfReader(lb).pages[0])
w = PdfWriter(); w.add_page(page)
pdf = OUT / "supp_s4_metabolite_changes.pdf"
with open(pdf, "wb") as fh:
    w.write(fh)
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(pdf), str(OUT / "supp_s4_metabolite_changes")], check=True)
nums = {"n_shown": int(len(S)), "q<0.05 higher": n_up, "q<0.05 lower": n_dn}
(OUT / "supp_s4_numbers.json").write_text(json.dumps(nums, indent=2), encoding="utf-8")
print(nums)
