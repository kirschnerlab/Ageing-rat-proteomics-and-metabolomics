"""Figure 5 — older kidneys diverge along one remodeling axis that the proteome, kidney RNA from other rats and the
metabolome all detect. (a) every kidney relative to its age-mates for marker proteins, with the remodeling score (PC1 of
all proteins, relative to age-mates) on top; (b) the score by age; (c) alignment with the age response, with the clock
proteins of Figure 4D in black; (d) the proteins most associated with each end of the score; (e-f) recurrence in RNA
from 54 other same-colony rats (age-only weights); (g-h) the metabolome's own axis and its metabolites.
Writes outputs/figure5/figure5_remodeling_axis.{pdf,png} and figure5_numbers.json."""
import sys, io, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'shared')]
import paths
from style import *
import omics_data as D
import matplotlib.colors as mcolors
OUT = paths.out('figure5')
AGE, STATE = D.AGE, D.STATE
nums = {}
W, H = 183, 266
fig = plt.figure(figsize=(W * MM, H * MM))
rng = np.random.default_rng(7)


def axes(x0, y0, w, h):
    return fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H])


def letter(x, y, s, text=None):
    fig.text(x / W, 1 - y / H, s, fontsize=9.5, fontweight='bold', va='bottom')
    if text:
        fig.text((x + 4) / W, 1 - y / H, text, fontsize=7.5, va='bottom', color=INK)


ok = np.isfinite(STATE)
cols = [i for a in (9, 18, 21, 25) for i in sorted(np.flatnonzero(ok & (AGE == a)), key=lambda j: STATE[j])]
ages_c = AGE[cols]

# ---------- a: each kidney relative to its age-mates ----------
GROUPS = [('Tubular injury', [('KIM-1', 'Havcr1'), ('NGAL', 'Lcn2'), ('Osteopontin', 'Spp1'), ('VCAM1', 'Vcam1'), ('Vimentin', 'Vim'),
                              ('SOX9', 'Sox9'), ('TIMP1', 'Timp1')]),
          ('Immune', [('CD45', 'Ptprc'), ('CD38', 'Cd38'), ('CD68', 'Cd68')]),
          ('Differentiated\ntubule', [('OAT1', 'Slc22a6'), ('OAT3', 'Slc22a8'), ('NaPi-IIa', 'Slc34a1'), ('SGLT2', 'Slc5a2'),
                                     ('Megalin', 'Lrp2'), ('AQP1', 'Aqp1'), ('OXPHOS (66 proteins)', 'prog:OXPHOS'),
                                     ('Fatty-acid oxidation (50)', 'prog:Fatty-acid oxidation')])]
rows, labels, gstarts = [], [], []
for gname, items in GROUPS:
    gstarts.append((len(rows), gname, len(items)))
    for lab, key in items:
        v = (D.PROGS.loc[key].to_numpy(float) if key.startswith('prog:') else D.prot(key)).copy()
        for a in (9, 18, 21, 25):
            k = AGE == a
            v[k] = v[k] - np.nanmean(v[k])
        rows.append(v[cols]); labels.append(lab)
M = np.array(rows)
x0, hx, hw, hh = 46, 18, 86, 50
axs = axes(x0, 12, hw, 4)
axs.imshow(STATE[cols][None, :], aspect='auto', cmap='PuOr_r', vmin=-2.2, vmax=2.2, interpolation='nearest')
axs.set_xticks([]); axs.set_yticks([0], ['Remodeling score'], fontsize=6.2); axs.tick_params(length=0)
for s in axs.spines.values():
    s.set_visible(False)
axh = axes(x0, hx, hw, hh)
cmap = plt.colormaps['RdBu_r']
axh.imshow(M, aspect='auto', cmap=cmap, vmin=-1.0, vmax=1.0, interpolation='nearest')
axh.set_yticks(range(len(labels)), labels, fontsize=6); axh.tick_params(length=0); axh.set_xticks([])
for s in axh.spines.values():
    s.set_visible(False)
bounds = np.cumsum([np.sum(ages_c == a) for a in (9, 18, 21, 25)])
for b in bounds[:-1]:
    for a_ in (axh, axs):
        a_.axvline(b - 0.5, color='white', lw=2)
for st, gname, n in gstarts[1:]:
    axh.axhline(st - 0.5, color='white', lw=2)
for st, gname, n in gstarts:
    axh.text(-0.33, st + (n - 1) / 2, gname, transform=axh.get_yaxis_transform(), rotation=90, ha='center', va='center',
             fontsize=6.2, fontweight='bold', color=INK)
start = 0
for a, b in zip((9, 18, 21, 25), bounds):
    axh.text((start + b - 1) / 2, len(labels) - 0.1, f'{a} months', ha='center', va='top', fontsize=6.4,
             color=AGEC[a] if a != 9 else '#7a68b0', fontweight='bold')
    start = b
axh.text(0.0, -0.09, 'One column per rat, ordered within each age by its remodeling score', transform=axh.transAxes, ha='left', va='top', fontsize=6)
cax = axes(151, 70, 28, 2)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(-1, 1), cmap=cmap), cax=cax, orientation='horizontal')
cb.set_label('log2 vs age-group mean', fontsize=5.8); cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_visible(False)
letter(4, 8, 'a'); fig.text(x0 / W, 1 - 8 / H, 'Within each age, kidneys differ in one coordinated way', fontsize=7.5, va='bottom', color=INK)

# ---------- b: the score by age ----------
axb = axes(152, 12, 28, 38)
for j, a in enumerate((9, 18, 21, 25)):
    v = STATE[ok & (AGE == a)]
    axb.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=10, c=AGEC[a], edgecolor='white', lw=0.3, zorder=3)
    axb.plot([j - 0.24, j + 0.24], [0, 0], color=INK, lw=0.9)
sdr = np.std(STATE[ok & (AGE == 25)], ddof=1) / np.std(STATE[ok & (AGE == 9)], ddof=1)
axb.axhline(0, color=PALE, lw=0.6)
axb.set_xticks(range(4), [9, 18, 21, 25]); axb.set_xlabel('Age (months)'); axb.set_ylabel('Remodeling score (SD)')
axb.text(0.04, 0.97, f'spread {sdr:.1f}× wider\nat 25 than 9 months', transform=axb.transAxes, fontsize=5.8, va='top')
title(axb, 'b', 'Old kidneys diverge', dx=-0.42)
nums['5b sd_ratio'] = round(float(sdr), 2)
h_ = [plt.Line2D([], [], marker='o', ls='', color=AGEC[a], ms=4, label=f'{a} months') for a in (9, 18, 21, 25)]
fig.legend(handles=h_, loc='center', bbox_to_anchor=(166 / W, 1 - 61.5 / H), ncol=2, frameon=False, fontsize=6, handletextpad=0.2, columnspacing=0.8)

# ---------- c: the age response is the same biology ----------
rec = pd.read_csv(paths.PROTEIN_AGE_SCORE_STATS)
axc = axes(14, 94, 42, 38)
axc.scatter(rec.effect, rec.full_r, s=1.4, c='#c9ccd0', lw=0, rasterized=True)
sr = stats.spearmanr(rec.effect, rec.full_r)[0]
for g, lab, xy in [('Havcr1', 'KIM-1', (2.6, 1.12)), ('Cd38', 'CD38', (1.2, 1.12)), ('Slc22a6', 'OAT1', (-2.0, -1.0))]:
    r = D.pstat(g)
    axc.scatter(r.effect, r.full_r, s=11, c=UP if r.full_r > 0 else DOWN, edgecolor='white', lw=0.3, zorder=3)
    axc.annotate(lab, (r.effect, r.full_r), xytext=xy, fontsize=6, ha='center', va='center', arrowprops=dict(arrowstyle='-', lw=0.4, color='#777'))
# every protein the clocks use at least five times (Figure 4D), in black
CLOCK = [('Hs3st3a1', 'HS3ST3A/B'), ('Timp3', 'TIMP3'), ('Nid1', 'NID1'), ('Mmp19', 'MMP19'), ('Agrn', 'AGRN'), ('Stk35', 'STK35'),
         ('Rnaset2', 'RNASET2')]
CLOCK_XY = {'HS3ST3A/B': (2.8, 0.24), 'TIMP3': (1.9, -0.92), 'NID1': (-0.3, 0.88), 'MMP19': (2.4, 0.66), 'AGRN': (-1.4, 0.55),
            'STK35': (2.9, -0.45), 'RNASET2': (-0.9, -0.42)}
for g, lab in CLOCK:
    r = D.pstat(g)
    axc.scatter(r.effect, r.full_r, s=12, c=INK, edgecolor='white', lw=0.3, zorder=4)
    axc.annotate(lab, (r.effect, r.full_r), xytext=CLOCK_XY[lab], fontsize=5.6, ha='center', va='center', color=INK,
                 arrowprops=dict(arrowstyle='-', lw=0.4, color='#777'))
    nums[f'5c clock {lab}'] = {'effect_25v9': round(float(r.effect), 3), 'r_with_score': round(float(r.full_r), 3)}
axc.scatter([], [], s=12, c=INK, label='clock protein (Fig. 4D)')
axc.legend(loc='lower right', frameon=False, fontsize=5.4, handletextpad=0.1, borderaxespad=0.1)
axc.axhline(0, color=PALE, lw=0.5); axc.axvline(0, color=PALE, lw=0.5)
axc.set(xlim=(-3, 4.3), ylim=(-1.15, 1.25), yticks=[-1, -0.5, 0, 0.5, 1])
axc.set_xlabel('Change with age, 25 vs 9 months (log2)'); axc.set_ylabel('Association with remodeling score (r)')
axc.text(0.03, 0.97, f'ρ = {sr:.2f}', transform=axc.transAxes, fontsize=6, va='top')
title(axc, 'c', 'Proteins that change with age\nseparate kidneys of the same age', dx=-0.3)
nums['5c'] = {'spearman': round(float(sr), 3), 'n_proteins': int(len(rec))}

# ---------- d: proteins that move most with the score ----------
# size of change (log2 per SD of score, relative to age-mates) among proteins that track the score (|r| >= 0.5)
X = D.P.to_numpy(float)[:, ok]
fin = np.isfinite(X).all(1)
X, gn = X[fin], D.P.index[fin]
for a in (9, 18, 21, 25):
    k = AGE[ok] == a
    X[:, k] -= X[:, k].mean(1, keepdims=True)
sc = STATE[ok] - STATE[ok].mean()
g = pd.DataFrame({'slope': X @ sc / (sc @ sc), 'r': X @ sc / np.sqrt((X ** 2).sum(1) * (sc @ sc))}, index=gn)
g = g[~g.index.str.contains(r'LOC|ENSRNOG|RGD|^Ig[hkl]|^RT1|\.|;|_', regex=True) & (g.r.abs() >= 0.5)]
neg = g.sort_values('slope').head(12)
pos = g.sort_values('slope', ascending=False).head(12)
COMMON = {'Havcr1': 'KIM-1', 'Lcn2': 'NGAL', 'Spp1': 'osteopontin', 'Ctsg': 'cathepsin G', 'Slc34a3': 'NaPi-IIc'}
letter(66, 90, 'd', 'Proteins most associated with the remodeling score')
for k, (tab, col, x, ttl) in enumerate([(neg, DOWN, 86, 'lower in remodeled kidneys'), (pos, UP, 152, 'higher in remodeled kidneys')]):
    ax = axes(x, 96, 24, 36)
    yy = np.arange(len(tab))
    ax.barh(yy, tab.slope.abs(), color=col, height=0.7)
    ax.set_yticks(yy, [s.upper() + (f' ({COMMON[s]})' if s in COMMON else '') for s in tab.index], fontsize=5.9)
    ax.invert_yaxis(); ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel('log2 change per SD\nof score', fontsize=6.2)
    ax.set_title(ttl, fontsize=6.4, color=col, pad=3)
    nums[f'5d {ttl}'] = [(gg.upper(), round(float(s), 3), round(float(r), 3)) for gg, s, r in zip(tab.index, tab.slope, tab.r)]

# ---------- e-f: the same axis in RNA from 54 other rats ----------
rp = pd.read_csv(paths.RNA_SCORE_BY_RAT)
rp = rp[rp.method == 'current_unweighted_300'].copy()
m6, s6 = rp[rp.age == 6].score.mean(), rp[rp.age == 6].score.std(ddof=1)
rp['z'] = (rp.score - m6) / s6
RAGES = [6, 12, 18, 21, 27]
RC = {6: '#d7cdee', 12: '#b9a7e3', 18: '#8a6fd1', 21: '#5b3ea8', 27: '#2e1a66'}
axe = axes(14, 152, 50, 36)
for j, a in enumerate(RAGES):
    v = rp[rp.age == a].z.to_numpy()
    axe.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=9, c=RC[a], edgecolor='white', lw=0.3, zorder=3)
    axe.plot([j - 0.24, j + 0.24], [v.mean()] * 2, color=INK, lw=0.9)
axe.set_xticks(range(5), RAGES); axe.set_xlabel('Age (months)'); axe.set_ylabel('RNA remodeling score\n(SD of 6-month rats)')
axe.text(0.04, 0.97, '54 other rats,\nkidney RNA', transform=axe.transAxes, fontsize=5.8, va='top')
title(axe, 'e', 'Remodeling score in kidney RNA from 54 other rats', dx=-0.2)
nums['5e means'] = {int(a): round(float(rp[rp.age == a].z.mean()), 2) for a in RAGES}
ma = pd.read_csv(paths.RNA_SCORE_MARKERS)
ma = ma[ma.method == 'current_unweighted_300'].set_index('target')
EL = [('KIM-1', 'HAVCR1'), ('NGAL', 'LCN2'), ('CD45', 'PTPRC_CD45'), ('TIMP1', 'TIMP1'), ('Adaptive PT genes', 'Lake_aPT'),
      ('Adaptive TAL genes', 'Lake_aTAL'), ('OAT1', 'SLC22A6_OAT1')]
axf = axes(122, 152, 50, 36)
for i, (lab, key) in enumerate(EL):
    r = float(ma.loc[key, 'target_excluded_within_age_r'])
    axf.barh(i, r, color=UP if r > 0 else DOWN, height=0.62)
    nums[f'5f {lab}'] = round(r, 3)
axf.set_yticks(range(len(EL)), [e[0] for e in EL], fontsize=6); axf.invert_yaxis(); axf.tick_params(axis='y', length=0)
axf.axvline(0, color=PALE, lw=0.7); axf.set_xlim(-1, 1)
axf.set_xlabel('Correlation with RNA score\namong age-mates')
letter(86, 148, 'f', 'RNA score versus injury and immune transcripts')

# ---------- g-h: the metabolome's own axis ----------
f5 = pd.read_csv(paths.FIG5G_AXES)
axg = axes(14, 208, 50, 40)
for a in (9, 18, 21, 25):
    k = f5.age == a
    axg.scatter(f5.proteome_STATE[k], f5.metabolome_axis[k], s=11, c=AGEC[a], edgecolor='white', lw=0.3, zorder=3)
axg.axhline(0, color=PALE, lw=0.5); axg.axvline(0, color=PALE, lw=0.5)
rr = stats.pearsonr(f5.proteome_STATE, f5.metabolome_axis)[0]
per_age = {a: stats.spearmanr(f5.proteome_STATE[f5.age == a], f5.metabolome_axis[f5.age == a])[0] for a in (9, 18, 21, 25)}
axg.text(0.03, 0.97, f'r = {rr:.2f}', transform=axg.transAxes, fontsize=6, va='top')
axg.text(0.97, 0.03, 'within each age\n' + '\n'.join(f'{a} mo  ρ {per_age[a]:+.2f}' for a in (9, 18, 21, 25)), transform=axg.transAxes,
         fontsize=5.4, ha='right', va='bottom', color=MUTED)
axg.set_xlabel('Protein remodeling score (SD)'); axg.set_ylabel('Metabolite score (SD)')
title(axg, 'g', 'Metabolite axis versus protein remodeling score', dx=-0.2)
nums['5g'] = {'r': round(float(rr), 3), 'per_age': {a: round(float(v), 2) for a, v in per_age.items()}}
ld = pd.read_csv(paths.FIG5H_LOADINGS, index_col=0)
a_ = ld[ld.tag.eq('A')].sort_values('r')
sel = pd.concat([a_.head(8), a_.tail(8)])
clean = {'L(-)-Carnitine': 'Carnitine', 'L-Ergothioneine': 'Ergothioneine', 'DL-4-Hydroxyphenyllactic acid': '4-Hydroxyphenyllactate',
         "Cytidine 5'-monophosphate (hydrate)": 'CMP', "Pyridoxal 5'-phosphate": 'Pyridoxal 5′-phosphate',
         'N-Isovalerylglycine; peak2 (indistinguishable MS2)': 'N-Isovalerylglycine', 'Acetamidobenzoic acid': 'Acetamidobenzoate',
         'Ferulic acid': 'Ferulate', 'NAD+ (peaks summed)': NADP}
labs = [clean.get(n, str(n).split(';')[0]) for n in sel.name]
yy = np.r_[np.arange(8), np.arange(8) + 8.9]
axm = axes(128, 206, 44, 46)
axm.barh(yy, sel.r, color=[DOWN if r < 0 else UP for r in sel.r], height=0.72)
axm.set_yticks(yy, labs, fontsize=5.9); axm.set_ylim(yy.max() + 0.6, -0.6); axm.tick_params(axis='y', length=0)
axm.axvline(0, color=PALE, lw=0.6); axm.set(xlim=(-1, 1), xticks=[-1, 0, 1])
axm.set_xlabel('Correlation with metabolite score (r)', fontsize=6.3)
axm.text(0.97, 0.78, 'reabsorbed\nnutrients', transform=axm.transAxes, ha='right', fontsize=5.8, color=DOWN)
axm.text(0.03, 0.18, 'retained\nsolutes', transform=axm.transAxes, ha='left', fontsize=5.8, color=UP)
letter(86, 202, 'h', 'Metabolites at each end of the metabolite axis')
fig.savefig(OUT / 'figure5_remodeling_axis.pdf'); fig.savefig(OUT / 'figure5_remodeling_axis.png', dpi=300)
(OUT / 'figure5_numbers.json').write_text(json.dumps(nums, indent=2, default=float), encoding='utf-8')
print(json.dumps({k: v for k, v in nums.items() if k.startswith('5d')}, indent=0))

