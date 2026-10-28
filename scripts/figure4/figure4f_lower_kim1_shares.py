"""Panel data for Fig 4f: how much of each age change is already present in old kidneys with lower KIM-1?

For each protein, program and metabolite: (mean of the lower-KIM-1 half of 21- and 25-month rats - 9-month mean) /
(mean of all 21- and 25-month rats - 9-month mean), a ratio of subgroup-average changes. Each old age group is split at
its own KIM-1 median; rats at the median (ties) go to the lower half. The lower-KIM-1 halves are the same rats as the
less-remodeled halves by the Fig 5 score (8 of 8 at each age). 1 = the lower-KIM-1 half shows the same average change as
all old kidneys; 0 = no average change in that half. 95% interval: 2.5th-97.5th percentiles of 2,000 bootstrap resamples of
rats within age, with the KIM-1 median recomputed in every resample. Also writes the old (21-25 months) vs 9-month fold
change.

Program rows and CYP2C11 use exactly the signals plotted in Fig 4b (figure4b_rows.csv): collagen IV alpha3-alpha5 =
mean of COL4A3/4/5; terminal complement = the six-member set including C8G; CYP2C11 = reconstruction from its 11
gene-unique sequences.
Writes outputs/figure4f_lower_kim1_shares/figure4f_lower_kim1_shares.csv; figure4_supp_s4a_s5_s7.py draws it
(it reads the shipped copy in data/intermediate/)."""
import sys, io
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'shared')]
import paths
import omics_data as D
AGE, STATE = D.AGE, D.STATE
KIM1 = D.prot('Havcr1')
rng = np.random.default_rng(20261027)
F4B = pd.read_csv(paths.FIG4B_ROWS, index_col=0).reindex(columns=list(D.SID))     # Fig 4b signals, by rat
ROWS = [('Proteins', 'HS3ST3A/B', 'p', 'Hs3st3b1'), ('Proteins', 'TIMP3', 'p', 'Timp3'),
        ('Proteins', 'NID1', 'p', 'Nid1'), ('Proteins', 'MMP19', 'p', 'Mmp19'),
        ('Proteins', 'Collagen IV α3–α5', 'f4b', 'GBM-associated collagen IV (3)'),
        ('Proteins', 'CYP2C11', 'f4b', 'CYP2C11'), ('Proteins', 'Terminal complement', 'f4b', 'Terminal complement (6)'),
        ('Proteins', 'CD45', 'p', 'Ptprc'), ('Proteins', 'CD38', 'p', 'Cd38'),
        ('Proteins', 'Proximal tubule', 'f4b', 'Proximal tubule (66)'), ('Proteins', 'OXPHOS', 'f4b', 'Oxidative phosphorylation (66)'),
        ('Metabolites', 'Taurochenodeoxycholate', 'm', 'TCDCA'), ('Metabolites', 'Creatine', 'm', 'CREATINE'), ('Metabolites', 'Alanine', 'm', 'ALA'),
        ('Metabolites', 'Pantothenate', 'm', 'PANTO'), ('Metabolites', 'Pyridoxal 5′-phosphate', 'm', 'PLP'), ('Metabolites', 'Neu5Ac', 'm', 'NEU5AC'),
        ('Metabolites', 'NMN', 'm', 'NMN'), ('Metabolites', 'ADP-ribose', 'm', 'ADPR')]


def value(kind, key):
    if kind == 'f4b':
        return F4B.loc[key].to_numpy(float)
    return {'p': D.prot, 'm': D.met}[kind](key)


def low_share(v, idx):
    vv, aa, ss = v[idx], AGE[idx], KIM1[idx]
    m9 = np.nanmean(vv[aa == 9]); lo, al = [], []
    for a in (21, 25):
        k = (aa == a) & np.isfinite(vv) & np.isfinite(ss)
        med = np.median(ss[k]); lo.append(vv[k & (ss <= med)]); al.append(vv[k])
    lo, al = np.concatenate(lo), np.concatenate(al)
    return (lo.mean() - m9) / (al.mean() - m9)


ok = np.isfinite(KIM1)
for a in (21, 25):           # the KIM-1 halves are the remodeling-score halves
    k = ok & (AGE == a)
    same = ((KIM1[k] <= np.median(KIM1[k])) == (STATE[k] <= np.median(STATE[k]))).sum()
    print(f'{a} months: lower-KIM-1 half = less-remodeled half for {same} of {k.sum()} rats')
base = np.flatnonzero(ok & np.isin(AGE, (9, 21, 25)))
boots = [np.concatenate([rng.choice(base[AGE[base] == a], (AGE[base] == a).sum()) for a in (9, 21, 25)]) for _ in range(2000)]
out = []
for grp, lab, kind, key in ROWS:
    v = value(kind, key)
    est = low_share(v, base)
    bs = np.array([low_share(v, b) for b in boots])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    old = np.isin(AGE, (21, 25)) & np.isfinite(v)
    out.append({'group': grp, 'label': lab, 'signal': key, 'share': est, 'lo': lo, 'hi': hi,
                'old_vs_9_log2': np.nanmean(v[old]) - np.nanmean(v[AGE == 9]),
                'rises_with_age': bool(np.nanmean(v[AGE == 25]) > np.nanmean(v[AGE == 9]))})
T = pd.DataFrame(out)
T.to_csv(paths.out('figure4f_lower_kim1_shares') / 'figure4f_lower_kim1_shares.csv', index=False)
print(T.round(3).to_string(index=False))
