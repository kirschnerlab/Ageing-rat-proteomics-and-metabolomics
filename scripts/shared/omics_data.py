"""Inputs shared by the figure scripts. Proteome: the 23-fraction protein matrix (Supplementary Table 2; 11,485 complete records, one
representative record per gene). Metabolome: 31 rats, P64 excluded, NAD+ peaks summed, cyclic-ADP-ribose rows not
used. STATE (remodeling score): PC1 of the age-centred, SD-scaled proteome, oriented to CD45. Age group is the only
grouping term anywhere."""
import numpy as np, pandas as pd
from scipy import stats
import paths
import loaders as C

SID, AGE, SERIES = np.array(C.SID), C.AGE, C.SERIES
P, PINFO = C.old_proteins()
PINFO = PINFO.rename(columns={'n_PSMs': 'n_PSMs_used'})
PROGS = pd.read_csv(paths.PROGRAM_SCORES, index_col=0).reindex(columns=SID)
ALIAS = {'Hs3st3b1': 'Hs3st3a1'}       # the matrix reports the HS3ST3A1/B1 signal under Hs3st3a1
MT, MNAMES = C.metabolites()
MSTAT = pd.read_csv(paths.METABOLITE_AGE_STATISTICS).set_index('feature_id')
_rec = pd.read_csv(paths.PROTEIN_AGE_SCORE_STATS)
PSTAT = _rec[_rec.rep.astype(bool)].drop_duplicates('gene').set_index('gene')   # Fig 4 age statistics (HC3, BH over all records)
MET = {'NMN': 'MET_ROW_0173', 'NAM': 'MET_ROW_0014', 'NAD': 'MET_NAD_SUM', 'ADPR': 'MET_ROW_0110', 'PLP': 'MET_ROW_0169',
       'PL': 'MET_ROW_0163', 'FMN': 'MET_ROW_0194', 'PANTO': 'MET_ROW_0050', 'MTHF': 'MET_ROW_0263', 'CREATINE': 'MET_ROW_0095',
       'NEU5AC': 'MET_ROW_0048', 'IS': 'MET_ROW_0036', 'TCDCA': 'MET_ROW_0261', 'ALA': 'MET_ROW_0047', 'CREATININE': 'MET_ROW_0033'}


def gene(g):
    return ALIAS.get(g, g)


def _state():
    """Same-age axis exactly as in Fig 5: PC1 of the age-centred, SD-scaled matrix of all 11,485 records."""
    ok = np.isfinite(P.loc['Havcr1'].to_numpy(float))
    full = pd.read_csv(C.OLD)
    Y = full[[C.LABEL[s] for s in SID[ok]]].to_numpy(float)
    a = AGE[ok]
    R = Y.copy()
    for l in set(a):
        R[:, a == l] -= R[:, a == l].mean(1, keepdims=True)
    R = R / R.std(1, ddof=1, keepdims=True)
    U, Sv, _ = np.linalg.svd(R.T, full_matrices=False)
    s = U[:, 0] * Sv[0]
    if np.corrcoef(s, P.loc['Ptprc'].to_numpy(float)[ok])[0, 1] < 0:      # oriented toward higher CD45
        s = -s
    out = np.full(len(SID), np.nan); out[ok] = s / s.std(ddof=1)
    return out, float(Sv[0] ** 2 / (Sv ** 2).sum())


STATE, STATE_VAR = _state()


def met(key):
    return MT.loc[MET.get(key, key)].to_numpy(float)


def prot(g):
    return P.loc[gene(g)].to_numpy(float)


def pstat(g):
    return PSTAT.loc[gene(g)]


def n_psm(g):
    return int(PINFO.n_PSMs_used.get(gene(g), 0))


def hc3_25v9(v):
    """25- vs 9-month difference from the saturated categorical-age model with HC3 standard errors (df = N - 4)."""
    v = np.asarray(v, float); ok = np.isfinite(v)
    out = {}
    for a in (9, 25):
        x = v[ok & (AGE == a)]; n = len(x); e = x - x.mean()
        out[a] = (x.mean(), (e ** 2 / (1 - 1 / n) ** 2).sum() / n ** 2)
    d = out[25][0] - out[9][0]; se = np.sqrt(out[25][1] + out[9][1]); df = ok.sum() - 4
    t = stats.t.ppf(0.975, df)
    return d, d - t * se, d + t * se, 2 * stats.t.sf(abs(d / se), df)


def age_adj(x, y, mask=None):
    """Age-adjusted association as in Fig 5: correlation of within-age rank residuals (= C.partial_spearman)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        ok &= mask
    return C.partial_spearman(np.asarray(x, float)[ok], np.asarray(y, float)[ok], AGE[ok])[0], int(ok.sum())
