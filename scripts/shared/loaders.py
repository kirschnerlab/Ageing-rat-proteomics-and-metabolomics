"""Shared inputs and statistics helpers: rat IDs and ages, the protein matrix (one representative record per gene) and
the metabolite matrix.

Animal universe = the 32 collected rats. Proteins exist for 31 (N146 excluded from proteomics); metabolomes for 31
(P64 excluded). No covariates beyond chronological age group; no imputation or offsets; NAD+ peaks summed; cyclic
ADP-ribose treated as an NAD+ in-source fragment (not analysed separately).
"""
import numpy as np, pandas as pd
from scipy import stats
import paths

OLD = paths.PROTEIN_MATRIX                                  # protein matrix (Supplementary Table 2)
FEATS = paths.METABOLITE_FEATURES
META = paths.SAMPLE_METADATA

a32 = pd.read_csv(paths.METABOLOMICS_ANIMALS).set_index('sample_id')
SID = a32.index.tolist()                                   # 32 rats, e.g. N146, N153, O17, M41
AGE = a32.age_months.astype(int).to_numpy()
meta = pd.read_csv(META)
meta = meta[meta.role.eq('biological_sample')].copy()
meta['sid'] = meta.label.str.split('_').str[0]
LABEL = dict(zip(meta.sid, meta.label))                    # proteomic label per rat (31)
SERIES = np.array([meta.set_index('sid').series.get(s, '') for s in SID])   # TMT labeling set H / D ('' for N146); checks only


def _to_sid(df_label_cols):
    out = df_label_cols.rename(columns={v: k for k, v in LABEL.items()})
    return out.reindex(columns=SID)


def old_proteins():
    """Protein matrix, one gene-representative record per gene; rows = gene symbol, columns = SID (N146 = NaN)."""
    m = pd.read_csv(OLD)
    rep = pd.read_csv(paths.PROTEIN_AGE_STATISTICS, usecols=['protein_id', 'gene', 'selected_gene_representative'])
    m = m.merge(rep, on='protein_id', how='left')
    m = m[m.selected_gene_representative.fillna(False).astype(bool)].set_index('gene')
    return _to_sid(m[list(LABEL.values())]), m[['protein_id', 'n_PSMs']]


def metabolites():
    """log2 metabolites (rows = feature id, columns = SID); P64 = NaN; NAD+ peaks summed as MET_NAD_SUM."""
    mt = pd.read_csv(paths.METABOLITES_ALL, index_col=0).T.reindex(columns=SID)
    mt.loc[:, 'P64'] = np.nan
    mt.loc['MET_NAD_SUM'] = np.log2(2 ** mt.loc['MET_ROW_0082'] + 2 ** mt.loc['MET_ROW_0062'])
    names = pd.read_csv(FEATS).set_index('feature_id').reported_name.to_dict()
    names['MET_NAD_SUM'] = 'NAD+ (peaks 1+2 summed)'
    return mt, names


def partial_spearman(y, x, age, mask=None):
    """Spearman correlation after removing age-group means from the ranks."""
    ok = np.isfinite(y) & np.isfinite(x) if mask is None else (np.isfinite(y) & np.isfinite(x) & mask)
    ry, rx, a = stats.rankdata(y[ok]), stats.rankdata(x[ok]), age[ok]
    for l in set(a):
        ry[a == l] -= ry[a == l].mean(); rx[a == l] -= rx[a == l].mean()
    return float(ry @ rx / np.sqrt((ry @ ry) * (rx @ rx))), int(ok.sum())


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]; o = np.argsort(pv); r = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; qq = np.empty_like(pv); qq[o] = np.minimum(r, 1); q[ok] = qq
    return q
