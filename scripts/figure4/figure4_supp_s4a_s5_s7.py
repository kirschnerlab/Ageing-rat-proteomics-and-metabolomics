"""Figure 4 and Supplementary Figures S4A, S5 and S7 from the protein matrix (Supplementary Table 2) with age-only models.

    python scripts/figure4/figure4_supp_s4a_s5_s7.py

Rules: no covariates or normalization terms; TMT labeling set (H/D) is used only to check associations. No imputation
or offsets. Metabolomics: P64 excluded; NAD+ peaks summed; cyclic ADP-ribose features removed. Reported values are
asserted against reference values (check(..., expect=...)).
Writes to outputs/figure4_supp_s4a_s5_s7/: figure4, supp_s4a_selected_metabolites, supp_s5_protein_changes,
supp_s7_extracellular_proteins (pdf/png/svg); panel tables; number_checks.csv; protein_age_and_score_statistics.csv.
It also draws metabolome_axis_overview (not used in the paper), whose tables feed Figure 5G and H.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import paths
OUT = paths.out(paths.OUT_FIG4)
OLD_MATRIX = paths.PROTEIN_MATRIX
FASTA = paths.RAT_FASTA

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": "#3a3a3a", "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True, "mathtext.default": "regular",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})
AGEC = {9: "#b9a7e3", 18: "#8a6fd1", 21: "#5b3ea8", 25: "#2e1a66"}
AGES = [9, 18, 21, 25]
UP, DOWN, NS = "#eb6834", "#2a78d6", "#9a9a9a"
INK, MUTED, PALE, RULE = "#1f1f1f", "#6b6b6b", "#c4c7cb", "#bdbdbd"
REGION = {"secreted_ectodomain": UP, "post_hepsin_tail": "#4d4d4d", "boundary_spanning": "#b8b8b8"}
NAD = "NAD$^+$"
ARROW = dict(arrowstyle="-", lw=.4, color="#777", shrinkA=0, shrinkB=1.5)
CHECKS, LOG = [], []
CHECK_REFERENCE = True          # assert the reference values given as expect=


def check(fig, panel, quantity, value, source, kind, expect=None, tol=.0015):
    if CHECK_REFERENCE and expect is not None:
        assert abs(value - expect) <= tol, f"{fig}{panel} {quantity}: {value} != reference {expect}"
    CHECKS.append(dict(figure=fig, panel=panel, quantity=quantity, value=float(value), kind=kind, source=source))


def log(msg):
    LOG.append(msg)
    print("LOG:", msg)


def panel(ax, letter, title, dx=-0.16, size=7.5):
    ax.set_title(title, loc="left", pad=6, fontsize=size, color=INK)
    ax.text(dx, 1.0, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="left")


def short(label):
    return label.split("_")[0]


def tokens(s):
    return {t.strip().casefold() for t in re.split(r"[;,/]", str(s)) if t.strip() and t.strip().lower() != "nan"}


def bh(p):
    p = np.asarray(p, float)
    ix = np.argsort(p)
    q = np.empty(len(p))
    q[ix] = np.minimum.accumulate((p[ix] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1].clip(0, 1)
    return q


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ================================================================ metadata
meta = pd.read_csv(paths.SAMPLE_METADATA).query("role=='biological_sample'").set_index("label")
OLD = pd.read_csv(OLD_MATRIX)
LABELS = [c for c in OLD.columns if c.endswith("mo")]
assert len(LABELS) == 31
RATS = [short(c) for c in LABELS]
AGE_OF = pd.Series(meta.loc[LABELS, "age_months"].astype(int).values, index=RATS)
SERIES = pd.Series(meta.loc[LABELS, "series"].values, index=RATS)
AGEV = AGE_OF.values
X = np.column_stack([np.ones(31)] + [(AGEV == a).astype(float) for a in (18, 21, 25)])
DF = 31 - 4
TCRIT = stats.t.ppf(.975, DF)

# ================================================================ record annotation: defines which records the figures use
pm = pd.read_csv(paths.PROTEIN_ANNOTATION).set_index("protein_id")
OLDANN = pd.DataFrame({"gene": pm.loc[OLD.protein_id, "gene"].values, "rep": pm.loc[OLD.protein_id, "selected_gene_representative"].values,
                       "n_psm": OLD.n_PSMs.values, "n_pf": OLD.n_peptide_forms.values, "n_runs": OLD.n_runs.values,
                       "tok": [tokens(a) | tokens(b) for a, b in zip(OLD.source_gene, OLD.reference_gene_symbol)]}, index=OLD.protein_id)
OLDREP = {g: p for p, g in OLDANN.loc[OLDANN.rep, "gene"].items()}            # gene -> representative protein_id


def old_choose(gene):
    """Record chosen for a named gene: any record carrying the gene token, most peptide forms, PSMs, runs."""
    m = OLDANN[OLDANN.tok.map(lambda t: gene.casefold() in t)]
    return m.sort_values(["n_pf", "n_psm", "n_runs"], ascending=False).index[0] if len(m) else None


# ================================================================ protein matrix used for all analyses
Y = OLD.set_index("protein_id")[LABELS]
Y.columns = RATS
ANN = OLDANN.copy()
ANN["n_seq"] = np.nan
N_QUANT, N_POS, N_ACCEPTED = len(OLD), len(OLD), 11528
REP = {g.casefold(): i for i, g in ANN.loc[ANN.rep, "gene"].items()}


def cur(old_id, gene=None, why=""):
    """Record in the matrix (identity; None if absent)."""
    if old_id is None:
        return None
    return old_id if old_id in Y.index else None


def gene_rec(gene, why=""):
    """Record for a named gene (see old_choose)."""
    return cur(old_choose(gene), gene, why)


def rep_rec(gene, why=""):
    return cur(OLDREP.get(gene), gene, why)


def series(rid):
    v = Y.loc[rid]
    return v - v[AGE_OF.reindex(v.index) == 9].mean()


# ================================================================ age model (all records), STATE
B = np.linalg.lstsq(X, Y.values.T, rcond=None)[0]                 # 4 x records
XP = np.linalg.pinv(X)
H = (X * XP.T).sum(1)
RES_ = Y.values.T - X @ B


def hc3_cols(resid):
    return np.einsum("ki,li,ip->pkl", XP, XP, (resid / (1 - H[:, None])) ** 2)


COV = hc3_cols(RES_)
SE = np.sqrt(np.diagonal(COV, axis1=1, axis2=2)).T
PV = 2 * stats.t.sf(np.abs(B / SE), DF)
AGE = pd.DataFrame({"effect": B[3], "se": SE[3], "lo": B[3] - TCRIT * SE[3], "hi": B[3] + TCRIT * SE[3], "p": PV[3], "q": bh(PV[3])}, index=Y.index)
AGE["gene"] = ANN.loc[AGE.index, "gene"].values
AGE["rep"] = ANN.loc[AGE.index, "rep"].values
Z = RES_ / RES_.std(axis=0, ddof=1)
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
ptprc = rep_rec("Ptprc", "STATE orientation")
j = Y.index.get_loc(ptprc)
sgn = np.sign(np.corrcoef(Z[:, j], U[:, 0])[0, 1])
SCORE = pd.Series(U[:, 0] * S[0] * sgn, index=RATS)
STATE = SCORE / SCORE.std(ddof=1)
FRAC = float(S[0] ** 2 / (S ** 2).sum())
zc = Z - Z.mean(0)
sc = (SCORE - SCORE.mean()).values
AGE["full_r"] = (zc * sc[:, None]).sum(0) / np.sqrt((zc ** 2).sum(0) * (sc ** 2).sum())


def fit1(y):
    y = np.asarray(y, float)
    b = XP @ y
    r = y - X @ b
    cov = XP @ np.diag((r / (1 - H)) ** 2) @ XP.T
    out = {}
    for k, a in enumerate((18, 21, 25), 1):
        se = np.sqrt(cov[k, k])
        out[a] = dict(effect=b[k], lo=b[k] - TCRIT * se, hi=b[k] + TCRIT * se, p=2 * stats.t.sf(abs(b[k] / se), DF))
    return out


# ================================================================ metabolomics
feat = pd.read_csv(paths.METABOLITE_FEATURES).set_index("feature_id")
NAD_PEAKS, CADPR = ["MET_ROW_0082", "MET_ROW_0062"], ["MET_ROW_0099", "MET_ROW_0083"]
MID = dict(NMN="MET_ROW_0173", ADPR="MET_ROW_0110", PLP="MET_ROW_0169", PANTO="MET_ROW_0050", CREATINE="MET_ROW_0095",
           NEU5AC="MET_ROW_0048", IS="MET_ROW_0036", TCDCA="MET_ROW_0261", GCA="MET_ROW_0186", ALA="MET_ROW_0047",
           ASN="MET_ROW_0208", GGLU="MET_ROW_0126", ACLYS="MET_ROW_0153")


def add_nad(m):
    m = m.copy()
    m["NAD_SUM"] = np.log2((2 ** m[NAD_PEAKS]).sum(axis=1))
    return m.drop(columns=NAD_PEAKS + CADPR)


MET32 = add_nad(pd.read_csv(paths.METABOLITES_ALL).set_index("sample_id"))
MATCH = add_nad(pd.read_csv(paths.METABOLITES_PROTEOMICS_RATS, index_col=0)).drop(index="P64")
AGE32 = pd.Series([AGE_OF.get(i, np.nan) for i in MET32.index], index=MET32.index)
AGE32["N146"] = 9
MST_STATS = pd.read_csv(paths.METABOLITE_AGE_STATISTICS).set_index("feature_id")   # P64 excluded


def hc3_25v9(y, a):
    Xm = np.column_stack([np.ones(len(a))] + [(a == g).astype(float) for g in (18, 21, 25)])
    XtXi = np.linalg.inv(Xm.T @ Xm)
    b = XtXi @ Xm.T @ y
    e = y - Xm @ b
    h = np.einsum("ij,jk,ik->i", Xm, XtXi, Xm)
    cov = XtXi @ (Xm.T * (e ** 2 / (1 - h) ** 2)) @ Xm @ XtXi
    se = np.sqrt(cov[3, 3])
    t = stats.t.ppf(.975, len(y) - 4)
    return b[3], b[3] - t * se, b[3] + t * se, 2 * stats.t.sf(abs(b[3] / se), len(y) - 4)


def arank(x, a):
    r = pd.Series(stats.rankdata(x), index=x.index)
    return r - r.groupby(a).transform("mean")


def pair_stats(x, y):
    a = AGE_OF[x.index]
    rho = stats.spearmanr(x, y).statistic
    ra, pa = stats.pearsonr(arank(x, a), arank(y, a))
    chem = {}
    for s in ("H", "D"):
        ix = [i for i in x.index if SERIES[i] == s]
        chem[s] = stats.pearsonr(arank(x[ix], a[ix]), arank(y[ix], a[ix])).statistic
    return rho, ra, pa, chem


def metabolome_axis():
    a = AGE_OF.reindex(MATCH.index)
    R = MATCH - MATCH.groupby(a).transform("mean")
    Zm = R / R.std(ddof=1)
    Zm = Zm.loc[:, Zm.std() > 0]
    u, s, _ = np.linalg.svd(Zm.values - Zm.values.mean(0), full_matrices=False)
    pc = pd.Series(u[:, 0] * s[0], index=MATCH.index)
    st = STATE.reindex(pc.index)
    if stats.pearsonr(pc, st).statistic < 0:
        pc = -pc
    pc = pc / pc.std(ddof=1)
    load = Zm.apply(lambda c: stats.pearsonr(c, pc).statistic)
    rng = np.random.default_rng(20261027)
    obs = abs(stats.pearsonr(pc, st).statistic)
    null = np.array([abs(np.corrcoef(pc, st.groupby(a).transform(lambda v: rng.permutation(v.values)))[0, 1]) for _ in range(9999)])
    return pc, s[0] ** 2 / (s ** 2).sum(), load, (1 + (null >= obs).sum()) / 10000


# ================================================================ FASTA mapping (UMOD regions, CYP2C11 gene-unique peptides)
_FASTA = None


def fasta():
    global _FASTA
    if _FASTA is None:
        recs, head, seq = [], None, []
        with FASTA.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if head is not None:
                        recs.append((head, "".join(seq)))
                    head, seq = line[1:], []
                else:
                    seq.append(line)
        recs.append((head, "".join(seq)))
        rows = []
        for h, s in recs:
            tok = h.split()[0]
            if tok.startswith("##") or "contaminant" in h.lower():
                continue
            gm = re.search(r"\bGN=([^ ]+)", h)
            rows.append((tok.split("|")[1] if "|" in tok else tok, (gm.group(1) if gm else "").casefold(), s.replace("I", "L")))
        _FASTA = rows
    return _FASTA


def map_seqs(seqs, acc=None):
    """For each sequence: target genes containing it, and start/end in the given accession (I/L equivalent)."""
    out = {}
    for sq in seqs:
        n = sq.replace("I", "L")
        genes, start = set(), None
        for a, g, s in fasta():
            k = s.find(n)
            if k >= 0:
                genes.add(g)
                if a == acc and start is None:
                    start = k + 1
        out[sq] = dict(genes=genes, start=start, end=None if start is None else start + len(sq) - 1)
    return out


def backbone(form):
    return re.sub("[^A-Z]", "", str(form))


def load_peptides(protein_ids=None, gene_token=None):
    """Peptide forms (log2, 9-month centred) with sequence and PSM counts."""
    p = pd.read_csv(paths.PEPTIDE_LOG2)
    p["sequence"] = p.peptide_form.map(backbone)
    p = p.rename(columns={"protein_id": "rid"})
    tok = [tokens(a) | tokens(b) for a, b in zip(p.source_gene, p.reference_gene_symbol)]
    p["tok"] = tok
    p = p.rename(columns=dict(zip(LABELS, RATS)))
    if protein_ids is not None:
        p = p[p.rid.isin(protein_ids)]
    if gene_token is not None:
        p = p[p.tok.map(lambda t: gene_token in t)]
    return p[np.isfinite(p[RATS]).all(axis=1)].copy()


def seq_profiles(p):
    out = []
    for sq, blk in p.groupby("sequence"):
        out.append(dict(sequence=sq, n_forms=len(blk), n_PSMs=int(blk.n_PSMs.sum()),
                        **dict(zip(RATS, np.average(blk[RATS].to_numpy(float), axis=0, weights=blk.n_PSMs.to_numpy(float))))))
    return pd.DataFrame(out)


# ================================================================ derived tables
def program_scores():
    memb = pd.read_csv(paths.PROGRAM_MEMBERS)
    programs = [("Neutrophil", "Neutrophil_markers"), ("Plasma / B cell", "B_plasma"), ("T / NK cell", "T_NK"), ("Myeloid", "Macrophage_myeloid"),
                ("Interferon", "Interferon_response"), ("Tubular injury", ["Havcr1", "Lcn2", "Spp1", "Timp1", "Clu"]), ("Early complement", "Complement_early"),
                ("Terminal complement", ["C5", "C6", "C8a", "C8b", "C8g", "C9"]), ("GBM-associated collagen IV", ["Col4a3", "Col4a4", "Col4a5"]),
                ("Fibrillar collagen I / III", "Collagen_I_III"), ("Extracellular matrix", "ECM"), ("Oxidative phosphorylation", "OXPHOS_structural"),
                ("TCA cycle", ["Cs", "Aco2", "Idh2", "Idh3a", "Idh3b", "Idh3g", "Ogdh", "Dlst", "Sucla2", "Suclg1", "Fh", "Mdh2"]),
                ("Fatty-acid oxidation", "FAO_beta_oxidation"), ("SLC / SLCO transport", "Transport_SLC_SLCO"), ("Proximal tubule", "Proximal_tubule_markers"),
                ("Male-specific PT program", "Androgen_supported_male_Xiong")]
    flagged = {"sirt1", "psmb10", "nmrk1", "cav1", "sirt2", "hk3", "nadsyn1"}
    rows, labels, members = [], [], {}
    for label, m in programs:
        genes = list(memb.loc[memb.program.eq(m), "gene"]) if isinstance(m, str) else m
        old = list(dict.fromkeys(OLDREP[g] for g in genes if g in OLDREP))
        ids = list(dict.fromkeys(i for i in (cur(o, OLDANN.gene[o], f"[program {label}]") for o in old) if i is not None))
        for g in genes:
            if g.casefold() in flagged and g in OLDREP:
                log(f"flagged protein in program {label}: {g}")
        z = Y.loc[ids].mean(axis=0)
        z = z - z[AGE_OF == 9].mean()
        rows.append(z)
        labels.append(f"{label} ({len(ids)})")
        members[label] = ids
    return pd.DataFrame(rows, index=labels), members


def cyp2c11_unique():
    """Summed linear signal of CYP2C11 gene-unique peptides, log2, relative to the 9-month mean."""
    lp = pd.read_csv(paths.PEPTIDE_LOG2, usecols=["peptide_form", "source_gene", "reference_gene_symbol"])
    cand = {backbone(f) for f, a, b in zip(lp.peptide_form, lp.source_gene, lp.reference_gene_symbol) if "cyp2c11" in tokens(a) | tokens(b)}
    m = map_seqs(sorted(cand))
    uniq = {s for s, v in m.items() if v["genes"] == {"cyp2c11"}}
    lin = pd.read_csv(paths.PEPTIDE_LINEAR)
    bio = meta.loc[LABELS]
    cols = {}
    for lab, r in bio.iterrows():
        t = str(r.tag)
        cols[short(lab)] = t if r.series == "H" else t.upper()
    assert all(c in lin.columns for c in cols.values()), cols
    lin["sequence"] = lin.peptide_form.map(backbone)
    sel = lin[lin.sequence.isin(uniq)]
    tot = pd.Series({k: sel[c].sum() for k, c in cols.items()})
    v = np.log2(tot[RATS])
    return v - v[AGE_OF == 9].mean(), len(uniq), int(sel.shape[0])


def rna_table():
    rna = pd.read_csv(paths.PROTEIN_RNA_EFFECTS)
    rec = [cur(p, g, "[RNA comparison]") for p, g in zip(rna.protein_id, rna.gene)]
    rna["rid"] = rec
    n0 = len(rna)
    rna = rna[rna.rid.notna()].copy()
    dup = rna.rid.duplicated(keep=False).sum()
    if n0 - len(rna) or dup:
        log(f"RNA comparison: {n0} genes -> {len(rna)} matched; {dup} rows share a group with another gene")
    rna["protein"] = AGE.loc[rna.rid, "effect"].values
    fit = sm.RLM(rna.protein, sm.add_constant(rna.RNA_27vs6), M=sm.robust.norms.HuberT()).fit()
    rna["resid"] = rna.protein - (fit.params.iloc[0] + fit.params.iloc[1] * rna.RNA_27vs6)
    return rna, float(fit.params.iloc[0]), float(fit.params.iloc[1])


def composite(ids):
    return Y.loc[[i for i in ids if i is not None]].mean(axis=0)


PLASMA = ["Alb", "Apoa1", "Apoa2", "Apoa4", "Apob", "F2", "Fga", "Fgb", "Fgg", "Gc", "Hpx", "Kng1", "Plg", "Tf", "Ttr", "Vtn"]
TAL = ["Slc12a1", "Kcnj1", "Cldn16", "Cldn10", "Casr", "Cldn19"]
TERMINAL = ["C5", "C6", "C8a", "C8b", "C8g", "C9"]


def complement():
    term = {g: gene_rec(g, "[terminal complement]") for g in TERMINAL}
    plasma = [gene_rec(g, "[plasma comparator]") for g in PLASMA]
    tvals = composite(term.values())
    out = {"none": fit1(tvals)[25], "broad_plasma_composite": fit1(tvals - composite(plasma))[25],
           "albumin": fit1(tvals - Y.loc[gene_rec("Alb")])[25], "C3": fit1(tvals - Y.loc[gene_rec("C3")])[25]}
    comps = {g: AGE.loc[i, ["effect", "lo", "hi"]].to_dict() for g, i in term.items() if i is not None}
    return out, comps, sum(p is not None for p in plasma)


def umod():
    rid = gene_rec("Umod", "[UMOD]")
    p = load_peptides(protein_ids=[rid])
    prof = seq_profiles(p)
    acc = rid.split("|")[1]
    m = map_seqs(prof.sequence, acc)
    prof["start"] = [m[s]["start"] for s in prof.sequence]
    prof["end"] = [m[s]["end"] for s in prof.sequence]
    prof["gene_unique"] = [m[s]["genes"] == {"umod"} for s in prof.sequence]
    prof = prof[prof.start.notna()].copy()
    prof["region"] = np.where(prof.end <= 591, "secreted_ectodomain", np.where(prof.start >= 592, "post_hepsin_tail", "boundary_spanning"))
    gu = prof[prof.gene_unique]
    ecto = gu.loc[gu.region.eq("secreted_ectodomain"), RATS].mean()
    tail = gu.loc[gu.region.eq("post_hepsin_tail"), RATS].mean()
    tal = composite([gene_rec(g, "[TAL markers]") for g in TAL])
    outcomes = {"UMOD_secreted_ectodomain": ecto, "UMOD_post_hepsin_tail": tail, "TAL_composite": tal,
                "UMOD_ectodomain_minus_tail": ecto - tail, "UMOD_ectodomain_minus_TAL": ecto - tal}
    res = {}
    for k, v in outcomes.items():
        f = fit1(v.values)
        q = bh([f[a]["p"] for a in (18, 21, 25)])
        res[k] = dict(**f[25], q=q[2])
    forms = p.assign(region=p.sequence.map(dict(zip(prof.sequence, prof.region))))
    nforms = forms.groupby("region").size().to_dict()
    return res, prof, pd.DataFrame({"ecto": ecto, "tail": tail}), nforms, int((~prof.gene_unique).sum())


MATRIX = ["Col4a1", "Col4a2", "Col4a3", "Col4a4", "Col4a5", "Lama5", "Lamb1", "Lamb2", "Lamc1", "Nid1", "Nid2", "Col1a1", "Col1a2", "Col3a1"]
FIBRILLAR = ["Col1a1", "Col1a2", "Col3a1", "Fn1", "Dcn", "Lum"]


def matrix():
    recs = {g: gene_rec(g, "[matrix]") for g in MATRIX + ["Fn1", "Dcn", "Lum"]}
    eff = pd.DataFrame({g: AGE.loc[r, ["effect", "lo", "hi", "q"]] for g, r in recs.items() if r is not None and g in MATRIX}).T
    comp = {"collagen_iv_345": fit1(composite([recs[g] for g in ("Col4a3", "Col4a4", "Col4a5")]).values)[25],
            "interstitial_fibrillar_ecm": fit1(composite([recs[g] for g in FIBRILLAR]).values)[25]}
    return eff, comp


# ================================================================ FIGURE 4
CLOCK_LABEL = {"Hs3st3a1": "HS3ST3A/B"}          # the record carries peptides shared by HS3ST3A1 and HS3ST3B1
CLOCK_MATRIX = {"Hs3st3a1", "Timp3", "Nid1", "Mmp19", "Agrn", "Lamc1"}
CLOCK_SHOW = ["Hs3st3a1", "Timp3", "Nid1", "Mmp19"]


def clock():
    """Age clock from the nested analysis of this same matrix (scripts/figure4/protein_clock.py). Each rat's age is
    predicted by a clock built without that rat: inside each training set, proteins are ranked by stability across
    age-stratified half-samples and the top five enter a least-squares fit."""
    man = json.loads(paths.CLOCK_MANIFEST.read_text(encoding="utf-8"))
    assert man["matrix_sha256"] == hashlib.sha256(OLD_MATRIX.read_bytes()).hexdigest(), "clock was fit to a different matrix"
    assert man["outer_cv"] == "leave-one-animal-out"
    p = pd.read_csv(paths.CLOCK_PREDICTIONS, index_col=0)
    pred = pd.Series(p["nested_stability_top5"].values, index=[short(i) for i in p.index]).reindex(RATS)
    assert pred.notna().all()
    z = np.load(paths.CLOCK_ARRAYS)
    assert z["rank_orders"].shape == (31, len(OLD))
    use = pd.Series(np.bincount(z["rank_orders"][:, :5].ravel(), minlength=len(OLD)), index=OLD.protein_id)
    return pred, use[use > 0].sort_values(ascending=False, kind="stable")


def figure4(ps, cyp, rna, hub):
    fig = plt.figure(figsize=(7.2, 11.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[.95, 2.6, 1.75, 1.85], hspace=.52)

    # a: coverage (this study in orange, mouse studies in gray, light gray = in at least one animal)
    STUDY, EVERY, ANY = "#c8531f", "#6e6e6e", "#dcdcd8"
    top = gs[0].subgridspec(1, 3, width_ratios=[1.32, .3, 1], wspace=0)
    ax = fig.add_subplot(top[0])
    d = pd.read_csv(paths.PROTEOME_DEPTH)
    # show this study as proteins with signal in every animal
    d.loc[0, "records_in_at_least_one_animal"] = d.loc[0, "records_in_every_animal"]
    y = np.arange(len(d))
    ax.barh(y, d.records_in_at_least_one_animal, color=ANY, height=.62, label="in at least one animal")
    ax.barh(y, d.records_in_every_animal, color=[STUDY] + [EVERY] * (len(d) - 1), height=.62, label="in every animal")
    ax.set_yticks(y, [f"{'This study' if s.startswith('This') else s}\n({c})" for s, c in zip(d.study, d.cohort)], fontsize=6.3)
    ax.set_ylim(len(d) - .5, -.5)
    for j, r in d.iterrows():
        t = f"{r.records_in_every_animal:,}" + ("" if r.records_in_every_animal == r.records_in_at_least_one_animal else f" / {r.records_in_at_least_one_animal:,}")
        ax.text(r.records_in_at_least_one_animal + 250, j, t, va="center", fontsize=6.2)
    ax.set(xlim=(0, 17000), xticks=[0, 4000, 8000, 12000], xlabel="proteins quantified")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(axis="y", length=0)
    h_ = [plt.Rectangle((0, 0), 1, 1, color=EVERY), plt.Rectangle((0, 0), 1, 1, color=ANY)]
    ax.legend(h_, ["every animal", "at least one animal"], frameon=False, loc="lower right", fontsize=5.6, handlelength=.8,
              handletextpad=.4, bbox_to_anchor=(1.0, 0.0), borderaxespad=0, labelspacing=.2)
    panel(ax, "a", "Proteome coverage", dx=-.45)
    check(4, "a", "complete records, this study", int(d.records_in_every_animal[0]), "dataset row count", "count", 11485, 0)

    # a (right): transcription factors only
    ax = fig.add_subplot(top[2].subgridspec(1, 3, width_ratios=[.3, 1, .3])[0, 1])
    cov = pd.read_csv(paths.TF_COUNTS)
    rat = cov.iloc[0]
    rows = [rat, cov[cov.dataset.str.startswith("Keele 2025")].iloc[0], cov[cov.dataset.str.startswith("Wang 2025")].iloc[0], cov[cov.dataset.str.startswith("Takasugi")].iloc[0]]
    vals = [int(r["TF_genes"]) for r in rows]
    ax.bar(range(4), vals, color=[STUDY] + [EVERY] * 3, width=.72)
    for j, v in enumerate(vals):
        ax.text(j, v + max(vals) * .02, f"{v:,}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(range(4), ["This study", "Keele 2025", "Wang 2025", "Takasugi 2024"], rotation=45, ha="right", rotation_mode="anchor", fontsize=6.2)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("genes quantified\nin every animal")
    ax.set_title("Transcription factors", pad=6, fontsize=7.5, color=INK)
    check(4, "a", "rat TFs genes", int(rat["TF_genes"]), "transcription_factor_counts_by_study.csv", "count")

    # supplementary figure: volcano (one record per gene; q from BH across all records) and protein vs RNA
    figv = plt.figure(figsize=(7.0, 2.85))
    ax = figv.add_axes([.075, .17, .37, .72])
    rep = AGE[AGE.rep]
    fc, q = rep.effect.to_numpy(), rep.q.to_numpy()
    sig = (q < .01) & (np.abs(fc) > np.log2(1.5))
    nup, ndn = int((sig & (fc > 0)).sum()), int((sig & (fc < 0)).sum())
    check("S", "a", "q<0.01 gene representatives", int((q < .01).sum()), "age-only OLS HC3; BH across all records", "count", 956, 0)
    check("S", "a", "higher >1.5-fold", nup, "", "count", 401, 0)
    check("S", "a", "lower >1.5-fold", ndn, "", "count", 82, 0)
    check("S", "a", "gene representatives", len(rep), "one record per gene", "count", 11040, 0)
    col = np.where(sig, np.where(fc > 0, UP, DOWN), "#cfd2d6")
    order = np.argsort(sig)
    ax.scatter(fc[order], -np.log10(q[order]), s=2.5, c=col[order], lw=0, rasterized=True)
    ax.axhline(2, c=RULE, ls=":", lw=.6)
    ax.set(xlim=(-4, 4.6), xlabel="log2 change, 25 vs 9 months", ylabel="−log10 q")
    ax.text(.03, .97, f"{nup} higher\n{ndn} lower", transform=ax.transAxes, va="top", fontsize=6.5)
    ymax = -np.log10(q.min())
    for g, lab, xy in [("Havcr1", "KIM-1", (3.4, .46)), ("C9", "C9", (2.3, .63)), ("Slco1a1", "SLCO1A1", (-2.7, .55)), ("Slc22a6", "OAT1", (-1.0, .74))]:
        rid = rep_rec(g, "[volcano label]") if g != "C9" else gene_rec(g)
        if rid is None:
            continue
        r = AGE.loc[rid]
        ax.annotate(lab, (r.effect, -np.log10(r.q)), xytext=(xy[0], xy[1] * ymax), fontsize=6, ha="center", arrowprops=ARROW)
    panel(ax, "a", "Proteins changing with age", dx=-.17)
    ax = figv.add_axes([.6, .17, .36, .72])
    r_pr = stats.pearsonr(rna.RNA_27vs6, rna.protein).statistic
    check("S", "b", "protein-RNA r", r_pr, "age-only protein effects vs RNA table", "r", .6991, .0006)
    check("S", "b", "shared genes", len(rna), "", "count", 9580, 0)
    ax.scatter(rna.RNA_27vs6, rna.protein, s=2, c="#c9ccd0", lw=0, rasterized=True)
    xx = np.array([-3, 4])
    ax.plot(xx, hub[0] + hub[1] * xx, c=INK, lw=.8)
    for g, xy in {"C8b": (.35, 3.0), "Umod": (-1.3, 3.0), "Col4a3": (-2.2, 2.25), "Col4a4": (-2.2, .25), "Col4a5": (1.1, 2.2)}.items():
        r = rna[rna.gene.eq(g)]
        if not len(r):
            continue
        r = r.iloc[0]
        ax.scatter(r.RNA_27vs6, r.protein, s=13, c=UP, edgecolor="white", lw=.3, zorder=3)
        ax.annotate(g.upper(), (r.RNA_27vs6, r.protein), xytext=xy, fontsize=6, ha="center", arrowprops=ARROW)
    ax.set(xlim=(-3, 4), ylim=(-3, 4), xlabel="RNA change, 27 vs 6 months (log2)\n54 other rats", ylabel="Protein change (log2)\n25 vs 9 months")
    ax.text(.97, .04, f"{len(rna):,} genes\nr = {r_pr:.2f}", transform=ax.transAxes, ha="right", fontsize=6.3)
    panel(ax, "b", "Protein change follows RNA change, with exceptions", dx=-.25)

    # b: heat map, rats ordered within age by KIM-1 (tubular injury)
    kim1 = Y.loc[rep_rec("Havcr1", "[heat-map order]")]
    colorder = sorted(ps.columns, key=lambda r: (AGE_OF[r], kim1[r]))
    ps = ps.copy()
    ps.loc["CYP2C11"] = cyp.reindex(ps.columns)
    mrows = [("NEU5AC", "Neu5Ac"), ("ADPR", "ADP-ribose"), ("TCDCA", "Taurochenodeoxycholate"), ("NAD_SUM", NAD),
             ("NMN", "NMN"), ("PLP", "Pyridoxal 5′-phosphate"), ("PANTO", "Pantothenate"), ("CREATINE", "Creatine")]
    mm = MATCH.rename(columns={v: k for k, v in MID.items()})
    mm = mm - mm[AGE_OF.reindex(mm.index) == 9].mean()
    mblock = pd.DataFrame({lab: mm[k] for k, lab in mrows}).T.reindex(columns=colorder)
    ps[colorder].to_csv(OUT / "figure4b_rows.csv")
    sub = gs[1].subgridspec(2, 2, height_ratios=[len(ps), len(mblock)], width_ratios=[1, .016], hspace=.07, wspace=.025)
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad("#eeeeee")
    axp = fig.add_subplot(sub[0, 0])
    im = axp.imshow(ps[colorder].to_numpy(float), aspect="auto", cmap=cmap, vmin=-1.2, vmax=1.2, interpolation="nearest")
    axp.set_yticks(range(len(ps)), ps.index, fontsize=6.3)
    axm = fig.add_subplot(sub[1, 0], sharex=axp)
    axm.imshow(mblock.to_numpy(float), aspect="auto", cmap=cmap, vmin=-1.2, vmax=1.2, interpolation="nearest")
    axm.set_yticks(range(len(mblock)), mblock.index, fontsize=6.3)
    axm.text(colorder.index("P64"), len(mblock) / 2 - .5, "excluded", rotation=90, ha="center", va="center", fontsize=4.8, color=MUTED)
    for a_ in (axp, axm):
        a_.tick_params(length=0)
        a_.set_xticks([])
        for s in a_.spines.values():
            s.set_visible(False)
        start = 0
        for a in AGES[:-1]:
            start += sum(AGE_OF[c] == a for c in colorder)
            a_.axvline(start - .5, c="white", lw=1.6)
    axp.axhline(len(ps) - 2.5, c="white", lw=1.6)
    start = 0
    for a in AGES:
        n = sum(AGE_OF[c] == a for c in colorder)
        axm.text(start + (n - 1) / 2, len(mblock) + .05, f"{a} months", ha="center", va="top", fontsize=6.6, color="#7a68b0" if a == 9 else AGEC[a], fontweight="bold")
        start += n
    axm.set_xlabel("One column per rat; ordered within each age by KIM-1", labelpad=13, fontsize=6.5)
    for a_, lab in ((axp, "Proteins"), (axm, "Metabolites")):
        a_.text(-.3, .5, lab, rotation=90, transform=a_.transAxes, ha="center", va="center", fontsize=7, color=INK)
    cax = fig.add_subplot(sub[:, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log2 vs 9-month mean (clipped at ±1.2)", fontsize=6.3)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2)
    panel(axp, "b", "Proteins and metabolites move together in each kidney", dx=-.33)

    # c: age clock, each rat predicted by a clock built without it
    pred, use = clock()
    gname = OLD.set_index("protein_id").source_gene
    lab_ = lambda pid: CLOCK_LABEL.get(gname[pid], str(gname[pid]).upper())
    mid = gs[2].subgridspec(1, 5, width_ratios=[1.15, .7, 1.45, .45, 1.6], wspace=0)
    ax = fig.add_subplot(mid[0])
    rng_ = np.random.default_rng(5)
    for a in AGES:
        p = pred[AGE_OF == a].to_numpy()
        ax.scatter(a + rng_.uniform(-.55, .55, len(p)), p, s=13, c=AGEC[a], edgecolor="white", lw=.3, zorder=3)
    ax.plot([6, 28], [6, 28], c=RULE, lw=.7, ls="--", zorder=1)
    err = (pred - AGE_OF).abs()
    mae, r_clk = float(err.mean()), float(stats.pearsonr(AGE_OF, pred).statistic)
    guess = float(np.mean([abs(AGE_OF.iloc[i] - AGE_OF.drop(AGE_OF.index[i]).mean()) for i in range(31)]))
    check(4, "c", "clock error, mean absolute (months)", mae, "nested clock, top 5 re-selected per left-out rat", "months")
    check(4, "c", "clock r, predicted vs actual age", r_clk, "", "r")
    check(4, "c", "error when guessing the mean age (months)", guess, "", "months")
    ax.text(.04, .97, f"error {mae:.1f} months\nr = {r_clk:.2f}", transform=ax.transAxes, va="top", fontsize=6.3)
    ax.text(.97, .04, "each rat predicted by\na clock built without it", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=MUTED)
    ax.set(xlim=(6, 28), ylim=(5, 28), xticks=AGES, yticks=[10, 15, 20, 25], xlabel="Actual age (months)", ylabel="Predicted age (months)")
    panel(ax, "c", "Protein clock predicts age", dx=-.5)
    axc = ax

    # d: the proteins the 31 clocks use
    ax = fig.add_subplot(mid[2])
    shown = use[use >= 5]
    ismat = [gname[p] in CLOCK_MATRIX for p in shown.index]
    ax.bar(range(len(shown)), shown.values, width=.66, color=["#4d4d4d" if m else "#b8b8b8" for m in ismat])
    for j, v in enumerate(shown.values):
        ax.text(j, v + .7, str(int(v)), ha="center", va="bottom", fontsize=6.2)
    nm = sum(ismat)
    assert all(ismat[:nm]) and not any(ismat[nm:])
    ax.plot([-.33, nm - .67], [35, 35], c=INK, lw=.6)
    ax.text((nm - 1) / 2, 35.8, "matrix and heparan sulfate", ha="center", va="bottom", fontsize=6)
    ax.text(len(shown) - .6, 25, f"5 proteins per clock;\n{len(use)} different\nproteins in all", ha="right", va="bottom", fontsize=5.8, color=MUTED)
    ax.set_xticks(range(len(shown)), [lab_(p) for p in shown.index], fontsize=6.2, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set(xlim=(-.6, len(shown) - .4), ylim=(0, 41), yticks=[0, 10, 20, 31], ylabel="Clocks using the protein (of 31)")
    ax.spines["left"].set_bounds(0, 31)
    panel(ax, "d", "Clocks reuse the same proteins", dx=-.17)
    for p, v in shown.items():
        check(4, "d", f"clocks using {lab_(p)}", int(v), "top 5 of the fold-contained ranking", "count")
    check(4, "d", "distinct proteins used across 31 clocks", len(use), "", "count")
    use.rename(index=lambda p: f"{lab_(p)} | {p}").to_csv(OUT / "figure4d_clock_protein_use.csv", header=["clocks_using_protein_of_31"])
    pred.to_frame("predicted_age").assign(age=AGE_OF).to_csv(OUT / "figure4c_clock_predictions.csv")

    # e: trajectories of the clock proteins
    tg = mid[4].subgridspec(3, 2, height_ratios=[.1, 1, 1], hspace=.55, wspace=.62)
    pid_of = {gname[p]: p for p in use.index}
    for k, g in enumerate(CLOCK_SHOW):
        pid = pid_of[g]
        ax = fig.add_subplot(tg[1 + k // 2, k % 2])
        v = series(pid)
        for j, a in enumerate(AGES):
            vv = v[AGE_OF == a].to_numpy()
            ax.scatter(j + rng_.uniform(-.17, .17, len(vv)), vv, s=6, c=AGEC[a], lw=0, zorder=3)
            ax.plot([j - .27, j + .27], [vv.mean()] * 2, c=INK, lw=.9, zorder=4)
        eff = float(AGE.loc[pid, "effect"])
        ax.set_title(f"{lab_(pid)}  {2 ** eff:.1f}×", loc="left", pad=2, fontsize=6.5, color=INK)
        ax.set_xticks(range(4), AGES if k // 2 else [], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        if k % 2 == 0:
            ax.set_ylabel("log2 vs 9 months", fontsize=6.3)
        if k // 2:
            ax.set_xlabel("Age (months)", fontsize=6.3)
        if k == 0:      # panel title on the same line as c and d
            for x_, s_, kw in ((-.75, "e", dict(fontsize=9.5, fontweight="bold")), (0, "Clock proteins rise steadily", dict(fontsize=7.5, color=INK))):
                ax.annotate(s_, xy=(x_, 1), xycoords=(ax.transAxes, axc.transAxes), xytext=(0, 6 if x_ == 0 else 0),
                            textcoords="offset points", va="bottom", annotation_clip=False, **kw)
        older = v[AGE_OF > 9]
        check(4, "e", f"{lab_(pid)} older rats above the 9-month maximum", int((older > v[AGE_OF == 9].max()).sum()), "of 24", "count")
        check(4, "e", f"{lab_(pid)} 25 vs 9 months (log2)", eff, "age-only OLS", "effect")

    # supplementary c: metabolite forest
    save(figv, "supp_s5_protein_changes")
    figm = plt.figure(figsize=(5.4, 3.3))           # metabolite changes: their own supplementary figure
    ax = figm.add_axes([.43, .14, .55, .78])
    groups = [(f"{NAD} pathway", [("NMN", "NMN"), ("NAD_SUM", NAD), ("ADPR", "ADP-ribose")]),
              ("Cofactors and energy buffer", [("PLP", "Pyridoxal 5′-phosphate"), ("PANTO", "Pantothenate"), ("CREATINE", "Creatine")]),
              ("Amino acids", [("ALA", "Alanine"), ("ASN", "Asparagine"), ("GGLU", "γ-Glutamyl-glutamate"), ("ACLYS", "N6-Acetyl-lysine")]),
              ("Glycan and bile acids", [("NEU5AC", "Neu5Ac"), ("TCDCA", "Taurochenodeoxycholate"), ("GCA", "Glycocholate")])]
    yy, ticks, labels, heads = 0, [], [], []
    XF, XQ = 3.2, 3.95
    ax.text(XF, -1.05, "25 / 9", fontsize=6, color=MUTED, ha="left", va="center")
    ax.text(XQ, -1.05, "q", fontsize=6, color=MUTED, ha="left", va="center")
    for gi, (gname_, items) in enumerate(groups):
        yy += .35 if gi else 0
        ticks.append(yy); labels.append(gname_); heads.append(len(labels) - 1)
        yy += 1.05
        for key, lab in items:
            if key == "NAD_SUM":
                e, lo, hi, p = hc3_25v9(MET32["NAD_SUM"].drop("P64").to_numpy(), AGE32.drop("P64").to_numpy())
                qtxt = f"p = {p:.3f}"
            else:
                r = MST_STATS.loc[MID[key]]
                e, lo, hi = r.log2FC_25_vs_9, r.CI95_HC3_low_log2FC, r.CI95_HC3_high_log2FC
                qtxt = f"{r.q_HC3_25_vs_9:.3f}"
            c = UP if e > 0 else DOWN
            ax.plot([lo, hi], [yy, yy], c=c, lw=1)
            ax.scatter(e, yy, s=14, c=c, zorder=3, edgecolor="white", lw=.3)
            ax.text(XF, yy, f"{2 ** e:.2f}×", fontsize=6.2, va="center", ha="left")
            ax.text(XQ, yy, qtxt, fontsize=6, va="center", ha="left", color=MUTED)
            ticks.append(yy); labels.append("   " + lab)
            check("S", "c", f"{lab} 25v9 log2", float(e), "metabolite age statistics", "effect")
            yy += 1
    ax.axvline(0, c=RULE, lw=.6)
    ax.set(ylim=(yy - .4, -1.6), xlim=(-2.5, 4.6), xticks=[-2, -1, 0, 1, 2])
    ax.spines["bottom"].set_bounds(-2.5, 2.8)
    ax.spines["left"].set_visible(False)
    ax.set_yticks(ticks, labels, fontsize=6.3)
    ax.tick_params(axis="y", length=0, pad=104)
    for i, t in enumerate(ax.get_yticklabels()):
        t.set_ha("left")
        if i in heads:
            t.set_fontweight("bold")
    ax.set_xlabel("log2 change, 25 vs 9 months (95% CI)", x=.42)
    ax.set_title("Selective metabolite changes", loc="left", pad=6, fontsize=7.5, color=INK)
    save(figm, "supp_s4a_selected_metabolites")

    # f: which age changes do all old kidneys share? (old kidneys split within age at the KIM-1 median)
    sv = pd.read_csv(paths.FIG4F_SHARES)
    bot = gs[3].subgridspec(1, 5, width_ratios=[.3, 1, .8, 1, .3], wspace=0)
    for j, grp in enumerate(("Proteins", "Metabolites")):
        ax = fig.add_subplot(bot[1 + 2 * j])
        t = sv[sv.group == grp].reset_index(drop=True)
        yy = np.arange(len(t))
        for i, r in t.iterrows():
            c = UP if r.rises_with_age else DOWN
            ax.plot([max(r.lo, -.5), min(r.hi, 1.6)], [i, i], c="#4d4d4d", lw=.9)
            ax.scatter(np.clip(r.share, -.5, 1.6), i, s=16, c=c, edgecolor="white", lw=.3, zorder=3)
            ax.text(1.03, i, f"{2 ** r.old_vs_9_log2:.2f}×", transform=ax.get_yaxis_transform(), fontsize=6, va="center")
            check(4, "f", f"{r.label}: share in lower-KIM-1 old kidneys", float(r.share), "figure4f_lower_kim1_shares.py", "share")
        ax.text(1.03, -1.1, "old / young", transform=ax.get_yaxis_transform(), fontsize=5.6, color=MUTED, va="center")
        ax.axvline(1, c=INK, lw=.6, ls=":")
        ax.axvline(0, c=PALE, lw=.6)
        ax.set(xlim=(-.55, 1.65), ylim=(len(t) - .4, -1.6), xticks=[0, .5, 1])
        ax.set_yticks(yy, t.label, fontsize=6.2)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.text(1.0, -1.1, "same as all\nold kidneys", fontsize=5.4, color=MUTED, ha="center", va="center")
        ax.text(0.0, -1.1, "no average\nchange", fontsize=5.4, color=MUTED, ha="center", va="center")
        ax.set_xlabel("Average age change in lower-KIM-1 old kidneys\nrelative to all old kidneys", fontsize=6.3)
        ax.set_title(grp, fontsize=6.8, color=INK, pad=2, loc="left")
        if j == 0:
            ax.text(-.62, 1.1, "f", transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")
            ax.text(-.08, 1.1, "Age changes in old kidneys with lower KIM-1", transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom")
    save(fig, "figure4")


# ================================================================ METABOLOME AXIS (tables for Figure 5G, H; overview plot not used)
def metabolome_axis_overview():
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.12, .92], hspace=.62)
    row0 = gs[0].subgridspec(1, 3, wspace=.55)
    row1 = gs[1].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=.95)
    row2 = gs[2].subgridspec(1, 4, wspace=.62)

    ax = fig.add_subplot(row0[0, 0])
    sdr = STATE[AGE_OF == 25].std(ddof=1) / STATE[AGE_OF == 9].std(ddof=1)
    check(5, "a", "STATE PC1 variance fraction", FRAC, "age-mean residual PCA", "frac", .3568, .0006)
    check(5, "a", "25/9 within-age SD ratio", sdr, "", "ratio", 3.129, .006)
    chem = stats.pearsonr(STATE, (SERIES == "D").astype(float))
    check(5, "a", "STATE PC1 vs chemistry r^2 (check only)", chem.statistic ** 2, f"p = {chem.pvalue:.3f}", "frac")
    rng = np.random.default_rng(7)
    for j, a in enumerate(AGES):
        vals = STATE[AGE_OF == a].values
        ax.scatter(j + rng.uniform(-.17, .17, len(vals)), vals, s=11, c=AGEC[a], edgecolor="white", lw=.3, zorder=3)
        ax.plot([j - .24, j + .24], [vals.mean()] * 2, c=INK, lw=.9, zorder=4)
    ax.set_xticks(range(4), AGES)
    ax.set_xlabel("Age (months)")
    ax.axhline(0, c=RULE, lw=.6)
    ax.set_ylabel("Same-age remodeling axis (SD)")
    ax.text(.03, .03, f"25 / 9-month SD ratio = {sdr:.2f}", transform=ax.transAxes, va="bottom", fontsize=6.1)
    panel(ax, "a", "Older kidneys diverge", dx=-.3)

    ax = fig.add_subplot(row0[0, 1])
    cnt, _, _ = ax.hist(AGE.full_r, bins=np.linspace(-1, 1, 41), color="#d3d6da", edgecolor="white", lw=.3)
    top = cnt.max()
    for g, lab, yf in [("Slc22a6", "OAT1", 1.45), ("Slc34a1", "SLC34A1", 1.3), ("Slc22a8", "OAT3", 1.15),
                       ("Havcr1", "KIM-1", 1.15), ("Cd38", "CD38", 1.3), ("Ptprc", "CD45", 1.45)]:
        rid = rep_rec(g, "[STATE marks]")
        if rid is None:
            continue
        x = float(AGE.loc[rid, "full_r"])
        c = DOWN if x < 0 else UP
        ax.plot([x, x], [0, yf * top], c=c, lw=.7)
        ax.text(x + (.03 if x < 0 else -.03), yf * top, lab, ha="left" if x < 0 else "right", va="center", fontsize=6.1, color=c)
        check(5, "b", f"{lab} full-component r", x, "", "r")
    ax.set(xlim=(-1.05, 1.05), ylim=(0, 1.55 * top), xlabel="Association with same-age axis (r)", ylabel="Proteins")
    ax.text(.5, .45, f"{len(AGE):,} proteins", transform=ax.transAxes, ha="center", fontsize=6, color=MUTED)
    panel(ax, "b", "Injury and differentiated poles", dx=-.33)

    ax = fig.add_subplot(row0[0, 2])
    x, y = AGE.effect, AGE.full_r
    pr, sr = stats.pearsonr(x, y).statistic, stats.spearmanr(x, y).statistic
    sig = AGE.q < .01
    same = int((np.sign(x[sig]) == np.sign(y[sig])).sum())
    check(5, "c", "AGE-STATE Pearson", pr, "", "r", .5812, .0006)
    check(5, "c", "AGE-STATE Spearman", sr, "", "r", .7556, .0006)
    check(5, "c", "same-sign age-changing records", same, "", "count", 957, 0)
    check(5, "c", "age-changing records q<0.01", int(sig.sum()), "", "count", 994, 0)
    ax.scatter(x, y, s=1.6, c="#c9ccd0", lw=0, rasterized=True)
    for g, lab, xy in [("Havcr1", "KIM-1", (3.3, 1.12)), ("Cd38", "CD38", (1.7, 1.12)), ("Slc22a6", "OAT1", (-2.2, -.98)), ("Timp3", "TIMP3", (1.6, -.98))]:
        rid = rep_rec(g)
        if rid is None:
            continue
        r = AGE.loc[rid]
        c = UP if r.full_r > 0 else DOWN
        ax.scatter(r.effect, r.full_r, s=11, c=c, edgecolor="white", lw=.3, zorder=3)
        ax.annotate(lab, (r.effect, r.full_r), xytext=xy, fontsize=6, ha="center", va="center", arrowprops=ARROW)
    ax.axhline(0, c=RULE, lw=.5)
    ax.axvline(0, c=RULE, lw=.5)
    ax.set(xlim=(-3, 4.3), ylim=(-1.12, 1.25), yticks=[-1, -.5, 0, .5, 1], xlabel="Age effect, 25 vs 9 months (log2)", ylabel="Same-age association (r)")
    ax.text(.02, .98, f"r = {pr:.3f}\nρ = {sr:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=5.9)
    panel(ax, "c", "Age response and same-age axis align", dx=-.33)

    pc, mvar, load, pperm = metabolome_axis()
    st = STATE.reindex(pc.index)
    ag = AGE_OF.reindex(pc.index)
    r_ms = stats.pearsonr(pc, st).statistic
    per_age = {a: stats.spearmanr(pc[ag == a], st[ag == a]).statistic for a in AGES}
    check(5, "d", "metabolome PC1 vs proteome STATE r", r_ms, "n = 30", "r")
    check(5, "d", "within-age permutation p", pperm, "9,999 permutations", "p")
    check(5, "d", "metabolome residual PC1 variance fraction", mvar, "unchanged", "frac")
    for a in AGES:
        check(5, "d", f"Spearman within {a} months", per_age[a], "", "r")
    for s in ("H", "D"):
        ix = [i for i in pc.index if SERIES[i] == s]
        check(5, "d", f"metabolome axis vs STATE r, {s} animals only (check)", stats.pearsonr(pc[ix], st[ix]).statistic, f"n = {len(ix)}", "r")
    loo = [stats.pearsonr(pc.drop(i), st.drop(i)).statistic for i in pc.index]
    check(5, "d", "leave-one-rat-out r, minimum", min(loo), "", "r")
    pd.DataFrame({"age": ag, "series": SERIES.reindex(pc.index), "proteome_STATE": st, "metabolome_axis": pc}).to_csv(OUT / "figure5g_axes_by_rat.csv")
    ax = fig.add_subplot(row1[0, 0])
    for a in AGES:
        m = ag == a
        ax.scatter(st[m], pc[m], s=14, c=AGEC[a], edgecolor="white", lw=.3, zorder=3)
    ax.axhline(0, c=RULE, lw=.5)
    ax.axvline(0, c=RULE, lw=.5)
    ax.set(xlabel="Proteome same-age axis (SD)", ylabel="Metabolome same-age axis (SD)")
    ax.text(.03, .98, f"r = {r_ms:.2f}, n = 30\nwithin-age permutation\np = {pperm:.4f}", transform=ax.transAxes, va="top", fontsize=6.1)
    ax.text(.98, .03, "Within-age ρ\n" + "\n".join(f"{a} mo  {per_age[a]:+.2f}" for a in AGES), transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=MUTED)
    panel(ax, "d", "The metabolome finds the same axis", dx=-.3)

    ax = fig.add_subplot(row1[0, 1])
    tags = feat.collaborator_tag.reindex(load.index)
    tags["NAD_SUM"] = "A"
    names = feat.reported_name.reindex(load.index)
    names["NAD_SUM"] = "NAD+ (peaks summed)"
    tab = pd.DataFrame({"r": load, "tag": tags, "name": names}).sort_values("r")
    tab.to_csv(OUT / "figure5h_metabolite_axis_loadings.csv")
    a_ = tab[tab.tag.eq("A")]
    k = 8
    sel = pd.concat([a_.head(k), a_.tail(k)])
    clean = {"L(-)-Carnitine": "Carnitine", "L-Ergothioneine": "Ergothioneine", "DL-4-Hydroxyphenyllactic acid": "4-Hydroxyphenyllactate",
             "Cytidine 5'-monophosphate (hydrate)": "CMP", "Pyridoxal 5'-phosphate": "Pyridoxal 5′-phosphate", "N-Acetylneuraminic acid": "Neu5Ac",
             "3-Indoxyl sulphate": "Indoxyl sulfate", "N-Isovalerylglycine; peak2 (indistinguishable MS2)": "N-Isovalerylglycine",
             "Acetamidobenzoic acid": "Acetamidobenzoate", "Ferulic acid": "Ferulate", "Guanosine monophosphate": "GMP", "NAD+ (peaks summed)": NAD}
    labs = [clean.get(n, n.split(";")[0]) for n in sel.name]
    yy = np.r_[np.arange(k), np.arange(k) + k + .9]
    ax.barh(yy, sel.r, color=[DOWN if r < 0 else UP for r in sel.r], height=.72)
    ax.set_yticks(yy, labs, fontsize=6.2)
    ax.set_ylim(yy.max() + .6, -.6)
    ax.axvline(0, c=RULE, lw=.6)
    ax.set(xlim=(-1, 1), xticks=[-1, -.5, 0, .5, 1], xlabel="Correlation with metabolome same-age axis (r)")
    ax.tick_params(axis="y", length=0)
    ax.text(.08, k / 2 - .5, "Differentiated\npole", fontsize=6.4, color=DOWN, va="center", ha="left")
    ax.text(-.08, k + .9 + k / 2 - .5, "Injury\npole", fontsize=6.4, color=UP, va="center", ha="right")
    panel(ax, "e", "Metabolites that define each pole", dx=-.52)
    for _, r in sel.iterrows():
        check(5, "e", f"{r['name']} r with metabolome axis", float(r.r), "", "r")

    hinges = [("f", "Transporter loss,\nsolute retention", "Slc22a6", MATCH[MID["IS"]], "OAT1 protein (log2)", "Indoxyl sulfate (log2)"),
              ("g", "Metabolite partner\nof KIM-1", "Havcr1", MATCH[MID["NEU5AC"]], "KIM-1 protein (log2)", "Neu5Ac (log2)"),
              ("h", "CD38 tracks\nimmune cells", "Ptprc", "Cd38", "CD45 protein (log2)", "CD38 protein (log2)"),
              ("i", "NMN falls as\nCD38 rises", "Cd38", MATCH[MID["NMN"]], "CD38 protein (log2)", "NMN (log2)")]
    for j, (letter, title, xg, yv, xl, yl) in enumerate(hinges):
        ax = fig.add_subplot(row2[0, j])
        xs = series(rep_rec(xg, "[Fig 5 pair]"))
        if isinstance(yv, str):
            ys = series(rep_rec(yv, "[Fig 5 pair]"))
        else:
            ys = yv - yv[AGE_OF.reindex(yv.index) == 9].mean()
        idx = xs.index.intersection(ys.index)
        xs, ys = xs[idx], ys[idx]
        rho, ra, pa, chem = pair_stats(xs, ys)
        for a in AGES:
            m = AGE_OF[idx] == a
            ax.scatter(xs[m], ys[m], s=10, c=AGEC[a], edgecolor="white", lw=.3, zorder=3)
        ax.set(xlabel=xl, ylabel=yl)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        loc = (.03, .97, "top") if rho > 0 else (.03, .03, "bottom")
        ax.text(loc[0], loc[1], f"ρ = {rho:.2f}\nage-adj. {ra:+.2f}\nn = {len(idx)}", transform=ax.transAxes, va=loc[2], ha="left", fontsize=5.9)
        panel(ax, letter, title, dx=-.45, size=6.8)
        check(5, letter, f"{yl} vs {xl} Spearman", rho, "", "r")
        check(5, letter, f"{yl} vs {xl} age-adjusted rank r", ra, "", "r")
        check(5, letter, f"{yl} vs {xl} age-adjusted p", pa, "", "p")
        for s in ("H", "D"):
            check(5, letter, f"{yl} vs {xl} age-adjusted rank r, {s} animals only (check)", chem[s], "", "r")
    for gname, mid, letter, lab in (("Slc22a8", "IS", "f", "Indoxyl sulfate vs OAT3"), ("Cd38", "ADPR", "i", "ADP-ribose vs CD38")):
        xs = series(rep_rec(gname))[MATCH.index]
        rho, ra, pa, chem = pair_stats(xs, MATCH[MID[mid]])
        check(5, letter, f"{lab} Spearman", rho, "", "r")
        check(5, letter, f"{lab} age-adjusted rank r", ra, "", "r")
        for s in ("H", "D"):
            check(5, letter, f"{lab} age-adjusted rank r, {s} animals only (check)", chem[s], "", "r")
    h = [plt.Line2D([], [], marker="o", ls="", mfc=AGEC[a], mec="white", mew=.3, ms=4.5, label=f"{a} months") for a in AGES]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(.5, .045), ncol=4, frameon=False, handletextpad=.1, columnspacing=1.2, fontsize=6.5)
    save(fig, "metabolome_axis_overview")


# ================================================================ SUPPLEMENTARY FIGURE S7
def dircol(lo, hi):
    return UP if lo > 0 else DOWN if hi < 0 else NS


def forest_rows(ax, rows, xf=None, fold=True):
    ticks, labels, heads, yy = [], [], [], 0
    for r in rows:
        if len(r) == 1:
            ticks.append(yy); labels.append(r[0]); heads.append(len(labels) - 1)
        else:
            lab, e, lo, hi = r
            c = dircol(lo, hi)
            ax.plot([lo, hi], [yy, yy], c=c, lw=1)
            ax.scatter(e, yy, s=14, c=c, zorder=3, edgecolor="white", lw=.3)
            if xf is not None and fold:
                ax.text(xf, yy, f"{2 ** e:.2f}×", fontsize=6.2, va="center", ha="left")
            ticks.append(yy); labels.append(("   " if heads else "") + lab)
        yy += 1
    ax.set_yticks(ticks, labels, fontsize=6.3)
    ax.set_ylim(yy - .4, -.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, c=RULE, lw=.6)
    for i, t in enumerate(ax.get_yticklabels()):
        if i in heads:
            t.set_fontweight("bold")


def supp_s7(rna, comp_out, comps, nplasma, um_res, um_prof, um_animals, um_forms, um_shared, mx_eff, mx_comp):
    fig = plt.figure(figsize=(7.2, 8.1))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.35], hspace=.62)
    row0 = outer[0].subgridspec(1, 3, width_ratios=[1, .9, 1.05], wspace=.85)
    row1 = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=1.0)
    row2 = outer[2].subgridspec(1, 3, wspace=.62)
    cc = pd.read_csv(paths.COMPLEMENT_PROTEIN_RNA)
    cc = cc[cc.contrast.eq("25vs9") & cc.primary_gene_representative.eq(True)].set_index("gene")
    bm12 = pd.read_csv(paths.MATRIX_PROTEIN_RNA).set_index("gene")

    ax = fig.add_subplot(row0[0, 0])
    classes = [("plasma-derived / circulating", "Plasma-derived"), ("secreted", "Secreted"), ("extracellular matrix", "Matrix"),
               ("transmembrane / cell-surface", "Transmembrane"), ("intracellular / unclassified", "Intracellular")]
    tab = []
    for k, lab in classes:
        d = rna.loc[rna.localization.eq(k), "resid"]
        row = dict(cls=k, label=lab, n=len(d))
        for t in (.5, 1):
            row[f"up{t}"], row[f"dn{t}"] = 100 * (d >= t).mean(), 100 * (d <= -t).mean()
        tab.append(row)
    tab = pd.DataFrame(tab)
    tab.to_csv(OUT / "supp_s7a_rna_departure_by_class.csv", index=False)
    y = np.arange(len(tab))
    for t, hgt, alpha in ((.5, .72, .38), (1, .36, 1)):
        ax.barh(y, tab[f"up{t}"], height=hgt, color=UP, alpha=alpha, lw=0)
        ax.barh(y, -tab[f"dn{t}"], height=hgt, color=DOWN, alpha=alpha, lw=0)
    ax.plot([0, 0], [-.5, len(tab) - .5], c=INK, lw=.6)
    ax.set_yticks(y, [f"{r.label} ({r.n:,})" for r in tab.itertuples()], fontsize=6.2)
    ax.set_ylim(len(tab) - .45, -1.25)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-30, 78)
    ax.set_xticks([-20, 0, 20, 40], ["20", "0", "20", "40"])
    ax.spines["bottom"].set_bounds(-30, 45)
    ax.set_xlabel("% of class: protein below ← RNA prediction → above", x=.36)
    ax.text(62, -.95, "≥0.5   ≥1", fontsize=5.8, color=MUTED, ha="center", va="center")
    for j, r in tab.iterrows():
        tot5, tot1 = r["up0.5"] + r["dn0.5"], r["up1"] + r["dn1"]
        ax.text(62, j, f"{tot5:.0f}%   {tot1:.0f}%" if tot1 >= 1 else f"{tot5:.0f}%   {tot1:.1f}%", fontsize=6, ha="center", va="center")
        check(6, "a", f"{r.cls}: % departing >=0.5 (up/down)", tot5, f"n = {r.n}", "pct")
        check(6, "a", f"{r.cls}: % departing >=1 (up/down)", tot1, f"n = {r.n}", "pct")
    ax.spines["left"].set_visible(False)
    for x0, alpha, lab in ((-29, .38, "≥0.5 log2"), (14, 1, "≥1")):
        ax.add_patch(plt.Rectangle((x0, -1.12), 5, .34, color=MUTED, alpha=alpha, lw=0, clip_on=False))
        ax.text(x0 + 7, -.95, lab, fontsize=5.8, va="center", ha="left", color=MUTED)
    panel(ax, "a", "Departures from RNA by protein class", dx=-.62)

    ax = fig.add_subplot(row0[0, 1])
    for j, g in enumerate(TERMINAL):
        if g in comps:
            r = comps[g]
            ax.plot([r["lo"], r["hi"]], [j, j], c=UP, lw=1)
            ax.scatter(r["effect"], j, s=14, c=UP, zorder=3, edgecolor="white", lw=.3)
            check(6, "b", f"{g.upper()} protein 25v9 log2", r["effect"], "", "effect")
        rr = cc.loc[g]
        if rr.RNA_detection_class == "low_or_absent":
            ax.text(-.95, j, "RNA low/ND", fontsize=5.6, color=MUTED, va="center", ha="left")
        else:
            ax.scatter(rr.RNA_effect_27vs6, j, s=16, marker="s", facecolor="white", edgecolor=INK, lw=.7, zorder=3)
    ax.set_yticks(range(6), [g.upper() for g in TERMINAL], fontsize=6.3)
    ax.set_ylim(len(TERMINAL) + 1.5, -.6)
    ax.axvline(0, c=RULE, lw=.6)
    ax.tick_params(axis="y", length=0)
    ax.set(xlim=(-1, 3.4), xlabel="log2 change\n(protein 25 vs 9; RNA 27 vs 6 months)")
    h = [plt.Line2D([], [], marker="o", ls="", mfc=UP, mec="white", ms=4, label="Protein (95% CI)"),
         plt.Line2D([], [], marker="s", ls="", mfc="white", mec=INK, ms=4, label="RNA, other rats")]
    ax.legend(handles=h, loc="lower right", frameon=False, fontsize=5.8, handletextpad=.2, borderaxespad=.1, labelspacing=.25)
    check(6, "b", "terminal complement mean 25v9 log2", comp_out["none"]["effect"], "", "effect", 1.9952, .0006)
    panel(ax, "b", "Terminal complement", dx=-.26)

    ax = fig.add_subplot(row0[0, 2])
    zc = [("Plasma proteins (16)", "broad_plasma_composite"), ("Albumin", "albumin"), ("C3", "C3")]
    forest_rows(ax, [(lab, comp_out[k]["effect"], comp_out[k]["lo"], comp_out[k]["hi"]) for lab, k in zc], xf=2.3)
    ax.set(xlim=(0, 2.85), xticks=[0, .5, 1, 1.5, 2])
    ax.spines["bottom"].set_bounds(0, 2.15)
    ax.set_xlabel("Terminal-complement excess\nover comparator, 25 vs 9 months (log2)", x=.4)
    r = comp_out["broad_plasma_composite"]
    check(6, "c", "excess over plasma comparator log2", r["effect"], f"{nplasma} of 16 plasma markers found", "effect", 1.3705, .0006)
    check(6, "c", "excess CI low", r["lo"], "", "effect", .9113, .0006)
    check(6, "c", "excess CI high", r["hi"], "", "effect", 1.8297, .0006)
    check(6, "c", "excess over albumin log2", comp_out["albumin"]["effect"], "", "effect", .8739, .0006)
    check(6, "c", "excess over C3 log2", comp_out["C3"]["effect"], "", "effect", 1.1067, .0006)
    panel(ax, "c", "Excess over plasma proteins", dx=-.62)

    ax = fig.add_subplot(row1[0, 0])
    seqs = um_prof.sort_values("start").reset_index(drop=True)
    nseq = seqs[seqs.gene_unique].groupby("region").size()
    for j, r in seqs.iterrows():
        ax.plot([r.start, r.end], [j, j], lw=2.4, c=REGION[r.region] if r.gene_unique else "#dddddd", solid_capstyle="butt")
    ax.axvline(591, c=MUTED, ls="--", lw=.7)
    ax.text(585, 1, "R591", rotation=90, ha="right", va="bottom", fontsize=6, color=MUTED)
    ax.set(xlim=(0, 650), xticks=[0, 200, 400, 600], xlabel="Rat UMOD residue (precursor, 644 aa)", ylabel=f"{len(seqs)} sequences")
    ax.set_yticks([])
    h = [plt.Line2D([], [], c=REGION[k], lw=2.4, label=f"{lab} ({int(nseq.get(k, 0))})") for k, lab in
         [("secreted_ectodomain", "Upstream"), ("post_hepsin_tail", "C-terminal"), ("boundary_spanning", "Boundary")]]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=5.9, handlelength=1.2, borderaxespad=.1)
    for k, lab in (("secreted_ectodomain", "upstream"), ("post_hepsin_tail", "C-terminal")):
        check(6, "d", f"{lab} sequences / forms", int(nseq.get(k, 0)), f"{um_forms.get(k, 0)} forms; {um_shared} shared sequences not gene-unique", "count",
              {"upstream": 18, "C-terminal": 3}[lab], 0)
    panel(ax, "d", "UMOD peptide map", dx=-.12)

    ax = fig.add_subplot(row1[0, 1])
    rng = np.random.default_rng(11)
    for col, k, lab, off in [("ecto", "secreted_ectodomain", "Upstream", -.13), ("tail", "post_hepsin_tail", "C-terminal", .13)]:
        c = REGION[k]
        for j, a in enumerate(AGES):
            v = um_animals.loc[AGE_OF[AGE_OF == a].index, col]
            ax.scatter(j + off + rng.uniform(-.05, .05, len(v)), v, s=8, c=c, alpha=.55, lw=0)
        m = um_animals.groupby(AGE_OF)[col].mean().reindex(AGES)
        ax.plot(np.arange(4) + off, m, "o-", c=c, ms=3.5, lw=1.1, label=lab, zorder=4)
    ax.axhline(0, c=RULE, lw=.6)
    ax.set_xticks(range(4), AGES)
    ax.set(xlabel="Age (months)", ylabel="Region, log2 vs 9 months")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.4)
    panel(ax, "e", "Upstream region accumulates", dx=-.3)

    ax = fig.add_subplot(row1[0, 2])
    u = um_res
    rows = [("Regions",), ("Upstream", *[u["UMOD_secreted_ectodomain"][k] for k in ("effect", "lo", "hi")]),
            ("C-terminal", *[u["UMOD_post_hepsin_tail"][k] for k in ("effect", "lo", "hi")]),
            ("TAL markers (6)", *[u["TAL_composite"][k] for k in ("effect", "lo", "hi")]),
            ("Contrasts",), ("Upstream − C-terminal", *[u["UMOD_ectodomain_minus_tail"][k] for k in ("effect", "lo", "hi")]),
            ("Upstream − TAL", *[u["UMOD_ectodomain_minus_TAL"][k] for k in ("effect", "lo", "hi")])]
    forest_rows(ax, rows, fold=False)
    ax.set(xlim=(-.6, 2.6), xticks=[0, 1, 2], xlabel="log2 change, 25 vs 9 months")
    ax.tick_params(axis="y", pad=62)
    for t in ax.get_yticklabels():
        t.set_ha("left")
    mc = u["UMOD_ectodomain_minus_tail"]
    ax.text(mc["hi"] + .06, 5, f"q = {mc['q']:.3f}", fontsize=5.9, va="center", color=MUTED)
    check(6, "f", "upstream region log2", u["UMOD_secreted_ectodomain"]["effect"], "", "effect", 1.4276, .0006)
    check(6, "f", "C-terminal region log2", u["UMOD_post_hepsin_tail"]["effect"], "", "effect", -.1062, .0006)
    check(6, "f", "upstream - C-terminal log2", mc["effect"], "", "effect", 1.5338, .0006)
    check(6, "f", "upstream - C-terminal CI low", mc["lo"], "", "effect", .6436, .0006)
    check(6, "f", "upstream - C-terminal CI high", mc["hi"], "", "effect", 2.4240, .0006)
    check(6, "f", "upstream - C-terminal q", mc["q"], "BH within outcome over 3 contrasts", "q", .00448, .00002)
    check(6, "f", "upstream - TAL log2", u["UMOD_ectodomain_minus_TAL"]["effect"], "", "effect", 1.5358, .0006)
    check(6, "f", "TAL composite log2", u["TAL_composite"]["effect"], "", "effect", -.1082, .0006)
    panel(ax, "f", "Regions diverge within one precursor", dx=-.62)

    ax = fig.add_subplot(row2[0, 0:2])
    groups = [("Collagen IV", ["Col4a1", "Col4a2", "Col4a3", "Col4a4", "Col4a5"]), ("Laminins", ["Lama5", "Lamb1", "Lamb2", "Lamc1"]),
              ("Nidogens", ["Nid1", "Nid2"]), ("Fibrillar collagens (recovered)", ["Col1a1", "Col1a2", "Col3a1"])]
    rows = []
    for gname, gl in groups:
        rows.append((gname,))
        for g in gl:
            if g in mx_eff.index:
                r = mx_eff.loc[g]
                rows.append((g.upper(), r.effect, r.lo, r.hi))
    forest_rows(ax, rows, xf=2.3)
    ax.set(xlim=(-.6, 2.75), xticks=[-.5, 0, .5, 1, 1.5, 2])
    ax.spines["bottom"].set_bounds(-.6, 2.15)
    ax.tick_params(axis="y", pad=100)
    for t in ax.get_yticklabels():
        t.set_ha("left")
    ax.text(2.3, -1.05, "25 / 9", fontsize=6, color=MUTED, ha="left", va="center")
    ax.set_ylim(len(rows) - .4, -1.6)
    ax.set_xlabel("log2 change, 25 vs 9 months (95% CI)", x=.38)
    for g, exp in [("Col4a3", 2.5506), ("Col4a4", 1.9028), ("Col4a5", 2.6015), ("Lama5", 1.2832), ("Lamb2", 1.2839), ("Lamc1", 1.3170), ("Nid1", 1.4316)]:
        check(6, "g", f"{g.upper()} fold 25v9", 2 ** float(mx_eff.loc[g, "effect"]), "", "fold", exp, .006)
    check(6, "g", "COL4A3/4/5 composite fold", 2 ** mx_comp["collagen_iv_345"]["effect"], "", "fold")
    check(6, "g", "NID2 log2 (q)", float(mx_eff.loc["Nid2", "effect"]), f"q = {mx_eff.loc['Nid2', 'q']:.3f}", "effect")
    check(6, "g", "recovered fibrillar ECM composite log2", mx_comp["interstitial_fibrillar_ecm"]["effect"], f"p = {mx_comp['interstitial_fibrillar_ecm']['p']:.3f}", "effect")
    panel(ax, "g", "Basement-membrane components change selectively", dx=-.32)

    ax = fig.add_subplot(row2[0, 2])
    ax.axhline(0, c=RULE, lw=.5)
    ax.axvline(0, c=RULE, lw=.5)
    ax.plot([-1, 1.1], [-1, 1.1], ls=":", c="#9a9a9a", lw=.7)
    sets = [(["Col4a1", "Col4a2", "Col4a3", "Col4a4", "Col4a5"], dict(c=UP, edgecolor="white")),
            (["Lama5", "Lamb1", "Lamb2", "Lamc1", "Nid1", "Nid2"], dict(c="#4d4d4d", edgecolor="white")),
            (["Col1a1", "Col1a2", "Col3a1"], dict(facecolor="white", edgecolor="#4d4d4d"))]
    for gl, kw in sets:
        gl = [g for g in gl if g in mx_eff.index]
        ax.scatter(bm12.loc[gl, "rna_effect_27v6"], mx_eff.loc[gl, "effect"].astype(float), s=20, lw=.7, zorder=3, **kw)
    labels = {"Col4a3": (-.72, 1.84), "Col4a5": (.3, 1.84), "Col4a4": (-.72, .62), "Col4a2": (.68, 1.22), "Col4a1": (.74, .74),
              "Nid1": (-.55, .3), "Nid2": (-.45, -.2), "Col3a1": (.95, .42), "Col1a1": (.85, -.22)}
    for g, xy in labels.items():
        if g in mx_eff.index:
            ax.annotate(g.upper(), (bm12.loc[g, "rna_effect_27v6"], float(mx_eff.loc[g, "effect"])), xytext=xy, fontsize=5.9, ha="center", arrowprops=ARROW)
    lam = [g for g in ("Lama5", "Lamb1", "Lamb2", "Lamc1") if g in mx_eff.index]
    ax.annotate("Laminins", (bm12.loc[lam, "rna_effect_27v6"].mean(), mx_eff.loc[lam, "effect"].astype(float).mean()), xytext=(.42, .62), fontsize=5.9, ha="center", arrowprops=ARROW)
    ax.set(xlim=(-1, 1.1), ylim=(-.32, 1.98), xlabel="RNA log2 change, 27 vs 6 months\n(54 other rats)", ylabel="Protein log2 change, 25 vs 9 months")
    panel(ax, "h", "Collagen IV rises without RNA", dx=-.38)
    save(fig, "supp_s7_extracellular_proteins")


# ================================================================ run
if __name__ == "__main__":
    print(f"{len(Y)} records for log2 analyses")
    ps, members = program_scores()
    ref = pd.read_csv(paths.REF_PROGRAM_SCORES, index_col=0)
    ref.columns = [short(c) for c in ref.columns]
    assert np.allclose(ps.iloc[:16].values, ref.loc[:, ps.columns].values, atol=1e-10), "program scores differ from the reference table"
    print("program scores reproduce the reference program scores exactly")
    cyp, n_cyp_seq, n_cyp_forms = cyp2c11_unique()
    check(4, "c", "CYP2C11 gene-unique sequences", n_cyp_seq, f"{n_cyp_forms} peptide forms summed (linear)", "count")
    check(4, "c", "CYP2C11 gene-unique 25-month mean vs 9-month mean (display, unadjusted log2)", float(cyp[AGE_OF == 25].mean()), "", "effect")
    old = pd.read_csv(paths.REF_CYP2C11)
    old = pd.Series(old.CYP2C11_gene_unique_log2.values, index=old["sample"].map(short))
    old -= old[AGE_OF == 9].mean()
    print("CYP2C11 vs reference table: r =", round(stats.pearsonr(cyp[RATS], old[RATS]).statistic, 4), "max |diff| =", round(float((cyp[RATS] - old[RATS]).abs().max()), 4))
    check(4, "c", "male-specific PT program members mapped", len(members["Male-specific PT program"]), "", "count")
    rna, hint, hslope = rna_table()
    ref_rna = pd.read_csv(paths.PROTEIN_RNA_EFFECTS)
    f = sm.RLM(ref_rna.protein_25vs9, sm.add_constant(ref_rna.RNA_27vs6), M=sm.robust.norms.HuberT()).fit()
    assert abs(f.params.iloc[1] - .712534) < .002, f.params
    print("Huber implementation reproduces the reference fit:", f.params.round(4).tolist())
    comp_out, comps, nplasma = complement()
    um = umod()
    mx_eff, mx_comp = matrix()
    figure4(ps, cyp, rna, (hint, hslope))
    metabolome_axis_overview()
    supp_s7(rna, comp_out, comps, nplasma, *um, mx_eff, mx_comp)
    pd.DataFrame(CHECKS).to_csv(OUT / "number_checks.csv", index=False)
    AGE.to_csv(OUT / "protein_age_and_score_statistics.csv")
    (OUT / "mapping_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"{len(CHECKS)} numbers recorded" + ("; reference assertions passed" if CHECK_REFERENCE else ""))







