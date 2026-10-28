"""Every input and output location used by the scripts, relative to this folder.

Scripts add this folder to sys.path and `import paths`. Inputs live under data/, results are written under outputs/.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def out(name):
    """Output folder for one script (created on first use)."""
    d = OUTPUTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(path):
    """Path relative to this folder, for logs and manifests."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


# ---------------------------------------------------------------- proteomics
PROT = DATA / "proteomics"
PROTEIN_MATRIX = PROT / "protein_abundance_log2.csv"                      # Supplementary Table 2
PEPTIDE_LOG2 = PROT / "peptide_abundance_log2.csv.gz"
PEPTIDE_LINEAR = PROT / "peptide_signal_linear_by_channel.csv.gz"
SAMPLE_METADATA = PROT / "sample_metadata.csv"
PROTEIN_ANNOTATION = PROT / "protein_annotation.csv.gz"
PROTEIN_AGE_STATISTICS = PROT / "protein_age_statistics.csv"
PROGRAM_MEMBERS = PROT / "program_members.csv"
PROGRAM_SCORES = PROT / "program_scores.csv"
BLOOD_SIGNAL_SCORES = PROT / "blood_signal_scores.csv"
REF_PROGRAM_SCORES = PROT / "reference_checks" / "program_scores_reference.csv"
REF_CYP2C11 = PROT / "reference_checks" / "cyp2c11_unique_peptides_reference.csv"
CLOCK_DIR = PROT / "protein_clock"                                         # precomputed; regenerate with protein_clock.py
CLOCK_PREDICTIONS = CLOCK_DIR / "nested_predictions.csv"
CLOCK_ARRAYS = CLOCK_DIR / "model_arrays.npz"
CLOCK_MANIFEST = CLOCK_DIR / "run_manifest.json"
PT_SCORES_BY_RAT = PROT / "proximal_tubule" / "program_and_pt_scores_by_rat.csv"
PT_SEGMENT_EFFECTS = PROT / "proximal_tubule" / "segment_marker_effects.csv"
PT_ADJUSTED_EFFECTS = PROT / "proximal_tubule" / "program_pt_adjusted_effects.csv"
PT_MATCHED_SETS = PROT / "proximal_tubule" / "matched_declining_sets.csv"
STATE_ARRAYS = PROT / "remodeling_state" / "state_arrays.npz"
STATE_RAT_SCORES = PROT / "remodeling_state" / "rat_scores.csv"

# ---------------------------------------------------------------- metabolomics
MET = DATA / "metabolomics"
METABOLITES_ALL = MET / "metabolite_log2_all_rats.csv"
METABOLITES_PROTEOMICS_RATS = MET / "metabolite_log2_proteomics_rats.csv"
METABOLITE_FEATURES = MET / "metabolite_features.csv"
METABOLOMICS_ANIMALS = MET / "animals.csv"
METABOLOMICS_SAMPLE_KEY = MET / "sample_key.csv"
METABOLITES_BY_SAMPLE = MET / "metabolite_log2_by_sample.csv"
METABOLITE_AGE_STATISTICS = MET / "metabolite_age_statistics.csv"
METABOLOMICS_EXPORT = MET / "metabolomics_compounds_export.xlsx"

# ---------------------------------------------------------------- RNA (Shavlakadze et al., 2026)
RNA = DATA / "rna_shavlakadze2026"
PROTEIN_RNA_EFFECTS = RNA / "protein_rna_gene_effects.csv"
RNA_SCORE_BY_RAT = RNA / "rna_remodeling_score_by_rat.csv"
RNA_SCORE_MARKERS = RNA / "rna_score_marker_correlations.csv"
RNA_SCORE_AGREEMENT = RNA / "rna_score_method_agreement.csv"
RNA_TRAJECTORY_EXAMPLES = RNA / "trajectory_examples.csv"
COMPLEMENT_PROTEIN_RNA = RNA / "complement_protein_rna_effects.csv"
MATRIX_PROTEIN_RNA = RNA / "matrix_protein_rna_effects.csv"

# ---------------------------------------------------------------- published proteomes and annotations
PUB = DATA / "published_proteomes"
PROTEOME_DEPTH = PUB / "proteome_depth_by_study.csv"
TF_COUNTS = PUB / "transcription_factor_counts_by_study.csv"
RAT_FASTA = DATA / "annotations" / "rat_uniprot_proteome_UP000002494.fasta"

# ---------------------------------------------------------------- NoRI
NORI_PT_MEASURES = DATA / "nori" / "nori_proximal_tubule_measures.csv"

# ---------------------------------------------------------------- intermediate tables (outputs of earlier steps, shipped)
INTER = DATA / "intermediate"
PROTEIN_AGE_SCORE_STATS = INTER / "protein_age_and_score_statistics.csv"   # figure4_supp_s4a_s5_s7.py
FIG4B_ROWS = INTER / "figure4b_rows.csv"                                    # figure4_supp_s4a_s5_s7.py
FIG4F_SHARES = INTER / "figure4f_lower_kim1_shares.csv"                     # figure4f_lower_kim1_shares.py
FIG5G_AXES = INTER / "figure5g_axes_by_rat.csv"                             # figure4_supp_s4a_s5_s7.py
FIG5H_LOADINGS = INTER / "figure5h_metabolite_axis_loadings.csv"            # figure4_supp_s4a_s5_s7.py

# ---------------------------------------------------------------- output folders
OUT_FIG4 = "figure4_supp_s4a_s5_s7"
