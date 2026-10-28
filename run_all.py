"""Run every figure script in order; results go to outputs/.

    python run_all.py                # all figures (about 2 minutes)
    python run_all.py --with-clock   # also refit the protein clock first (about 30 minutes)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "scripts/preprocessing/metabolomics_export_to_table.py",
    "scripts/preprocessing/metabolite_age_statistics.py",
    "scripts/figure4/figure4_supp_s4a_s5_s7.py",
    "scripts/figure4/figure4f_lower_kim1_shares.py",
    "scripts/figure5/figure5_remodeling_axis.py",
    "scripts/supplementary/s3_program_heatmaps.py",
    "scripts/supplementary/s4_metabolite_changes.py",
    "scripts/supplementary/s6_protein_rna_trajectories.py",
    "scripts/supplementary/s8_proximal_tubule_identity.py",
    "scripts/supplementary/s9_remodeling_score_method.py",
    "scripts/supplementary/s10_rna_score_robustness.py",
    "scripts/supplementary/s11_study_design_blood_signal.py",
]
if "--with-clock" in sys.argv:
    STEPS.insert(0, "scripts/figure4/protein_clock.py")

for step in STEPS:
    t0 = time.time()
    print(f"=== {step}", flush=True)
    r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / step)], cwd=ROOT)
    if r.returncode:
        sys.exit(f"stopped: {step} exited with {r.returncode}")
    print(f"    done in {time.time() - t0:.0f} s", flush=True)
print("all steps finished; results are in outputs/")
