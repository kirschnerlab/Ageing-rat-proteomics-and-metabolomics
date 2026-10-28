"""Protein clock (Figure 4C-E): fully nested chronological-signature models for the 31-rat kidney proteome.

The biological unit is the animal. No technical correction. Scaling, penalty
selection, and feature ranking are fit without the outer held-out animal.
Writes to outputs/protein_clock/ (about 30 minutes with 4 processes). Figure 4 reads the
shipped copy of nested_predictions.csv, model_arrays.npz and run_manifest.json in
data/proteomics/protein_clock/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import paths

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta as beta_dist
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
ELASTIC_LAMBDAS = np.logspace(1.5, -2.5, 21)
RIDGE_LAMBDAS = np.logspace(-3, 5, 21)
AGES = [9, 18, 21, 25]
SEED = 20260917
N_STABILITY_RESAMPLES = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data():
    matrix_path = paths.PROTEIN_MATRIX
    meta_path = paths.SAMPLE_METADATA
    matrix = pd.read_csv(matrix_path)
    meta = pd.read_csv(meta_path)
    meta = meta.loc[meta["role"].eq("biological_sample")].copy()
    meta["age"] = meta["age_months"].astype(int)
    meta = meta.set_index("label")
    sample_cols = [c for c in matrix.columns if c in meta.index]
    if len(sample_cols) != 31:
        raise ValueError(f"Expected 31 animals, found {len(sample_cols)}")
    meta = meta.loc[sample_cols]
    ymat = matrix[sample_cols].to_numpy(float)
    if not np.isfinite(ymat).all():
        raise ValueError("Complete matrix contains non-finite values")
    info = matrix.drop(columns=sample_cols).copy()
    info["gene"] = info["source_gene"].fillna(info["protein_id"].str.split("|").str[-1])
    return info, ymat, meta, sample_cols, matrix_path, meta_path


def design(meta: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(meta)),
            labels == 18,
            labels == 21,
            labels == 25,
        ]
    ).astype(float)


def technical_adjust(ymat, meta, labels, train_idx):
    """Identity. No sample correction is learned or applied."""
    return np.asarray(ymat, dtype=float)


def split_scaled(
    ymat: np.ndarray,
    meta: pd.DataFrame,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
):
    adj = technical_adjust(ymat, meta, labels, fit_idx)
    scaler = StandardScaler().fit(adj[:, fit_idx].T)
    return scaler.transform(adj[:, fit_idx].T), scaler.transform(adj[:, eval_idx].T)


def prediction_path(xtr, ytr, xva, l1_ratio: float, lambdas: np.ndarray):
    if l1_ratio == 0:
        predictions = []
        coefficients = []
        for lam in lambdas:
            model = Ridge(alpha=float(lam)).fit(xtr, ytr)
            predictions.append(model.predict(xva))
            coefficients.append(model.coef_)
        return np.column_stack(predictions), np.column_stack(coefficients)

    from sklearn.linear_model import enet_path

    ymean = float(np.mean(ytr))
    _, coefs, _ = enet_path(
        np.asfortranarray(xtr),
        ytr - ymean,
        l1_ratio=l1_ratio,
        alphas=np.asarray(lambdas, float),
        max_iter=8000,
        tol=1e-4,
    )
    return xva @ coefs + ymean, coefs


def choose_lambda(
    ymat: np.ndarray,
    meta: pd.DataFrame,
    labels: np.ndarray,
    outer_train: np.ndarray,
    l1_ratio: float,
    seed: int,
):
    lambdas = RIDGE_LAMBDAS if l1_ratio == 0 else ELASTIC_LAMBDAS
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    errors = np.zeros((4, len(lambdas)))
    fold_models = []
    for fold, (itr, iva) in enumerate(splitter.split(np.zeros(len(outer_train)), labels[outer_train])):
        tr = outer_train[itr]
        va = outer_train[iva]
        xtr, xva = split_scaled(ymat, meta, labels, tr, va)
        pred, coef = prediction_path(xtr, labels[tr], xva, l1_ratio, lambdas)
        errors[fold] = np.mean((labels[va, None] - pred) ** 2, axis=0)
        fold_models.append((tr, va, coef))
    mean_err = errors.mean(axis=0)
    se_err = errors.std(axis=0, ddof=1) / math.sqrt(errors.shape[0])
    best = int(np.argmin(mean_err))
    eligible = np.where(mean_err <= mean_err[best] + se_err[best])[0]
    one_se = int(eligible[0])  # lambda grid is descending; first is strongest penalty
    return {
        "lambda_min": float(lambdas[best]),
        "lambda_1se": float(lambdas[one_se]),
        "best_index": best,
        "mean_cv_mse": float(mean_err[best]),
        "se_cv_mse": float(se_err[best]),
        "inner_models": fold_models,
    }


def refit_outer(
    ymat,
    meta,
    labels,
    train_idx,
    test_idx,
    l1_ratio,
    lam,
):
    xtr, xte = split_scaled(ymat, meta, labels, train_idx, test_idx)
    if l1_ratio == 0:
        model = Ridge(alpha=lam).fit(xtr, labels[train_idx])
    else:
        model = ElasticNet(
            alpha=lam,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=100000,
            tol=1e-5,
            selection="cyclic",
        ).fit(xtr, labels[train_idx])
    return float(model.predict(xte)[0]), np.asarray(model.coef_, float)


def fold_stability_ranking(
    ymat,
    meta,
    labels,
    train_idx,
    lam,
    seed,
):
    rng = np.random.default_rng(seed)
    m = ymat.shape[0]
    coefs = np.zeros((N_STABILITY_RESAMPLES, m), dtype=np.float32)
    train_labels = labels[train_idx]
    for b in range(N_STABILITY_RESAMPLES):
        chosen_local = []
        for age in AGES:
            local = np.where(train_labels == age)[0]
            take = max(2, int(math.ceil(len(local) / 2)))
            chosen_local.extend(rng.choice(local, take, replace=False).tolist())
        sub = train_idx[np.array(sorted(chosen_local))]
        xsub, _ = split_scaled(ymat, meta, labels, sub, sub[:1])
        model = ElasticNet(
            alpha=lam,
            l1_ratio=0.5,
            fit_intercept=True,
            max_iter=100000,
            tol=1e-5,
        ).fit(xsub, labels[sub])
        coefs[b] = model.coef_
    selected = np.abs(coefs) > 1e-10
    freq = selected.mean(axis=0)
    med_abs = np.median(np.abs(coefs), axis=0)
    mean_abs = np.mean(np.abs(coefs), axis=0)
    # Deterministic priority: selection frequency, then coefficient magnitude.
    order = np.lexsort((-mean_abs, -med_abs, -freq))
    return order, freq, coefs


def nested_minimal_predictions(
    ymat,
    meta,
    labels,
    train_idx,
    test_idx,
    order,
):
    xtr, xte = split_scaled(ymat, meta, labels, train_idx, test_idx)
    result = {}
    for k in range(1, 21):
        model = LinearRegression().fit(xtr[:, order[:k]], labels[train_idx])
        result[k] = float(model.predict(xte[:, order[:k]])[0])
    return result


def perf(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ae = np.abs(pred - y)
    slope, intercept = np.polyfit(pred, y, 1)
    out = {
        "mae": float(ae.mean()),
        "median_ae": float(np.median(ae)),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "pearson_r": float(stats.pearsonr(pred, y).statistic),
        "spearman_rho": float(stats.spearmanr(pred, y).statistic),
        "calibration_slope_true_on_pred": float(slope),
        "calibration_intercept_true_on_pred": float(intercept),
        "worst_abs_error": float(ae.max()),
    }
    for age in AGES:
        out[f"bias_{age}mo"] = float(np.mean(pred[y == age] - y[y == age]))
    return out


def process_outer_fold(j, ymat, meta, labels, samples):
    n, m = len(samples), ymat.shape[0]
    train = np.delete(np.arange(n), j)
    test = np.array([j])
    pred = {}
    pred_1se = {}
    coefs = {}
    lambdas = {}
    lambdas_1se = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for a in ALPHA_GRID:
            tuning = choose_lambda(
                ymat, meta, labels, train, a, SEED + 1000 * j + int(a * 100)
            )
            p, c = refit_outer(
                ymat, meta, labels, train, test, a, tuning["lambda_min"]
            )
            p1, _ = refit_outer(
                ymat, meta, labels, train, test, a, tuning["lambda_1se"]
            )
            pred[a] = p
            pred_1se[a] = p1
            coefs[a] = c.astype(np.float32)
            lambdas[a] = tuning["lambda_min"]
            lambdas_1se[a] = tuning["lambda_1se"]

        order, freq, stab_coef = fold_stability_ranking(
            ymat, meta, labels, train, lambdas[0.5], SEED + j
        )
        mini = nested_minimal_predictions(
            ymat, meta, labels, train, test, order
        )
    return {
        "j": j,
        "pred": pred,
        "pred_1se": pred_1se,
        "coefs": coefs,
        "lambdas": lambdas,
        "lambdas_1se": lambdas_1se,
        "order": order,
        "freq": freq.astype(np.float32),
        "stability_coefs": stab_coef,
        "mini": mini,
        "convergence_warnings": sum(
            issubclass(w.category, ConvergenceWarning) for w in caught
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=paths.OUTPUTS / "protein_clock")
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    info, ymat, meta, samples, matrix_path, meta_path = load_data()
    labels = meta["age"].to_numpy(float)
    n, m = len(samples), len(info)
    predictions = {a: np.zeros(n) for a in ALPHA_GRID}
    predictions_1se = {a: np.zeros(n) for a in ALPHA_GRID}
    outer_coefs = {a: np.zeros((n, m), dtype=np.float32) for a in ALPHA_GRID}
    chosen_lambda = {a: [] for a in ALPHA_GRID}
    chosen_lambda_1se = {a: [] for a in ALPHA_GRID}
    nested_preds = {k: np.zeros(n) for k in range(1, 21)}
    rank_orders = np.zeros((n, m), dtype=np.int32)
    rank_freqs = np.zeros((n, m), dtype=np.float32)
    stability_coefs = []

    fold_results = Parallel(n_jobs=4, backend="loky", verbose=10)(
        delayed(process_outer_fold)(j, ymat, meta, labels, samples) for j in range(n)
    )
    convergence_warning_count = 0
    for result in sorted(fold_results, key=lambda x: x["j"]):
        j = result["j"]
        for a in ALPHA_GRID:
            predictions[a][j] = result["pred"][a]
            predictions_1se[a][j] = result["pred_1se"][a]
            outer_coefs[a][j] = result["coefs"][a]
            chosen_lambda[a].append(result["lambdas"][a])
            chosen_lambda_1se[a].append(result["lambdas_1se"][a])
        rank_orders[j] = result["order"]
        rank_freqs[j] = result["freq"]
        stability_coefs.append(result["stability_coefs"])
        for k, value in result["mini"].items():
            nested_preds[k][j] = value
        convergence_warning_count += result["convergence_warnings"]
        print(
            f"outer {j + 1:02d}/{n} {samples[j]} age={int(labels[j])} "
            f"EN={predictions[0.5][j]:.2f} top3={nested_preds[3][j]:.2f} "
            f"warnings={result['convergence_warnings']}",
            flush=True,
        )

    # Prediction-level outputs.
    pred_table = pd.DataFrame(index=samples)
    pred_table.index.name = "animal"
    pred_table["age"] = labels.astype(int)
    for a in ALPHA_GRID:
        key = str(a).replace(".", "p")
        pred_table[f"pred_alpha_{key}_lambda_min"] = predictions[a]
        pred_table[f"pred_alpha_{key}_lambda_1se"] = predictions_1se[a]
    for k in nested_preds:
        pred_table[f"nested_stability_top{k}"] = nested_preds[k]
    pred_table.to_csv(out / "nested_predictions.csv")

    performance_rows = []
    for a in ALPHA_GRID:
        model_name = "ridge" if a == 0 else ("lasso" if a == 1 else f"elastic_net_alpha_{a}")
        for rule, values in [("lambda_min", predictions[a]), ("lambda_1se", predictions_1se[a])]:
            row = {"model": model_name, "selection_rule": rule, "cv": "LOO_outer_4fold_stratified_inner"}
            row.update(perf(labels, values))
            row["median_selected_features"] = (
                float(m)
                if a == 0
                else float(np.median((np.abs(outer_coefs[a]) > 1e-10).sum(axis=1)))
            )
            performance_rows.append(row)
    mean_pred = np.array([labels[np.arange(n) != j].mean() for j in range(n)])
    prior_pred = np.array([np.median(labels[np.arange(n) != j]) for j in range(n)])
    for name, values in [("training_mean_age", mean_pred), ("training_median_age_group_prior", prior_pred)]:
        row = {"model": name, "selection_rule": "none", "cv": "LOO"}
        row.update(perf(labels, values))
        row["median_selected_features"] = 0
        performance_rows.append(row)
    pd.DataFrame(performance_rows).to_csv(out / "cv_performance_summary.csv", index=False)

    hp_rows = []
    for a in ALPHA_GRID:
        p = perf(labels, predictions[a])
        hp_rows.append(
            {
                "alpha": a,
                "model": "ridge" if a == 0 else ("lasso" if a == 1 else "elastic_net"),
                "lambda_median": float(np.median(chosen_lambda[a])),
                "lambda_range_min": float(np.min(chosen_lambda[a])),
                "lambda_range_max": float(np.max(chosen_lambda[a])),
                "median_selected_features": float(m) if a == 0 else float(np.median((np.abs(outer_coefs[a]) > 1e-10).sum(1))),
                **p,
            }
        )
    pd.DataFrame(hp_rows).to_csv(out / "hyperparameter_sensitivity.csv", index=False)

    min_rows = []
    for k, values in nested_preds.items():
        min_rows.append({"k": k, "ranking": "fold_contained_half_sample_stability", **perf(labels, values)})
    pd.DataFrame(min_rows).to_csv(out / "minimal_signature_performance.csv", index=False)

    # Coefficient stability for primary alpha=0.5 outer fits.
    cmat = outer_coefs[0.5].astype(float)
    selected = np.abs(cmat) > 1e-10
    sign_nonzero = np.sign(cmat)
    selected_n = selected.sum(axis=0)
    positive_n = (sign_nonzero > 0).sum(axis=0)
    negative_n = (sign_nonzero < 0).sum(axis=0)
    sign_consistency = np.maximum(positive_n, negative_n) / np.maximum(selected_n, 1)
    top3 = np.zeros(m, int)
    top10 = np.zeros(m, int)
    rank_sum = np.zeros(m, float)
    for j in range(n):
        order = np.argsort(-np.abs(cmat[j]), kind="stable")
        rank = np.empty(m, int)
        rank[order] = np.arange(1, m + 1)
        rank_sum += rank
        top3[order[:3]] += 1
        top10[order[:10]] += 1
    ksel = selected_n.astype(float)
    ci_low = beta_dist.ppf(0.025, ksel + 0.5, n - ksel + 0.5)
    ci_high = beta_dist.ppf(0.975, ksel + 0.5, n - ksel + 0.5)
    stability = info[["protein_id", "gene", "n_PSMs", "n_peptide_forms"]].copy()
    stability["selection_frequency"] = selected.mean(axis=0)
    stability["selection_frequency_ci_low"] = ci_low
    stability["selection_frequency_ci_high"] = ci_high
    stability["median_standardized_coefficient"] = np.median(cmat, axis=0)
    stability["mean_standardized_coefficient"] = np.mean(cmat, axis=0)
    stability["coefficient_sd"] = np.std(cmat, axis=0, ddof=1)
    stability["coefficient_q025"] = np.quantile(cmat, 0.025, axis=0)
    stability["coefficient_q25"] = np.quantile(cmat, 0.25, axis=0)
    stability["coefficient_q75"] = np.quantile(cmat, 0.75, axis=0)
    stability["coefficient_q975"] = np.quantile(cmat, 0.975, axis=0)
    stability["sign_consistency_probability"] = sign_consistency
    stability["mean_abs_coefficient_rank"] = rank_sum / n
    stability["top3_frequency"] = top3 / n
    stability["top10_frequency"] = top10 / n
    stability = stability.sort_values(
        ["selection_frequency", "top10_frequency", "median_standardized_coefficient"],
        ascending=[False, False, False],
    )
    stability.to_csv(out / "coefficient_stability_summary.csv", index=False)

    # Fold-contained feature-ranking recurrence (separate from nonzero EN coefficients).
    top3_nested = np.zeros(m, int)
    top10_nested = np.zeros(m, int)
    mean_rank_nested = np.zeros(m, float)
    for j in range(n):
        order = rank_orders[j]
        rank = np.empty(m, int)
        rank[order] = np.arange(1, m + 1)
        mean_rank_nested += rank
        top3_nested[order[:3]] += 1
        top10_nested[order[:10]] += 1
    nested = info[["protein_id", "gene"]].copy()
    nested["mean_inner_half_sample_selection_frequency"] = rank_freqs.mean(axis=0)
    nested["top3_outer_folds"] = top3_nested
    nested["top10_outer_folds"] = top10_nested
    nested["mean_rank_across_outer_folds"] = mean_rank_nested / n
    nested.sort_values(
        ["top10_outer_folds", "top3_outer_folds", "mean_rank_across_outer_folds"],
        ascending=[False, False, True],
    ).to_csv(out / "nested_feature_selection.csv", index=False)

    np.savez_compressed(
        out / "model_arrays.npz",
        outer_coefficients_alpha_0p5=outer_coefs[0.5],
        rank_orders=rank_orders,
        rank_frequencies=rank_freqs,
        labels=labels,
        samples=np.asarray(samples),
    )

    manifest = {
        "analysis": "nested_protein_clock",
        "script": paths.rel(__file__),
        "seed": SEED,
        "n_animals": n,
        "age_counts": {str(a): int((labels == a).sum()) for a in AGES},
        "n_proteins": m,
        "matrix": paths.rel(matrix_path),
        "matrix_sha256": sha256(matrix_path),
        "metadata": paths.rel(meta_path),
        "metadata_sha256": sha256(meta_path),
        "alpha_grid": ALPHA_GRID,
        "elastic_lambda_grid": ELASTIC_LAMBDAS.tolist(),
        "ridge_lambda_grid": RIDGE_LAMBDAS.tolist(),
        "inner_cv": "4-fold age-stratified, shuffled; preprocessing refit in each inner training split",
        "outer_cv": "leave-one-animal-out",
        "stability_resamples_per_outer_fold": N_STABILITY_RESAMPLES,
        "convergence_warning_count": convergence_warning_count,
        "elapsed_seconds": time.time() - started,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
