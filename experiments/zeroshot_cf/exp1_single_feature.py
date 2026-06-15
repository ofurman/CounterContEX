"""Experiment 1: Single-feature reconstruction sanity check.

For each feature j in the dataset:
- Mask feature j in test points.
- Reconstruct via ConditionalDensitySampler (same-class context, near-MAP t=1e-9).
- Draw N_SAMPLES=50 posterior samples at t=1.0 for calibration.
- Compute MSE/MAE vs. TabPFN, marginal-mean baseline, and Ridge baseline.
- Report calibration: fraction of true values inside the 10–90% sampled interval.

Outputs:
  results/exp1_<dataset>.csv  — per-feature metrics table
  results/exp1_summary.md     — aggregate summary + gate verdict
  results/exp1_moons.png      — scatter plot (MOONS only)

Usage:
  uv run python experiments/zeroshot_cf/exp1_single_feature.py --dataset moons
  uv run python experiments/zeroshot_cf/exp1_single_feature.py --dataset heloc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT = 256    # cap context rows passed to TabPFNUnsupervisedModel

# Per-dataset runtime tuning: HELOC (23 features) needs fewer samples to stay
# under the 10-minute wall-clock budget (23 features × 2 classes × N_SAMPLES calls).
_DATASET_PARAMS = {
    "moons": {"n_samples": 50, "max_test": 50},
    "heloc": {"n_samples": 10, "max_test": 30},
}


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def run_experiment(dataset_name: str) -> None:
    print(f"\n=== Experiment 1: {dataset_name.upper()} ===")

    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import load_dataset
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    params = _DATASET_PARAMS.get(dataset_name, {"n_samples": 50, "max_test": 50})
    N_SAMPLES = params["n_samples"]
    MAX_TEST = params["max_test"]

    bundle = load_dataset(dataset_name)
    X_train = bundle.X_train
    y_train = bundle.y_train
    X_test = bundle.X_test[:MAX_TEST]
    y_test = bundle.y_test[:MAX_TEST]
    feat_names = bundle.feature_names
    n_features = X_train.shape[1]

    print(f"N_SAMPLES={N_SAMPLES}, MAX_TEST={MAX_TEST}, MAX_CONTEXT={MAX_CONTEXT}")

    print(f"Train: {X_train.shape}, Test (capped): {X_test.shape}, Features: {n_features}")

    print("Loading TabPFN models …")
    clf, reg = get_models(n_estimators=4)

    # Ridge baseline fit once on full X_train to predict each feature
    from sklearn.linear_model import RidgeCV
    ridge_models = {}
    for j in range(n_features):
        X_other = np.delete(X_train, j, axis=1)
        ridge_models[j] = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        ridge_models[j].fit(X_other, X_train[:, j])

    records = []

    for j in range(n_features):
        feat = feat_names[j]
        print(f"\n  Feature {j:2d}/{n_features-1}: {feat}")
        true_vals = X_test[:, j]

        # --- Marginal-mean baseline ---
        marginal_mean = float(np.mean(X_train[:, j]))
        pred_marginal = np.full(len(X_test), marginal_mean)
        mse_marginal = _mse(true_vals, pred_marginal)
        mae_marginal = _mae(true_vals, pred_marginal)

        # --- Ridge baseline ---
        X_other_test = np.delete(X_test, j, axis=1)
        pred_ridge = ridge_models[j].predict(X_other_test)
        mse_ridge = _mse(true_vals, pred_ridge)
        mae_ridge = _mae(true_vals, pred_ridge)

        # --- TabPFN: per-class conditional reconstruction ---
        classes = np.unique(y_test)
        tabpfn_preds = np.zeros(len(X_test))
        # posterior_samples[i] = array of N_SAMPLES values for test point i
        posterior_samples = [None] * len(X_test)

        for cls in classes:
            test_mask = y_test == cls
            X_test_cls = X_test[test_mask]
            if len(X_test_cls) == 0:
                continue

            sampler = ConditionalDensitySampler(
                clf=clf,
                reg=reg,
                append_target=False,
                n_permutations=5,
                temperature=1e-9,
                random_state=42 + j,
            )
            sampler.set_context(
                X_train,
                y_context=y_train,
                target_class=int(cls),
                max_context=MAX_CONTEXT,
            )

            # Near-MAP point estimate (t=1e-9 = sampler.temperature)
            point_est = sampler.sample_feature(X_test_cls, target_col=j, n_samples=1)
            tabpfn_preds[test_mask] = point_est

            # Posterior samples at t=1.0 for calibration interval (N_SAMPLES draws).
            # Use sample_temperature=1.0 so ALL draws explore the posterior;
            # the MAP draw above is kept separate and not included here.
            posterior = sampler.sample_feature(
                X_test_cls, target_col=j, n_samples=N_SAMPLES,
                sample_temperature=1.0,
            )  # (N_SAMPLES, m)
            idxs = np.where(test_mask)[0]
            for ii, gi in enumerate(idxs):
                posterior_samples[gi] = posterior[:, ii]

        mse_tabpfn = _mse(true_vals, tabpfn_preds)
        mae_tabpfn = _mae(true_vals, tabpfn_preds)

        # --- Calibration: fraction of true vals inside [10%, 90%] sampled interval ---
        calib_fracs = []
        for i, samps in enumerate(posterior_samples):
            if samps is None:
                continue
            lo, hi = np.percentile(samps, 10), np.percentile(samps, 90)
            calib_fracs.append(float(lo <= true_vals[i] <= hi))
        calibration = float(np.mean(calib_fracs)) if calib_fracs else float("nan")

        beats_marginal = mse_tabpfn < mse_marginal
        beats_ridge = mse_tabpfn < mse_ridge

        print(
            f"    MSE  marginal={mse_marginal:.4f}  ridge={mse_ridge:.4f}  tabpfn={mse_tabpfn:.4f}"
            f"  beats_marginal={beats_marginal}  calib10-90={calibration:.2f}"
        )

        records.append({
            "feature_idx": j,
            "feature_name": feat,
            "mse_marginal": mse_marginal,
            "mae_marginal": mae_marginal,
            "mse_ridge": mse_ridge,
            "mae_ridge": mae_ridge,
            "mse_tabpfn": mse_tabpfn,
            "mae_tabpfn": mae_tabpfn,
            "beats_marginal": beats_marginal,
            "beats_ridge": beats_ridge,
            "calibration_10_90": calibration,
        })

    # --- Write per-feature CSV ---
    import csv
    csv_path = RESULTS_DIR / f"exp1_{dataset_name}.csv"
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWrote {csv_path}")

    # --- Gate verdict ---
    n_beats = sum(r["beats_marginal"] for r in records)
    frac_beats = n_beats / len(records)
    avg_mse_tabpfn = np.mean([r["mse_tabpfn"] for r in records])
    avg_mse_marginal = np.mean([r["mse_marginal"] for r in records])
    avg_calibration = np.mean([r["calibration_10_90"] for r in records])

    if dataset_name == "moons":
        # Moons: both features should be well below marginal (class-separated 2-D)
        if frac_beats >= 1.0:
            verdict = "PASS"
        elif frac_beats >= 0.5:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
    else:
        # HELOC: majority of 23 features should beat marginal
        if frac_beats >= 0.5:
            verdict = "PASS"
        elif frac_beats >= 0.3:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

    print(f"\nGate verdict ({dataset_name}): {verdict}")
    print(f"  Beats marginal: {n_beats}/{len(records)} features ({frac_beats:.0%})")
    print(f"  Avg MSE — marginal={avg_mse_marginal:.4f} tabpfn={avg_mse_tabpfn:.4f}")
    print(f"  Avg calibration (10-90%): {avg_calibration:.2f}")

    # --- MOONS scatter plot ---
    if dataset_name == "moons":
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))
            if n_features == 1:
                axes = [axes]
            for j, ax in enumerate(axes):
                true_vals_j = X_test[:, j]
                # Collect point estimates: re-derive from records
                # (re-run would be expensive; derive from CSV data instead we
                # just re-predict for plot since n_features=2 and test is small)
                ax.set_title(f"Feature {feat_names[j]}")
                ax.set_xlabel("True value")
                ax.set_ylabel("TabPFN predicted")
                ax.set_aspect("equal")

            # Re-prediction per feature using the same per-class conditioning
            # that was used during scoring (not unconditional context).
            for j in range(n_features):
                pred_j = np.zeros(len(X_test))
                for cls in np.unique(y_test):
                    cls_mask = y_test == cls
                    X_cls = X_test[cls_mask]
                    s = ConditionalDensitySampler(
                        clf=clf, reg=reg, n_permutations=5, temperature=1e-9,
                        random_state=42 + j,
                    )
                    s.set_context(
                        X_train, y_context=y_train,
                        target_class=int(cls), max_context=MAX_CONTEXT,
                    )
                    pred_j[cls_mask] = s.sample_feature(X_cls, target_col=j, n_samples=1)
                ax = axes[j]
                ax.scatter(X_test[:, j], pred_j, alpha=0.5, s=20)
                ax.plot([0, 1], [0, 1], "r--", lw=1)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            plt.tight_layout()
            plot_path = RESULTS_DIR / "exp1_moons.png"
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"Wrote {plot_path}")
        except ImportError:
            print("matplotlib not available; skipping scatter plot")

    return records, verdict, frac_beats, avg_mse_tabpfn, avg_mse_marginal, avg_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: single-feature reconstruction")
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "all"],
        default="moons",
        help="Dataset to run (default: moons)",
    )
    args = parser.parse_args()

    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]

    summary_rows = []
    for ds in datasets:
        records, verdict, frac_beats, avg_mse_tf, avg_mse_mg, avg_calib = run_experiment(ds)
        summary_rows.append({
            "dataset": ds,
            "n_features": len(records),
            "beats_marginal_frac": frac_beats,
            "avg_mse_marginal": avg_mse_mg,
            "avg_mse_tabpfn": avg_mse_tf,
            "avg_calibration_10_90": avg_calib,
            "gate_verdict": verdict,
        })

    # Write exp1_summary.md
    summary_path = RESULTS_DIR / "exp1_summary.md"
    lines = [
        "# Experiment 1: Single-Feature Reconstruction — Summary",
        "",
        "## Results",
        "",
        "| Dataset | Features | Beats marginal | Avg MSE marginal | Avg MSE TabPFN | Avg calib 10-90% | Gate verdict |",
        "|---------|----------|----------------|-----------------|---------------|-----------------|--------------|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['dataset']} | {r['n_features']} | "
            f"{r['beats_marginal_frac']:.0%} | "
            f"{r['avg_mse_marginal']:.4f} | "
            f"{r['avg_mse_tabpfn']:.4f} | "
            f"{r['avg_calibration_10_90']:.2f} | "
            f"**{r['gate_verdict']}** |"
        )
    lines += [
        "",
        "## Gate Verdict Definitions",
        "",
        "- **PASS**: TabPFN beats the marginal-mean baseline on ≥50% of features (HELOC) or all features (MOONS). Proceed to Stage 5 with confidence.",
        "- **WEAK**: Beats marginal on ≥30% (HELOC) or ≥50% (MOONS). Proceed to Stage 5 but flag low expectations; refinement may be needed.",
        "- **FAIL**: Does not beat marginal baseline. Record that Experiment 2 is unlikely to work out-of-the-box; refinement focus shifts to context/temperature.",
        "",
        "## Notes",
        "",
        "- Context: same-class train rows (capped at 256), near-MAP temperature t=1e-9.",
        "- Calibration: fraction of true values inside the [10%, 90%] interval of 50 posterior samples at t=1.0.",
        "- Ridge baseline: RidgeCV trained to predict feature j from the other features.",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
