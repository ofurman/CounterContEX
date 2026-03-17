"""Unified experiment runner for counterfactual generation.

Takes a config and executes: data generator setup, model creation,
training with logging, evaluation with all metrics, and results output.

Usage:
    python -m tabpfn.experiments.run_experiment --experiment exp0
    python -m tabpfn.experiments.run_experiment --experiment exp1 --output-dir docs/results/exp1
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from tabpfn.train_counterfactual import train_counterfactual
from tabpfn.eval_counterfactual import (
    build_model,
    evaluate,
    run_inference,
    compute_metrics,
    print_report,
    save_examples,
    EvalMetrics,
)
from tabpfn.priors.counterfactual_prior import FixedSCMDataLoader
from tabpfn.experiments.configs import EXPERIMENT_REGISTRY


def _evaluate_fixed_scm(model, fixed_dl, num_test_datasets, num_features,
                        seq_len, save_path=None):
    """Evaluate a model using the same fixed SCM from training.

    Generates test data from the training SCM and computes SCM-based validity
    by re-running predicted counterfactuals through the original causal model.
    """
    from tabpfn.eval_counterfactual import compute_scm_validity

    test_batches = []
    single_eval_pos = seq_len // 2
    test_dl = FixedSCMDataLoader(
        num_features=num_features,
        seq_len=seq_len,
        batch_size=1,
        hyperparameters=fixed_dl.hyperparameters,
        device="cpu",
        single_eval_pos=single_eval_pos,
        num_steps=num_test_datasets,
        calibration_size=200,
    )
    # Use the SAME SCM and class assigner from training
    test_dl.scm = fixed_dl.scm
    test_dl.fixed_perm = fixed_dl.fixed_perm
    test_dl.class_assigner = fixed_dl.class_assigner

    # Generate test batches and collect SCM internals for each
    scm_data_list = []
    for data, target_y, sep in test_dl:
        style, x, y = data
        test_batches.append((x, y, target_y))
        # Get internals from the fixed SCM for SCM validity
        with torch.no_grad():
            _, _, internals, _ = fixed_dl.scm.forward_with_internals_fixed_mapping(
                fixed_dl.fixed_perm
            )
        scm_data_list.append({
            "scm": fixed_dl.scm,
            "internals": internals,
            "class_assigner": fixed_dl.class_assigner,
        })

    print(f"Generated {len(test_batches)} test batches from fixed SCM")

    pred_deltas, true_deltas, query_x, target_labels = run_inference(
        model, test_batches, single_eval_pos, num_features, "cpu",
        mask_supervision=True,
    )

    metrics = compute_metrics(
        pred_deltas, true_deltas, query_x, target_labels,
        num_features, "cpu",
        scm_data_list=scm_data_list,
        single_eval_pos=single_eval_pos,
    )
    metrics.num_test_datasets = num_test_datasets

    print_report(metrics)
    if save_path:
        save_examples(pred_deltas, true_deltas, query_x, target_labels, save_path)

    return metrics


def run_experiment(
    exp_config: dict,
    scm_config: dict,
    output_dir: str,
    device: str = "cpu",
    aggregate_k: int = 4,
    num_test_datasets: int = 50,
    with_scm_validity: bool = True,
) -> dict:
    """Run a complete experiment: setup -> train -> evaluate -> report.

    Args:
        exp_config: model/training params (emsize, nlayers, epochs, ...)
        scm_config: SCM/data params (num_features, num_layers, ...)
        output_dir: path for results and checkpoints
        device: 'cpu' or 'cuda'
        aggregate_k: gradient accumulation steps
        num_test_datasets: number of test datasets for evaluation
        with_scm_validity: whether to compute SCM-based validity

    Returns:
        results dict with all metrics and training log
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_features = exp_config["num_features"]
    epochs = exp_config["epochs"]

    # ---- Training with per-epoch logging ----
    training_log = []
    best_loss = float("inf")
    best_epoch = 0
    best_state_dict = None

    def epoch_callback(model, progress):
        """Called after each epoch to log and checkpoint."""
        nonlocal best_loss, best_epoch, best_state_dict
        epoch_num = int(progress * epochs)
        current_loss = training_log[-1]["loss"] if training_log else float("inf")
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch_num
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Prepare prior kwargs from SCM config
    extra_prior_kwargs = dict(scm_config)
    use_fixed_scm = extra_prior_kwargs.pop("use_fixed_scm", False)

    # Build fixed-SCM data loader if requested
    fixed_dl = None
    if use_fixed_scm:
        hp = dict(extra_prior_kwargs)
        hp["mask_supervision"] = True
        fixed_dl = FixedSCMDataLoader(
            num_features=num_features,
            seq_len=exp_config["seq_len"],
            batch_size=exp_config["batch_size"],
            hyperparameters=hp,
            device="cpu",
            single_eval_pos=exp_config["seq_len"] // 2,
            num_steps=exp_config["steps_per_epoch"],
        )
        print(f"  Using FixedSCMDataLoader (single frozen SCM)")

    print(f"\n{'=' * 60}")
    print(f"  Running experiment: {num_features} features, {epochs} epochs")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    total_loss, model, dl, loss_history = train_counterfactual(
        num_features=num_features,
        seq_len=exp_config["seq_len"],
        batch_size=exp_config["batch_size"],
        emsize=exp_config["emsize"],
        nlayers=exp_config["nlayers"],
        nhead=exp_config["nhead"],
        nhid=exp_config["nhid"],
        dropout=exp_config["dropout"],
        epochs=epochs,
        steps_per_epoch=exp_config["steps_per_epoch"],
        lr=exp_config["lr"],
        bptt=exp_config["seq_len"],
        warmup_epochs=exp_config.get("warmup_epochs", 5),
        weight_decay=exp_config.get("weight_decay", 0.0),
        gpu_device=device if "cuda" in device else "cuda:0",
        extra_prior_kwargs=extra_prior_kwargs,
        verbose=True,
        epoch_callback=epoch_callback,
        mask_supervision=True,
        mask_loss_weight=0.5,
        dataloader=fixed_dl,
    )

    train_time = time.time() - start_time

    # Build training log from loss history
    for i, loss in enumerate(loss_history):
        training_log.append({
            "epoch": i + 1,
            "loss": loss,
        })
        # Update best tracking
        if loss < best_loss:
            best_loss = loss
            best_epoch = i + 1

    # Save best model checkpoint
    if best_state_dict is not None:
        checkpoint_path = out / "best_model.pt"
        torch.save(best_state_dict, checkpoint_path)
        print(f"\nBest model saved to {checkpoint_path} (epoch {best_epoch}, loss {best_loss:.6f})")
    else:
        # If callback never fired, save final model
        checkpoint_path = out / "best_model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        best_loss = total_loss
        best_epoch = epochs
        print(f"\nFinal model saved to {checkpoint_path} (loss {total_loss:.6f})")

    # ---- Evaluation ----
    print(f"\nEvaluating with {num_test_datasets} test datasets...")
    if use_fixed_scm and fixed_dl is not None:
        # Generate test data from the same fixed SCM
        metrics = _evaluate_fixed_scm(
            model, fixed_dl, num_test_datasets, num_features,
            exp_config["seq_len"], str(out / "example_predictions.json"),
        )
    else:
        metrics = evaluate(
            model,
            num_test_datasets=num_test_datasets,
            num_features=num_features,
            seq_len=exp_config["seq_len"],
            device="cpu",
            save_path=str(out / "example_predictions.json"),
            mask_supervision=True,
            with_scm_validity=with_scm_validity,
        )

    # ---- Compute additional metrics ----
    # Sign accuracy: fraction of deltas with correct sign
    # (computed from example predictions if available)

    # ---- Assemble results ----
    results = {
        "config": exp_config,
        "scm_config": scm_config,
        "metrics": asdict(metrics),
        "training": {
            "final_loss": total_loss,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "total_epochs": epochs,
            "train_time_seconds": train_time,
            "initial_loss": loss_history[0] if loss_history else None,
            "loss_reduction": (
                1 - best_loss / loss_history[0]
                if loss_history and loss_history[0] > 0
                else None
            ),
        },
        "training_log": training_log,
    }

    # Save results
    results_path = out / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save training log separately for easy plotting
    log_path = out / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    # ---- Summary report ----
    print(f"\n{'=' * 60}")
    print("  Experiment Summary")
    print(f"{'=' * 60}")
    print(f"  Training time:     {train_time:.1f}s")
    print(f"  Best loss:         {best_loss:.6f} (epoch {best_epoch})")
    if loss_history:
        reduction = 1 - best_loss / loss_history[0] if loss_history[0] > 0 else 0
        print(f"  Loss reduction:    {reduction:.1%}")
    print(f"  Delta MSE:         {metrics.delta_mse:.6f}")
    print(f"  Heuristic validity:{metrics.validity_rate:.4f}")
    if metrics.scm_validity >= 0:
        print(f"  SCM validity:      {metrics.scm_validity:.4f}")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run counterfactual generation experiment"
    )
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENT_REGISTRY.keys()),
        required=True,
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: docs/results/<experiment>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda:N)",
    )
    parser.add_argument(
        "--num-test-datasets",
        type=int,
        default=50,
        help="Number of test datasets for evaluation",
    )
    parser.add_argument(
        "--no-scm-validity",
        action="store_true",
        help="Skip SCM-based validity evaluation (faster)",
    )
    args = parser.parse_args()

    exp_config, scm_config, criteria = EXPERIMENT_REGISTRY[args.experiment]
    output_dir = args.output_dir or f"docs/results/{args.experiment}"

    results = run_experiment(
        exp_config=exp_config,
        scm_config=scm_config,
        output_dir=output_dir,
        device=args.device,
        num_test_datasets=args.num_test_datasets,
        with_scm_validity=not args.no_scm_validity,
    )

    # Check against success criteria
    print(f"\n{'=' * 60}")
    print("  Success Criteria Check")
    print(f"{'=' * 60}")
    all_passed = True
    metrics = results["metrics"]
    training = results["training"]

    if "delta_mse" in criteria:
        passed = metrics["delta_mse"] < criteria["delta_mse"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Delta MSE: {metrics['delta_mse']:.6f} < {criteria['delta_mse']}")
        all_passed = all_passed and passed

    if "scm_validity" in criteria:
        val = metrics.get("scm_validity", -1)
        passed = val >= criteria["scm_validity"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] SCM validity: {val:.4f} >= {criteria['scm_validity']}")
        all_passed = all_passed and passed

    if "loss_reduction" in criteria and training["loss_reduction"] is not None:
        passed = training["loss_reduction"] >= criteria["loss_reduction"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Loss reduction: {training['loss_reduction']:.1%} >= {criteria['loss_reduction']:.0%}")
        all_passed = all_passed and passed

    if "max_epochs_to_converge" in criteria:
        # Find the epoch where loss first dropped below 1% of initial
        threshold = 0.01 * training["initial_loss"] if training["initial_loss"] else 0
        converge_epoch = training["total_epochs"]
        for entry in results.get("training_log", []):
            if entry["loss"] <= threshold:
                converge_epoch = entry["epoch"]
                break
        passed = converge_epoch <= criteria["max_epochs_to_converge"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Converged at epoch {converge_epoch} <= {criteria['max_epochs_to_converge']}")
        all_passed = all_passed and passed

    overall = "PASSED" if all_passed else "FAILED"
    print(f"\n  Overall: {overall}")
    print(f"{'=' * 60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
