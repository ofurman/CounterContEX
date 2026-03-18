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
from tabpfn.priors.counterfactual_prior import (
    FixedSCMDataLoader, SCMFamilyDataLoader, DiverseSCMDataLoader,
)
from tabpfn.experiments.configs import EXPERIMENT_REGISTRY


def _evaluate_fixed_scm(model, fixed_dl, num_test_datasets, num_features,
                        seq_len, save_path=None, mask_supervision=True):
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
    # Also share cached normalization stats
    test_dl._cached_mean = fixed_dl._cached_mean
    test_dl._cached_std = fixed_dl._cached_std

    # Generate test batches and collect matched SCM internals for each
    test_dl.return_internals = True
    scm_data_list = []
    for data, target_y, sep, batch_internals in test_dl:
        style, x, y = data
        test_batches.append((x, y, target_y))
        # Use the exact internals from data generation (matched noise)
        for internals in batch_internals:
            scm_data_list.append({
                "scm": fixed_dl.scm,
                "internals": internals,
                "class_assigner": fixed_dl.class_assigner,
            })

    print(f"Generated {len(test_batches)} test batches from fixed SCM")

    pred_deltas, true_deltas, query_x, target_labels = run_inference(
        model, test_batches, single_eval_pos, num_features, "cpu",
        mask_supervision=mask_supervision,
    )

    # Pass normalization stats so SCM validity can un-normalize predictions
    norm_stats = None
    if hasattr(fixed_dl, '_cached_mean') and fixed_dl._cached_mean is not None:
        norm_stats = (fixed_dl._cached_mean, fixed_dl._cached_std)

    metrics = compute_metrics(
        pred_deltas, true_deltas, query_x, target_labels,
        num_features, "cpu",
        scm_data_list=scm_data_list,
        single_eval_pos=single_eval_pos,
        norm_stats=norm_stats,
    )
    metrics.num_test_datasets = num_test_datasets

    print_report(metrics)
    if save_path:
        save_examples(pred_deltas, true_deltas, query_x, target_labels, save_path)

    return metrics


def _evaluate_scm_family(model, family_dl, num_test_datasets, num_features,
                         seq_len, save_path=None, mask_supervision=True):
    """Evaluate a model trained on an SCM family.

    Generates test data from each SCM in the family, computes metrics,
    and runs context ablation to verify in-context learning.
    """
    from tabpfn.eval_counterfactual import compute_scm_validity

    single_eval_pos = seq_len // 2

    # Generate test batches — each batch element uses a random SCM from family
    test_dl = SCMFamilyDataLoader(
        num_features=num_features,
        seq_len=seq_len,
        batch_size=1,
        num_scms=family_dl.num_scms,
        hyperparameters=family_dl.hyperparameters,
        device="cpu",
        single_eval_pos=single_eval_pos,
        num_steps=num_test_datasets,
    )
    # Share the same SCM family
    test_dl.scms = family_dl.scms
    test_dl.fixed_perms = family_dl.fixed_perms
    test_dl.class_assigners = family_dl.class_assigners
    test_dl._cached_mean = family_dl._cached_mean
    test_dl._cached_std = family_dl._cached_std

    test_dl.return_internals = True
    test_batches = []
    scm_data_list = []
    for data, target_y, sep, batch_internals in test_dl:
        style, x, y = data
        test_batches.append((x, y, target_y))
        for internals in batch_internals:
            scm_idx = internals["scm_idx"]
            scm_data_list.append({
                "scm": family_dl.scms[scm_idx],
                "internals": internals,
                "class_assigner": family_dl.class_assigners[scm_idx],
            })

    print(f"Generated {len(test_batches)} test batches from SCM family ({family_dl.num_scms} SCMs)")

    pred_deltas, true_deltas, query_x, target_labels = run_inference(
        model, test_batches, single_eval_pos, num_features, "cpu",
        mask_supervision=mask_supervision,
    )

    norm_stats = (family_dl._cached_mean, family_dl._cached_std)

    metrics = compute_metrics(
        pred_deltas, true_deltas, query_x, target_labels,
        num_features, "cpu",
        scm_data_list=scm_data_list,
        single_eval_pos=single_eval_pos,
        norm_stats=norm_stats,
    )
    metrics.num_test_datasets = num_test_datasets

    print_report(metrics)
    if save_path:
        save_examples(pred_deltas, true_deltas, query_x, target_labels, save_path)

    # --- Context ablation diagnostic ---
    print("\n--- Context Ablation Diagnostic ---")
    ablation_results = _context_ablation(
        model, family_dl, num_features, seq_len, single_eval_pos,
        mask_supervision, num_test=min(num_test_datasets, 20),
    )
    for key, val in ablation_results.items():
        print(f"  {key}: SCM validity = {val:.4f}")

    return metrics, ablation_results


def _context_ablation(model, family_dl, num_features, seq_len,
                      single_eval_pos, mask_supervision, num_test=20):
    """Run context ablation to verify in-context learning.

    Three conditions:
    1. Correct context: context from same SCM as query (normal)
    2. Wrong context: context from a different SCM than query
    3. No context: zero out context y values

    Returns dict of condition -> SCM validity.
    """
    import random as pyrandom
    from tabpfn.eval_counterfactual import compute_scm_validity
    from tabpfn.priors.counterfactual_prior import get_batch_fixed_scm

    results = {}

    # Generate test batches with correct context
    correct_batches = []
    correct_scm_data = []
    wrong_batches = []
    wrong_scm_data = []
    no_ctx_batches = []
    no_ctx_scm_data = []

    for t in range(num_test):
        # Pick a query SCM
        q_idx = pyrandom.randint(0, family_dl.num_scms - 1)
        # Pick a different SCM for wrong context
        w_idx = (q_idx + 1) % family_dl.num_scms

        # Generate data from the query SCM
        result = get_batch_fixed_scm(
            batch_size=1,
            seq_len=seq_len,
            num_features=num_features,
            hyperparameters=family_dl.hyperparameters,
            device="cpu",
            single_eval_pos=single_eval_pos,
            num_outputs=family_dl._n_outputs,
            scm=family_dl.scms[q_idx],
            fixed_perm=family_dl.fixed_perms[q_idx],
            class_assigner=family_dl.class_assigners[q_idx],
            cached_norm_stats=(family_dl._cached_mean, family_dl._cached_std),
            return_internals=True,
        )
        x, y, target_y, batch_int = result
        scm_entry = {
            "scm": family_dl.scms[q_idx],
            "internals": batch_int[0],
            "class_assigner": family_dl.class_assigners[q_idx],
        }

        # 1. Correct context (normal)
        correct_batches.append((x, y, target_y))
        correct_scm_data.append(scm_entry)

        # 2. Wrong context — replace context y with y from different SCM
        x_wrong = x.clone()
        y_wrong = y.clone()
        result_w = get_batch_fixed_scm(
            batch_size=1,
            seq_len=seq_len,
            num_features=num_features,
            hyperparameters=family_dl.hyperparameters,
            device="cpu",
            single_eval_pos=single_eval_pos,
            num_outputs=family_dl._n_outputs,
            scm=family_dl.scms[w_idx],
            fixed_perm=family_dl.fixed_perms[w_idx],
            class_assigner=family_dl.class_assigners[w_idx],
            cached_norm_stats=(family_dl._cached_mean, family_dl._cached_std),
        )
        x_w, y_w, _ = result_w
        # Replace context portion with wrong SCM's data
        x_wrong[:single_eval_pos] = x_w[:single_eval_pos]
        y_wrong[:single_eval_pos] = y_w[:single_eval_pos]
        wrong_batches.append((x_wrong, y_wrong, target_y))
        wrong_scm_data.append(scm_entry)

        # 3. No context — zero out context y
        y_no = y.clone()
        y_no[:single_eval_pos] = 0.0
        no_ctx_batches.append((x, y_no, target_y))
        no_ctx_scm_data.append(scm_entry)

    norm_stats = (family_dl._cached_mean, family_dl._cached_std)

    for name, batches, scm_data in [
        ("correct_context", correct_batches, correct_scm_data),
        ("wrong_context", wrong_batches, wrong_scm_data),
        ("no_context", no_ctx_batches, no_ctx_scm_data),
    ]:
        pred_deltas, true_deltas, query_x, target_labels = run_inference(
            model, batches, single_eval_pos, num_features, "cpu",
            mask_supervision=mask_supervision,
        )
        scm_val = compute_scm_validity(
            query_x, pred_deltas, target_labels,
            scm_data, single_eval_pos, num_features,
            norm_stats=norm_stats,
        )
        results[name] = scm_val

    return results


def _evaluate_diverse_scm(model, diverse_dl, num_test_datasets, num_features,
                          seq_len, save_path=None, mask_supervision=True):
    """Evaluate a model trained on diverse SCMs.

    Generates test data from new random SCMs (unseen at training time),
    computes metrics with SCM-based validity, and runs context ablation.
    """
    from tabpfn.eval_counterfactual import compute_scm_validity
    from tabpfn.priors.counterfactual_prior import get_batch_with_scm

    single_eval_pos = seq_len // 2

    # Generate test batches from new random SCMs (diverse structure)
    test_batches = []
    scm_data_list = []
    for t in range(num_test_datasets):
        hp = diverse_dl._random_hp()
        x, y, target_y, batch_scm_data = get_batch_with_scm(
            batch_size=1,
            seq_len=seq_len,
            num_features=num_features,
            hyperparameters=hp,
            device="cpu",
            single_eval_pos=single_eval_pos,
        )
        test_batches.append((x, y, target_y))
        scm_data_list.extend(batch_scm_data)

    print(f"Generated {len(test_batches)} test batches from diverse SCMs")

    pred_deltas, true_deltas, query_x, target_labels = run_inference(
        model, test_batches, single_eval_pos, num_features, "cpu",
        mask_supervision=mask_supervision,
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

    # --- Context ablation diagnostic ---
    print("\n--- Context Ablation Diagnostic ---")
    ablation_results = _context_ablation_diverse(
        model, diverse_dl, num_features, seq_len, single_eval_pos,
        mask_supervision, num_test=min(num_test_datasets, 20),
    )
    for key, val in ablation_results.items():
        print(f"  {key}: SCM validity = {val:.4f}")

    return metrics, ablation_results


def _context_ablation_diverse(model, diverse_dl, num_features, seq_len,
                              single_eval_pos, mask_supervision, num_test=20):
    """Run context ablation for diverse-SCM model.

    Three conditions:
    1. Correct context: context and query from same SCM (normal)
    2. Wrong context: context from a different SCM than query
    3. No context: zero out context y values

    Returns dict of condition -> SCM validity.
    """
    from tabpfn.eval_counterfactual import compute_scm_validity
    from tabpfn.priors.counterfactual_prior import get_batch_with_scm

    results = {}
    correct_batches, correct_scm = [], []
    wrong_batches, wrong_scm = [], []
    no_ctx_batches, no_ctx_scm = [], []

    for t in range(num_test):
        # Generate query data from one random SCM
        hp_q = diverse_dl._random_hp()
        x, y, target_y, scm_data = get_batch_with_scm(
            batch_size=1, seq_len=seq_len, num_features=num_features,
            hyperparameters=hp_q, device="cpu",
            single_eval_pos=single_eval_pos,
        )
        scm_entry = scm_data[0]

        # 1. Correct context (normal)
        correct_batches.append((x, y, target_y))
        correct_scm.append(scm_entry)

        # 2. Wrong context — context from a different SCM
        hp_w = diverse_dl._random_hp()
        x_w, y_w, _, _ = get_batch_with_scm(
            batch_size=1, seq_len=seq_len, num_features=num_features,
            hyperparameters=hp_w, device="cpu",
            single_eval_pos=single_eval_pos,
        )
        x_wrong = x.clone()
        y_wrong = y.clone()
        x_wrong[:single_eval_pos] = x_w[:single_eval_pos]
        y_wrong[:single_eval_pos] = y_w[:single_eval_pos]
        wrong_batches.append((x_wrong, y_wrong, target_y))
        wrong_scm.append(scm_entry)

        # 3. No context — zero out context y
        y_no = y.clone()
        y_no[:single_eval_pos] = 0.0
        no_ctx_batches.append((x, y_no, target_y))
        no_ctx_scm.append(scm_entry)

    for name, batches, scm_data in [
        ("correct_context", correct_batches, correct_scm),
        ("wrong_context", wrong_batches, wrong_scm),
        ("no_context", no_ctx_batches, no_ctx_scm),
    ]:
        pred_deltas, true_deltas, query_x, target_labels = run_inference(
            model, batches, single_eval_pos, num_features, "cpu",
            mask_supervision=mask_supervision,
        )
        scm_val = compute_scm_validity(
            query_x, pred_deltas, target_labels,
            scm_data, single_eval_pos, num_features,
        )
        results[name] = scm_val

    return results


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
    use_mask_supervision = extra_prior_kwargs.pop("mask_supervision", True)
    num_scms = extra_prior_kwargs.pop("num_scms", None)
    use_diverse_scm = extra_prior_kwargs.pop("diverse_scm", False)
    layer_range = extra_prior_kwargs.pop("layer_range", (2, 5))
    hidden_dim_range = extra_prior_kwargs.pop("hidden_dim_range", (8, 33))

    # Build data loader based on config
    fixed_dl = None
    family_dl = None
    diverse_dl = None
    if use_diverse_scm:
        hp = dict(extra_prior_kwargs)
        hp["mask_supervision"] = use_mask_supervision
        diverse_dl = DiverseSCMDataLoader(
            num_features=num_features,
            seq_len=exp_config["seq_len"],
            batch_size=exp_config["batch_size"],
            hyperparameters=hp,
            device="cpu",
            single_eval_pos=exp_config["seq_len"] // 2,
            num_steps=exp_config["steps_per_epoch"],
            layer_range=layer_range,
            hidden_dim_range=hidden_dim_range,
        )
        print(f"  Using DiverseSCMDataLoader (new random SCM per batch element)")
    elif num_scms is not None and num_scms > 1:
        # SCM family mode (Exp 3+)
        hp = dict(extra_prior_kwargs)
        hp["mask_supervision"] = use_mask_supervision
        family_dl = SCMFamilyDataLoader(
            num_features=num_features,
            seq_len=exp_config["seq_len"],
            batch_size=exp_config["batch_size"],
            num_scms=num_scms,
            hyperparameters=hp,
            device="cpu",
            single_eval_pos=exp_config["seq_len"] // 2,
            num_steps=exp_config["steps_per_epoch"],
        )
        print(f"  Using SCMFamilyDataLoader ({num_scms} frozen SCMs)")
    elif use_fixed_scm:
        hp = dict(extra_prior_kwargs)
        hp["mask_supervision"] = use_mask_supervision
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
        mask_supervision=use_mask_supervision,
        mask_loss_weight=0.5,
        dataloader=diverse_dl or family_dl or fixed_dl,
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
    ablation_results = None
    if diverse_dl is not None:
        # Diverse SCM evaluation with context ablation
        metrics, ablation_results = _evaluate_diverse_scm(
            model, diverse_dl, num_test_datasets, num_features,
            exp_config["seq_len"], str(out / "example_predictions.json"),
            mask_supervision=use_mask_supervision,
        )
    elif family_dl is not None:
        # SCM family evaluation with context ablation
        metrics, ablation_results = _evaluate_scm_family(
            model, family_dl, num_test_datasets, num_features,
            exp_config["seq_len"], str(out / "example_predictions.json"),
            mask_supervision=use_mask_supervision,
        )
    elif use_fixed_scm and fixed_dl is not None:
        # Generate test data from the same fixed SCM
        metrics = _evaluate_fixed_scm(
            model, fixed_dl, num_test_datasets, num_features,
            exp_config["seq_len"], str(out / "example_predictions.json"),
            mask_supervision=use_mask_supervision,
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
    if ablation_results is not None:
        results["context_ablation"] = ablation_results

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
    if metrics.sign_accuracy >= 0:
        print(f"  Sign accuracy:     {metrics.sign_accuracy:.4f}")
    if metrics.scm_validity >= 0:
        print(f"  SCM validity:      {metrics.scm_validity:.4f}")
    if metrics.zero_feature_accuracy >= 0:
        print(f"  Zero-feat acc:     {metrics.zero_feature_accuracy:.4f}")
    if ablation_results is not None:
        print(f"  Context ablation:")
        for key, val in ablation_results.items():
            print(f"    {key}: {val:.4f}")
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

    if "sign_accuracy" in criteria:
        val = metrics.get("sign_accuracy", -1)
        passed = val >= criteria["sign_accuracy"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Sign accuracy: {val:.4f} >= {criteria['sign_accuracy']}")
        all_passed = all_passed and passed

    if "scm_validity" in criteria:
        val = metrics.get("scm_validity", -1)
        passed = val >= criteria["scm_validity"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] SCM validity: {val:.4f} >= {criteria['scm_validity']}")
        all_passed = all_passed and passed

    if "zero_feature_accuracy" in criteria:
        val = metrics.get("zero_feature_accuracy", -1)
        passed = val >= criteria["zero_feature_accuracy"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Zero-feat acc: {val:.4f} >= {criteria['zero_feature_accuracy']}")
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

    if "context_ablation_gap" in criteria and "context_ablation" in results:
        ablation = results["context_ablation"]
        correct = ablation.get("correct_context", 0)
        wrong = ablation.get("wrong_context", 0)
        gap = correct - wrong
        passed = gap >= criteria["context_ablation_gap"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Context ablation gap: {gap:.4f} >= {criteria['context_ablation_gap']}")
        all_passed = all_passed and passed

    overall = "PASSED" if all_passed else "FAILED"
    print(f"\n  Overall: {overall}")
    print(f"{'=' * 60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
