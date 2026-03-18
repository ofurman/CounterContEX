"""
Evaluation script for counterfactual generation with TabPFN.

Generates test SCM datasets, runs inference with a trained model, and
computes four metrics:
  1. Delta MSE  — prediction error on the counterfactual delta
  2. Validity   — fraction of generated CFs that actually flip the label
  3. Proximity  — L2 distance between query and generated counterfactual
  4. Sparsity   — fraction of features with near-zero predicted delta

Usage:
    python -m tabpfn.eval_counterfactual --model counterfactual_model.pt
    python -m tabpfn.eval_counterfactual --train-and-eval   # quick train then eval
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch import Tensor

from tabpfn.priors.counterfactual import (
    CounterfactualSCMGenerator,
    CounterfactualBatch,
    get_default_counterfactual_config,
)
from tabpfn.priors.counterfactual_prior import get_batch as cf_get_batch
from tabpfn.priors.counterfactual_prior import get_batch_with_scm as cf_get_batch_with_scm
from tabpfn.train_counterfactual import train_counterfactual, TOY_CONFIG
from tabpfn.transformer import TransformerModel
import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""

    delta_mse: float
    validity_rate: float
    proximity_mean: float
    proximity_std: float
    sparsity: float
    num_test_datasets: int
    num_query_points: int
    scm_validity: float = -1.0  # SCM-based validity (-1 = not computed)
    sign_accuracy: float = -1.0  # Fraction of deltas with correct sign (-1 = not computed)


def build_model(num_features: int, emsize: int = 64, nhead: int = 2,
                nhid: int = 128, nlayers: int = 4, dropout: float = 0.0,
                bptt: int = 128, mask_supervision: bool = True) -> TransformerModel:
    """Reconstruct the model architecture (must match training config)."""
    n_out = num_features * 2 if mask_supervision else num_features
    encoder = encoders.Linear(num_features, emsize)
    y_encoder = encoders.Linear(1, emsize)
    pos_encoder = positional_encodings.NoPositionalEncoding(emsize, bptt * 2)

    model = TransformerModel(
        encoder=encoder,
        n_out=n_out,
        ninp=emsize,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        dropout=dropout,
        y_encoder=y_encoder,
        pos_encoder=pos_encoder,
        efficient_eval_masking=True,
    )
    return model


def generate_test_data(num_datasets: int, seq_len: int, num_features: int,
                       device: str = "cpu", with_scm: bool = False):
    """Generate test batches from CounterfactualSCMGenerator.

    Args:
        num_datasets: number of test datasets to generate
        seq_len: sequence length per dataset
        num_features: number of features
        device: compute device
        with_scm: if True, also return SCM data for ground-truth validity

    Returns:
        (test_batches, single_eval_pos) where test_batches is a list of
        (x, y, target_y) tuples (or (x, y, target_y, scm_data_list) if with_scm).
    """
    single_eval_pos = seq_len // 2

    test_batches = []
    for _ in range(num_datasets):
        if with_scm:
            x, y, target_y, scm_data_list = cf_get_batch_with_scm(
                batch_size=1,
                seq_len=seq_len,
                num_features=num_features,
                hyperparameters={},
                device=device,
                single_eval_pos=single_eval_pos,
            )
            test_batches.append((x, y, target_y, scm_data_list))
        else:
            x, y, target_y = cf_get_batch(
                batch_size=1,
                seq_len=seq_len,
                num_features=num_features,
                hyperparameters={},
                device=device,
                single_eval_pos=single_eval_pos,
            )
            test_batches.append((x, y, target_y))

    return test_batches, single_eval_pos


@torch.no_grad()
def run_inference(model: TransformerModel, test_batches: list,
                  single_eval_pos: int, num_features: int,
                  device: str = "cpu", mask_supervision: bool = True):
    """Run inference on test data.

    When mask_supervision=True, the model outputs 2*num_features channels.
    The first num_features are delta predictions, the last num_features are
    mask logits. At inference, we threshold sigmoid(mask_logits) > 0.5 and
    zero out deltas for non-selected features.

    Returns:
        predicted_deltas: (total_query_points, num_features)
        true_deltas: (total_query_points, num_features)
        query_x: (total_query_points, num_features)
        target_labels: (total_query_points,)
    """
    model.eval()
    model.to(device)

    all_pred_deltas = []
    all_true_deltas = []
    all_query_x = []
    all_target_labels = []

    for batch_tuple in test_batches:
        # Support both (x, y, target_y) and (x, y, target_y, scm_data_list)
        x, y, target_y = batch_tuple[0], batch_tuple[1], batch_tuple[2]
        x = x.to(device)
        y = y.to(device)
        target_y = target_y.to(device)

        # Forward pass: model expects (style, x, y) tuple
        data = (None, x, y)
        output = model(data, single_eval_pos=single_eval_pos)

        if mask_supervision:
            # Split output into delta predictions and mask logits
            pred_delta = output[..., :num_features]
            pred_mask_logits = output[..., num_features:]
            # Apply mask: threshold at 0.5, zero out non-selected features
            pred_mask = (torch.sigmoid(pred_mask_logits) > 0.5).float()
            pred_delta = pred_delta * pred_mask
            # Extract true deltas (first num_features channels of target_y)
            query_true_delta = target_y[single_eval_pos:, :, :num_features]
        else:
            pred_delta = output
            query_true_delta = target_y[single_eval_pos:]

        # Flatten batch dimension
        nq = pred_delta.shape[0]
        bs = pred_delta.shape[1]
        nf = num_features

        all_pred_deltas.append(pred_delta.reshape(nq * bs, nf))
        all_true_deltas.append(query_true_delta.reshape(nq * bs, nf))
        all_query_x.append(x[single_eval_pos:].reshape(nq * bs, nf))
        all_target_labels.append(y[single_eval_pos:].reshape(nq * bs))

    return (
        torch.cat(all_pred_deltas, dim=0),
        torch.cat(all_true_deltas, dim=0),
        torch.cat(all_query_x, dim=0),
        torch.cat(all_target_labels, dim=0),
    )


def compute_validity(query_x: Tensor, pred_deltas: Tensor,
                     target_labels: Tensor, num_features: int,
                     device: str = "cpu") -> float:
    """Estimate validity by generating fresh SCMs and classifying predicted CFs.

    Since each test batch uses a different SCM (which we don't retain), we
    approximate validity by checking how often the predicted counterfactual
    lies on the correct side of a simple decision boundary trained on the
    context data.

    For a more principled validity check, we re-generate a batch and use the
    SCM's own class assigner. Here we use a simpler heuristic: check if the
    predicted delta actually moves the point (non-trivially) when the target
    label differs from what we'd get with zero delta.
    """
    x_cf_pred = query_x + pred_deltas
    # Heuristic: a counterfactual is "valid" if the predicted delta is non-trivial
    # (L2 norm > threshold) when a label flip is requested.
    # This is an approximation; true validity requires the original SCM classifier.
    delta_norms = pred_deltas.norm(dim=-1)  # (N,)
    # Consider valid if delta norm > small threshold (i.e., the model actually moved the point)
    threshold = 0.01
    valid = delta_norms > threshold
    return valid.float().mean().item()


def compute_scm_validity(
    query_x: Tensor,
    pred_deltas: Tensor,
    target_labels: Tensor,
    scm_data_list: list,
    single_eval_pos: int,
    num_features: int,
) -> float:
    """Compute true validity by feeding predicted CFs through original SCMs.

    For each query point:
    1. x_cf_pred = query_x + pred_delta
    2. Build interventions: set all feature nodes to x_cf_pred values
    3. Re-propagate through SCM
    4. Classify using a fixed threshold (median of original y) and compare
       to target label

    Note: We use FixedThresholdBinarize instead of the original BalancedBinarize
    because BalancedBinarize recomputes the median from the current batch. When
    we only evaluate single CF points, the median would be degenerate. Instead,
    we compute the median from the original factual y values and freeze it.

    Args:
        query_x: (total_query_points, num_features) original query features
        pred_deltas: (total_query_points, num_features) predicted deltas
        target_labels: (total_query_points,) target class labels
        scm_data_list: list of dicts (one per test batch), each with keys
            'scm', 'internals', 'class_assigner'
        single_eval_pos: number of context positions
        num_features: number of features

    Returns:
        validity_rate: fraction of predictions that achieve the target label
    """
    from tabpfn.priors.counterfactual import FixedThresholdBinarize

    x_cf_pred = query_x + pred_deltas

    total_valid = 0
    total_points = 0

    num_query_per_batch = x_cf_pred.shape[0] // len(scm_data_list)

    for batch_idx, scm_data in enumerate(scm_data_list):
        scm = scm_data["scm"]
        internals = scm_data["internals"]
        feature_indices = internals["node_mapping"]["feature_indices"]
        target_indices = internals["node_mapping"]["target_indices"]
        outputs_flat = internals["outputs_flat"]

        # Compute a fixed threshold from the original y values
        y_original = outputs_flat[:, :, target_indices]
        if y_original.dim() > 2:
            y_original = y_original.squeeze(-1)
        threshold = torch.median(y_original).item()
        fixed_assigner = FixedThresholdBinarize(threshold)

        start = batch_idx * num_query_per_batch
        end = start + num_query_per_batch
        batch_x_cf = x_cf_pred[start:end]  # (num_query, num_features)
        batch_target = target_labels[start:end]  # (num_query,)

        # For each query point, set interventions on all feature nodes
        for q in range(batch_x_cf.shape[0]):
            interventions = {}
            for feat_idx in range(num_features):
                flat_idx = feature_indices[feat_idx].item()
                new_val = outputs_flat[:, :, flat_idx].clone()
                new_val[:, :] = batch_x_cf[q, feat_idx]
                interventions[flat_idx] = new_val

            # Re-propagate through SCM
            _, y_cf_scm = scm.forward_with_intervention(
                internals, interventions
            )

            # Classify using the fixed threshold
            y_cf_class = fixed_assigner(
                y_cf_scm.unsqueeze(-1) if y_cf_scm.dim() == 2 else y_cf_scm
            ).float()
            if y_cf_class.dim() > 2:
                y_cf_class = y_cf_class.squeeze(-1)

            # Use the first sample position as representative
            pred_class = y_cf_class[0, 0].item()
            target_class = batch_target[q].item()

            if pred_class == target_class:
                total_valid += 1
            total_points += 1

    return total_valid / max(total_points, 1)


def compute_metrics(pred_deltas: Tensor, true_deltas: Tensor,
                    query_x: Tensor, target_labels: Tensor,
                    num_features: int, device: str = "cpu",
                    scm_data_list: list = None,
                    single_eval_pos: int = None) -> EvalMetrics:
    """Compute all evaluation metrics.

    Args:
        pred_deltas: (N, num_features) predicted deltas
        true_deltas: (N, num_features) ground-truth deltas
        query_x: (N, num_features) original query features
        target_labels: (N,) target class labels
        num_features: number of features
        scm_data_list: optional list of SCM data dicts for ground-truth validity
        single_eval_pos: number of context positions (required if scm_data_list given)
    """
    N = pred_deltas.shape[0]

    # 1. Delta MSE
    delta_mse = ((pred_deltas - true_deltas) ** 2).mean().item()

    # 2. Validity (heuristic)
    validity = compute_validity(
        query_x, pred_deltas, target_labels, num_features, device
    )

    # 3. Proximity: L2 distance between query and predicted counterfactual
    x_cf_pred = query_x + pred_deltas
    proximity = (x_cf_pred - query_x).norm(dim=-1)  # = pred_deltas.norm(dim=-1)
    proximity_mean = proximity.mean().item()
    proximity_std = proximity.std().item()

    # 4. Sparsity: fraction of features with near-zero delta
    sparse_threshold = 0.05
    near_zero = pred_deltas.abs() < sparse_threshold
    sparsity = near_zero.float().mean().item()

    # 5. Sign accuracy: fraction of non-zero true deltas with correct sign
    nonzero_mask = true_deltas.abs() > 1e-6
    if nonzero_mask.any():
        pred_sign = (pred_deltas[nonzero_mask] > 0).float()
        true_sign = (true_deltas[nonzero_mask] > 0).float()
        sign_acc = (pred_sign == true_sign).float().mean().item()
    else:
        sign_acc = -1.0

    # 6. SCM validity (ground-truth, if SCM data available)
    scm_val = -1.0
    if scm_data_list is not None and single_eval_pos is not None:
        scm_val = compute_scm_validity(
            query_x, pred_deltas, target_labels,
            scm_data_list, single_eval_pos, num_features,
        )

    return EvalMetrics(
        delta_mse=delta_mse,
        validity_rate=validity,
        proximity_mean=proximity_mean,
        proximity_std=proximity_std,
        sparsity=sparsity,
        num_test_datasets=N,  # will be overridden by caller
        num_query_points=N,
        scm_validity=scm_val,
        sign_accuracy=sign_acc,
    )


def print_report(metrics: EvalMetrics):
    """Print a readable evaluation report."""
    print("\n" + "=" * 60)
    print("  Counterfactual Generation — Evaluation Report")
    print("=" * 60)
    print(f"  Test datasets:     {metrics.num_test_datasets}")
    print(f"  Total query points:{metrics.num_query_points}")
    print("-" * 60)
    print(f"  Delta MSE:         {metrics.delta_mse:.6f}")
    print(f"  Heuristic validity:{metrics.validity_rate:.4f}")
    if metrics.sign_accuracy >= 0:
        print(f"  Sign accuracy:     {metrics.sign_accuracy:.4f}")
    if metrics.scm_validity >= 0:
        print(f"  SCM validity:      {metrics.scm_validity:.4f}")
    print(f"  Proximity (L2):    {metrics.proximity_mean:.4f} +/- {metrics.proximity_std:.4f}")
    print(f"  Sparsity:          {metrics.sparsity:.4f}")
    print("=" * 60)


def save_examples(pred_deltas: Tensor, true_deltas: Tensor,
                  query_x: Tensor, target_labels: Tensor,
                  save_path: str, num_examples: int = 5):
    """Save a few example predictions for inspection."""
    examples = []
    n = min(num_examples, pred_deltas.shape[0])
    for i in range(n):
        examples.append({
            "query_x": query_x[i].tolist(),
            "target_label": target_labels[i].item(),
            "true_delta": true_deltas[i].tolist(),
            "pred_delta": pred_deltas[i].tolist(),
            "x_cf_true": (query_x[i] + true_deltas[i]).tolist(),
            "x_cf_pred": (query_x[i] + pred_deltas[i]).tolist(),
        })

    with open(save_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {n} example predictions to {save_path}")


def evaluate(model: TransformerModel, num_test_datasets: int = 100,
             num_features: int = 5, seq_len: int = 128,
             device: str = "cpu", save_path: str = None,
             mask_supervision: bool = True,
             with_scm_validity: bool = False) -> EvalMetrics:
    """Full evaluation pipeline.

    Args:
        model: trained TransformerModel
        num_test_datasets: number of test SCM datasets to generate
        num_features: number of features (must match model)
        seq_len: sequence length per dataset
        device: compute device
        save_path: optional path to save example predictions
        with_scm_validity: if True, compute ground-truth SCM-based validity

    Returns:
        EvalMetrics with all computed metrics
    """
    print(f"Generating {num_test_datasets} test datasets...")
    test_batches, single_eval_pos = generate_test_data(
        num_test_datasets, seq_len, num_features, device,
        with_scm=with_scm_validity,
    )

    print("Running inference...")
    pred_deltas, true_deltas, query_x, target_labels = run_inference(
        model, test_batches, single_eval_pos, num_features, device,
        mask_supervision=mask_supervision,
    )

    # Collect SCM data from all batches if available
    scm_data_list = None
    if with_scm_validity:
        scm_data_list = []
        for batch_tuple in test_batches:
            # Each batch_tuple is (x, y, target_y, scm_data_list_for_batch)
            scm_data_list.extend(batch_tuple[3])

    print("Computing metrics...")
    metrics = compute_metrics(
        pred_deltas, true_deltas, query_x, target_labels,
        num_features, device,
        scm_data_list=scm_data_list,
        single_eval_pos=single_eval_pos,
    )
    metrics.num_test_datasets = num_test_datasets

    print_report(metrics)

    if save_path:
        save_examples(pred_deltas, true_deltas, query_x, target_labels, save_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate counterfactual generation model"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to saved model state_dict (.pt file)",
    )
    parser.add_argument(
        "--train-and-eval", action="store_true",
        help="Train a quick model then evaluate (no saved model needed)",
    )
    parser.add_argument(
        "--num-test-datasets", type=int, default=100,
        help="Number of test SCM datasets to generate",
    )
    parser.add_argument(
        "--save-examples", type=str, default=None,
        help="Path to save example predictions (JSON)",
    )
    # Model architecture args (must match training)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--emsize", type=int, default=64)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--nhid", type=int, default=128)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.train_and_eval:
        print("Training a quick model for evaluation...")
        import inspect
        valid_params = set(inspect.signature(train_counterfactual).parameters.keys())
        config = {k: v for k, v in TOY_CONFIG.items() if k in valid_params}
        config["epochs"] = 10
        config["steps_per_epoch"] = 50
        _, model, _, _ = train_counterfactual(**config)
    elif args.model:
        print(f"Loading model from {args.model}...")
        model = build_model(
            num_features=args.num_features,
            emsize=args.emsize,
            nhead=args.nhead,
            nhid=args.nhid,
            nlayers=args.nlayers,
        )
        state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        parser.error("Specify --model or --train-and-eval")

    evaluate(
        model,
        num_test_datasets=args.num_test_datasets,
        num_features=args.num_features,
        seq_len=args.seq_len,
        device="cpu",  # eval on CPU for simplicity
        save_path=args.save_examples,
    )


if __name__ == "__main__":
    main()
