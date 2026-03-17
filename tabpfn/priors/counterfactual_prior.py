"""
Counterfactual prior data loader for TabPFN training.

Wraps CounterfactualSCMGenerator to produce (x, y, target_y) batches
compatible with TabPFN's training loop via get_batch_to_dataloader.

Encoding strategy:
- Context tokens (0..single_eval_pos-1): x = factual features, y = factual class label
- Query tokens (single_eval_pos..seq_len-1): x = factual features, y = target class label
- Targets: delta = x_counterfactual - x_factual (for all positions; only query used by trainer)
"""

import torch
from torch import Tensor

from tabpfn.utils import default_device
from tabpfn.priors.counterfactual import (
    CounterfactualSCMGenerator,
    get_default_counterfactual_config,
)
from tabpfn.priors.utils import get_batch_to_dataloader


def get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    hyperparameters: dict,
    device: str = default_device,
    single_eval_pos: int = 64,
    num_outputs: int = 1,
    epoch: int = None,
    **kwargs,
):
    """Generate a batch of counterfactual training data.

    Returns (x, y, target_y) where:
    - x: (seq_len, batch_size, num_features) factual features for all positions
    - y: (seq_len, batch_size) class labels — factual for context, target for queries
    - target_y: (seq_len, batch_size, num_features) deltas (x_cf - x_factual)
    """
    config = dict(get_default_counterfactual_config())
    if hyperparameters:
        config.update(hyperparameters)

    flip_only_queries = config.pop("flip_only_queries", True)
    min_flip_rate = config.pop("min_flip_rate", 0.20)
    max_retries = config.pop("max_retries", 5)

    gen = CounterfactualSCMGenerator(config, device=device)

    # Retry loop: regenerate if flip rate is too low
    batch = None
    for attempt in range(max_retries):
        batch = gen.generate_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            num_outputs=num_outputs,
        )
        flip_rate = batch.label_flipped.float().mean().item()
        if flip_rate >= min_flip_rate:
            break
        # Retry with uniform_random strategy and increased magnitude
        gen.config["perturbation_strategy"] = "uniform_random"
        gen.config["perturbation_magnitude"] = gen.config["perturbation_magnitude"] * 1.5

    # Reorder samples per batch element: label-flipped → query, non-flipped → context
    x, y, target_y = _reorder_and_encode(
        batch, single_eval_pos, num_features, device,
        flip_only_queries=flip_only_queries,
    )

    # Per-batch-element feature normalization
    normalize = config.get("normalize_features", True)
    if normalize:
        x, target_y = _normalize_per_batch(x, target_y)

    return x, y, target_y


def _normalize_per_batch(x, target_y):
    """Normalize features to zero mean / unit variance per batch element.

    Also scales deltas by the same standard deviation so they are expressed
    in units of feature standard deviations.

    Args:
        x: (seq_len, batch_size, num_features)
        target_y: (seq_len, batch_size, num_features) — deltas

    Returns:
        x_norm, target_y_norm with same shapes
    """
    batch_size = x.shape[1]
    for b in range(batch_size):
        mean = x[:, b, :].mean(dim=0, keepdim=True)   # (1, num_features)
        std = x[:, b, :].std(dim=0, keepdim=True).clamp(min=1e-6)  # (1, num_features)
        x[:, b, :] = (x[:, b, :] - mean) / std
        # Scale deltas by the same std so they're in normalized space
        target_y[:, b, :] = target_y[:, b, :] / std
    return x, target_y


def _reorder_and_encode(batch, single_eval_pos, num_features, device,
                        flip_only_queries=True):
    """Reorder samples so label-flipped ones are prioritized for query positions.

    When flip_only_queries=True, all query positions will have label-flipped
    samples. If fewer flipped samples than query slots, flipped samples are
    duplicated with small Gaussian noise to fill remaining query positions.

    Returns:
        x: (seq_len, batch_size, num_features)
        y: (seq_len, batch_size) — factual labels for context, target labels for queries
        target_y: (seq_len, batch_size, num_features) — deltas
    """
    seq_len = batch.x_factual.shape[0]
    batch_size = batch.x_factual.shape[1]
    num_query = seq_len - single_eval_pos

    x = torch.zeros(seq_len, batch_size, num_features, device=device)
    y = torch.zeros(seq_len, batch_size, device=device)
    target_y = torch.zeros(seq_len, batch_size, num_features, device=device)

    for b in range(batch_size):
        flipped_mask = batch.label_flipped[:, b]  # (seq_len,)
        flipped_idx = torch.where(flipped_mask)[0]
        non_flipped_idx = torch.where(~flipped_mask)[0]

        # Prioritize flipped samples for query positions
        if len(flipped_idx) >= num_query:
            # Enough flipped samples: use them for queries, rest for context
            query_idx = flipped_idx[:num_query]
            # Context: use non-flipped first, then remaining flipped
            remaining_flipped = flipped_idx[num_query:]
            context_pool = torch.cat([non_flipped_idx, remaining_flipped])
            context_idx = context_pool[:single_eval_pos]

            # Fill context positions
            for i, src_i in enumerate(context_idx):
                x[i, b] = batch.x_factual[src_i, b]
                y[i, b] = batch.y_factual_class[src_i, b]
                target_y[i, b] = (
                    batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
                )

            # Fill query positions
            for i, src_i in enumerate(query_idx):
                pos = single_eval_pos + i
                x[pos, b] = batch.x_factual[src_i, b]
                y[pos, b] = batch.y_counterfactual_class[src_i, b]
                target_y[pos, b] = (
                    batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
                )
        elif flip_only_queries and len(flipped_idx) > 0:
            # Not enough flipped samples but flip_only_queries is on:
            # duplicate flipped samples with small noise to fill query slots
            context_idx = non_flipped_idx[:single_eval_pos]

            # Fill context positions
            for i, src_i in enumerate(context_idx):
                x[i, b] = batch.x_factual[src_i, b]
                y[i, b] = batch.y_factual_class[src_i, b]
                target_y[i, b] = (
                    batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
                )

            # Fill query positions: cycle through flipped samples
            for i in range(num_query):
                src_i = flipped_idx[i % len(flipped_idx)]
                pos = single_eval_pos + i
                if i < len(flipped_idx):
                    # Original flipped sample
                    x[pos, b] = batch.x_factual[src_i, b]
                else:
                    # Duplicated flipped sample with small noise
                    noise = torch.randn(num_features, device=device) * 0.01
                    x[pos, b] = batch.x_factual[src_i, b] + noise
                y[pos, b] = batch.y_counterfactual_class[src_i, b]
                target_y[pos, b] = (
                    batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
                )
        else:
            # No flipped samples or flip_only_queries is off:
            # fall back to filling queries with non-flipped samples
            fill_count = num_query - len(flipped_idx)
            if len(non_flipped_idx) >= fill_count + single_eval_pos:
                query_idx = torch.cat([
                    flipped_idx,
                    non_flipped_idx[:fill_count],
                ]) if len(flipped_idx) > 0 else non_flipped_idx[:num_query]
                context_idx = non_flipped_idx[fill_count:fill_count + single_eval_pos] if len(flipped_idx) > 0 else non_flipped_idx[num_query:num_query + single_eval_pos]
            else:
                # Edge case: not enough samples overall, just split sequentially
                all_idx = torch.arange(seq_len, device=device)
                context_idx = all_idx[:single_eval_pos]
                query_idx = all_idx[single_eval_pos:]

            # Fill context positions
            for i, src_i in enumerate(context_idx):
                x[i, b] = batch.x_factual[src_i, b]
                y[i, b] = batch.y_factual_class[src_i, b]
                target_y[i, b] = (
                    batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
                )

            # Fill query positions
            for i, src_i in enumerate(query_idx):
                pos = single_eval_pos + i
                x[pos, b] = batch.x_factual[src_i, b]
                y[pos, b] = batch.y_counterfactual_class[src_i, b]
                target_y[pos, b] = (
                    batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
                )

    return x, y, target_y


DataLoader = get_batch_to_dataloader(get_batch)
