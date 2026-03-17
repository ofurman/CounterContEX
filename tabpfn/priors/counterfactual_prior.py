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

    gen = CounterfactualSCMGenerator(config, device=device)
    batch = gen.generate_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        num_features=num_features,
        num_outputs=num_outputs,
    )

    # Reorder samples per batch element: label-flipped → query, non-flipped → context
    x, y, target_y = _reorder_and_encode(
        batch, single_eval_pos, num_features, device
    )

    return x, y, target_y


def _reorder_and_encode(batch, single_eval_pos, num_features, device):
    """Reorder samples so label-flipped ones are prioritized for query positions.

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
        else:
            # Not enough flipped: use all flipped for queries, fill rest with non-flipped
            fill_count = num_query - len(flipped_idx)
            if len(non_flipped_idx) >= fill_count + single_eval_pos:
                query_idx = torch.cat([
                    flipped_idx,
                    non_flipped_idx[:fill_count],
                ])
                context_idx = non_flipped_idx[fill_count:fill_count + single_eval_pos]
            else:
                # Edge case: not enough samples overall, just split sequentially
                all_idx = torch.arange(seq_len, device=device)
                context_idx = all_idx[:single_eval_pos]
                query_idx = all_idx[single_eval_pos:]

        # Fill context positions (0..single_eval_pos-1)
        for i, src_i in enumerate(context_idx):
            x[i, b] = batch.x_factual[src_i, b]
            y[i, b] = batch.y_factual_class[src_i, b]
            target_y[i, b] = (
                batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
            )

        # Fill query positions (single_eval_pos..seq_len-1)
        for i, src_i in enumerate(query_idx):
            pos = single_eval_pos + i
            x[pos, b] = batch.x_factual[src_i, b]
            y[pos, b] = batch.y_counterfactual_class[src_i, b]  # target label
            target_y[pos, b] = (
                batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
            )

    return x, y, target_y


DataLoader = get_batch_to_dataloader(get_batch)
