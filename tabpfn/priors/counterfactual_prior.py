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
    FixedThresholdBinarize,
    get_default_counterfactual_config,
)
from tabpfn.priors.utils import get_batch_to_dataloader

# When mask supervision is enabled, target_y has 2x num_features channels:
# first num_features = delta values, last num_features = intervention mask (0/1)


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
    mask_supervision = config.pop("mask_supervision", True)

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
        mask_supervision=mask_supervision,
    )

    # Per-batch-element feature normalization
    normalize = config.get("normalize_features", True)
    if normalize:
        x, target_y = _normalize_per_batch(x, target_y)

    return x, y, target_y


def _normalize_per_batch(x, target_y):
    """Normalize features to zero mean / unit variance per batch element.

    Also scales deltas by the same standard deviation so they are expressed
    in units of feature standard deviations. If target_y contains mask channels
    (shape[-1] == 2 * num_features), only the delta channels are scaled.

    Args:
        x: (seq_len, batch_size, num_features)
        target_y: (seq_len, batch_size, num_features) or (seq_len, batch_size, num_features*2)

    Returns:
        x_norm, target_y_norm with same shapes
    """
    num_features = x.shape[2]
    batch_size = x.shape[1]
    for b in range(batch_size):
        mean = x[:, b, :].mean(dim=0, keepdim=True)   # (1, num_features)
        std = x[:, b, :].std(dim=0, keepdim=True).clamp(min=1e-6)  # (1, num_features)
        x[:, b, :] = (x[:, b, :] - mean) / std
        # Scale only the delta channels, not the mask channels
        target_y[:, b, :num_features] = target_y[:, b, :num_features] / std
    return x, target_y


def _reorder_and_encode(batch, single_eval_pos, num_features, device,
                        flip_only_queries=True, mask_supervision=True):
    """Reorder samples so label-flipped ones are prioritized for query positions.

    When flip_only_queries=True, all query positions will have label-flipped
    samples. If fewer flipped samples than query slots, flipped samples are
    duplicated with small Gaussian noise to fill remaining query positions.

    When mask_supervision=True, target_y contains 2*num_features channels:
    first num_features = delta values, last num_features = intervention mask (0/1).

    Returns:
        x: (seq_len, batch_size, num_features)
        y: (seq_len, batch_size) — factual labels for context, target labels for queries
        target_y: (seq_len, batch_size, num_features) or (seq_len, batch_size, num_features*2)
    """
    seq_len = batch.x_factual.shape[0]
    batch_size = batch.x_factual.shape[1]
    num_query = seq_len - single_eval_pos

    target_channels = num_features * 2 if mask_supervision else num_features
    x = torch.zeros(seq_len, batch_size, num_features, device=device)
    y = torch.zeros(seq_len, batch_size, device=device)
    target_y = torch.zeros(seq_len, batch_size, target_channels, device=device)

    def _set_target(pos, b, src_i):
        """Set target_y at (pos, b) with delta and optionally intervention mask."""
        delta = batch.x_counterfactual[src_i, b] - batch.x_factual[src_i, b]
        target_y[pos, b, :num_features] = delta
        if mask_supervision:
            target_y[pos, b, num_features:] = batch.intervention_mask[src_i, b].float()

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
                _set_target(i, b, src_i)

            # Fill query positions
            for i, src_i in enumerate(query_idx):
                pos = single_eval_pos + i
                x[pos, b] = batch.x_factual[src_i, b]
                y[pos, b] = batch.y_counterfactual_class[src_i, b]
                _set_target(pos, b, src_i)
        elif flip_only_queries and len(flipped_idx) > 0:
            # Not enough flipped samples but flip_only_queries is on:
            # duplicate flipped samples with small noise to fill query slots
            context_idx = non_flipped_idx[:single_eval_pos]

            # Fill context positions
            for i, src_i in enumerate(context_idx):
                x[i, b] = batch.x_factual[src_i, b]
                y[i, b] = batch.y_factual_class[src_i, b]
                _set_target(i, b, src_i)

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
                _set_target(pos, b, src_i)
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
                _set_target(i, b, src_i)

            # Fill query positions
            for i, src_i in enumerate(query_idx):
                pos = single_eval_pos + i
                x[pos, b] = batch.x_factual[src_i, b]
                y[pos, b] = batch.y_counterfactual_class[src_i, b]
                _set_target(pos, b, src_i)

    return x, y, target_y


DataLoader = get_batch_to_dataloader(get_batch)


def get_batch_with_scm(
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
    """Generate a batch of counterfactual data plus SCM objects for validity evaluation.

    Like get_batch but also returns a list of SCM data dicts (one per batch
    element) containing the SCM, internals, and class_assigner needed for
    ground-truth validity checking.

    Returns:
        (x, y, target_y, scm_data_list) where the first three have the same
        format as get_batch, and scm_data_list is a list of dicts with keys
        'scm', 'internals', 'class_assigner'.
    """
    config = dict(get_default_counterfactual_config())
    if hyperparameters:
        config.update(hyperparameters)

    flip_only_queries = config.pop("flip_only_queries", True)
    min_flip_rate = config.pop("min_flip_rate", 0.20)
    max_retries = config.pop("max_retries", 5)
    mask_supervision = config.pop("mask_supervision", True)

    gen = CounterfactualSCMGenerator(config, device=device)

    batch = None
    scm_data_list = None
    for attempt in range(max_retries):
        batch, scm_data_list = gen.generate_batch_with_scm(
            batch_size=batch_size,
            seq_len=seq_len,
            num_features=num_features,
            num_outputs=num_outputs,
        )
        flip_rate = batch.label_flipped.float().mean().item()
        if flip_rate >= min_flip_rate:
            break
        gen.config["perturbation_strategy"] = "uniform_random"
        gen.config["perturbation_magnitude"] = gen.config["perturbation_magnitude"] * 1.5

    x, y, target_y = _reorder_and_encode(
        batch, single_eval_pos, num_features, device,
        flip_only_queries=flip_only_queries,
        mask_supervision=mask_supervision,
    )

    normalize = config.get("normalize_features", True)
    if normalize:
        x, target_y = _normalize_per_batch(x, target_y)

    return x, y, target_y, scm_data_list


def get_batch_fixed_scm(
    batch_size: int,
    seq_len: int,
    num_features: int,
    hyperparameters: dict,
    device: str = "cpu",
    single_eval_pos: int = 64,
    num_outputs: int = 1,
    epoch: int = None,
    *,
    scm,
    fixed_perm,
    class_assigner,
    **kwargs,
):
    """Generate training data from a single fixed SCM.

    Like get_batch but uses a pre-built SCM with fixed node mapping and
    frozen class boundary for consistent data generation across batches.

    Args:
        scm: Pre-built _MLP instance
        fixed_perm: Fixed node permutation tensor
        class_assigner: Pre-built class assigner (e.g., FixedThresholdBinarize)
        (other args same as get_batch)

    Returns (x, y, target_y) with same format as get_batch.
    """
    config = dict(get_default_counterfactual_config())
    if hyperparameters:
        config.update(hyperparameters)

    flip_only_queries = config.pop("flip_only_queries", True)
    mask_supervision = config.pop("mask_supervision", True)
    # Remove retry-related keys (not needed for fixed SCM)
    config.pop("min_flip_rate", None)
    config.pop("max_retries", None)

    gen = CounterfactualSCMGenerator(config, device=device)

    batch = gen.generate_batch_fixed_scm(
        scm=scm,
        fixed_perm=fixed_perm,
        class_assigner=class_assigner,
        batch_size=batch_size,
        seq_len=seq_len,
        num_features=num_features,
        num_outputs=num_outputs,
    )

    x, y, target_y = _reorder_and_encode(
        batch, single_eval_pos, num_features, device,
        flip_only_queries=flip_only_queries,
        mask_supervision=mask_supervision,
    )

    normalize = config.get("normalize_features", True)
    if normalize:
        x, target_y = _normalize_per_batch(x, target_y)

    return x, y, target_y


class FixedSCMDataLoader:
    """Data loader that pre-creates a single SCM and reuses it for every batch.

    At initialization:
    1. Builds one SCM with random weights
    2. Runs forward_with_internals_fixed_mapping to get a fixed node permutation
    3. Calibrates a FixedThresholdBinarize from a large calibration batch

    Every __iter__ call generates fresh samples from the same SCM structure.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        batch_size: int,
        hyperparameters: dict = None,
        device: str = "cpu",
        single_eval_pos: int = 64,
        num_outputs: int = 1,
        num_steps: int = 100,
        calibration_size: int = 10000,
    ):
        self.num_features = num_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hyperparameters = hyperparameters or {}
        self.device = device
        self.single_eval_pos = single_eval_pos
        self.num_outputs = num_outputs
        self.num_steps = num_steps

        # Build config
        config = dict(get_default_counterfactual_config())
        config.update(self.hyperparameters)

        # Build SCM
        gen = CounterfactualSCMGenerator(config, device=device)
        self.scm = gen._sample_scm(seq_len, num_features, num_outputs)

        # Get fixed node permutation
        with torch.no_grad():
            _, y_cal, _, self.fixed_perm = self.scm.forward_with_internals_fixed_mapping()

        # Calibrate class boundary from a large batch
        with torch.no_grad():
            # Generate calibration data: use a larger seq_len temporarily
            cal_scm = gen._sample_scm(calibration_size, num_features, num_outputs)
            # We need to reuse the same MLP weights, so rebuild with calibration size
            # Instead, run multiple forward passes and collect y values
            y_values = []
            remaining = calibration_size
            while remaining > 0:
                chunk = min(remaining, seq_len)
                # Use a temporary SCM with the right seq_len
                _, y_chunk, _, _ = self.scm.forward_with_internals_fixed_mapping(
                    self.fixed_perm
                )
                y_values.append(y_chunk.squeeze(-1) if y_chunk.dim() > 2 else y_chunk)
                remaining -= chunk
            y_all = torch.cat(y_values, dim=0)
            threshold = torch.median(y_all).item()

        self.class_assigner = FixedThresholdBinarize(threshold)

    def __iter__(self):
        for _ in range(self.num_steps):
            yield get_batch_fixed_scm(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                num_features=self.num_features,
                hyperparameters=self.hyperparameters,
                device=self.device,
                single_eval_pos=self.single_eval_pos,
                num_outputs=self.num_outputs,
                scm=self.scm,
                fixed_perm=self.fixed_perm,
                class_assigner=self.class_assigner,
            )

    def __len__(self):
        return self.num_steps
