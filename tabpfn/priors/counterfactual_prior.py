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


def _normalize_with_cached_stats(x, target_y, cached_mean, cached_std):
    """Normalize features using pre-computed (cached) mean and std.

    Unlike _normalize_per_batch which computes stats from the current batch,
    this uses fixed stats from a calibration set. This ensures stationary
    normalization across batches for fixed-SCM experiments.

    Args:
        x: (seq_len, batch_size, num_features)
        target_y: (seq_len, batch_size, num_features) or (seq_len, batch_size, num_features*2)
        cached_mean: (1, 1, num_features) mean from calibration data
        cached_std: (1, 1, num_features) std from calibration data

    Returns:
        x_norm, target_y_norm with same shapes
    """
    num_features = x.shape[2]
    # cached_mean/std have shape (1, 1, num_features) — broadcast over seq_len and batch
    mean = cached_mean.squeeze(0)  # (1, num_features)
    std = cached_std.squeeze(0)    # (1, num_features)
    x = (x - mean) / std
    # Scale only the delta channels, not the mask channels
    target_y = target_y.clone()
    target_y[..., :num_features] = target_y[..., :num_features] / std
    return x, target_y


def _reorder_and_encode(batch, single_eval_pos, num_features, device,
                        flip_only_queries=True, mask_supervision=True,
                        return_query_indices=False):
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
    if return_query_indices:
        query_source_indices = torch.zeros(num_query, batch_size, dtype=torch.long, device=device)

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
                if return_query_indices:
                    query_source_indices[i, b] = src_i
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
                if return_query_indices:
                    query_source_indices[i, b] = src_i
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
                if return_query_indices:
                    query_source_indices[i, b] = src_i

    if return_query_indices:
        return x, y, target_y, query_source_indices
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
    cached_norm_stats=None,
    return_internals: bool = False,
    **kwargs,
):
    """Generate training data from a single fixed SCM.

    Like get_batch but uses a pre-built SCM with fixed node mapping and
    frozen class boundary for consistent data generation across batches.

    Args:
        scm: Pre-built _MLP instance
        fixed_perm: Fixed node permutation tensor
        class_assigner: Pre-built class assigner (e.g., FixedThresholdBinarize)
        cached_norm_stats: Optional (mean, std) tuple from calibration data.
            When provided, uses these fixed stats instead of per-batch stats,
            ensuring stationary normalization across batches.
        return_internals: If True, also return the per-batch-element internals
            from data generation (for SCM validity checking).
        (other args same as get_batch)

    Returns:
        (x, y, target_y) when return_internals=False (default).
        (x, y, target_y, batch_internals) when return_internals=True.
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

    batch, batch_internals = gen.generate_batch_fixed_scm(
        scm=scm,
        fixed_perm=fixed_perm,
        class_assigner=class_assigner,
        batch_size=batch_size,
        seq_len=seq_len,
        num_features=num_features,
        num_outputs=num_outputs,
    )

    reorder_result = _reorder_and_encode(
        batch, single_eval_pos, num_features, device,
        flip_only_queries=flip_only_queries,
        mask_supervision=mask_supervision,
        return_query_indices=return_internals,
    )
    if return_internals:
        x, y, target_y, query_source_indices = reorder_result
        # Attach query source indices to each batch element's internals
        for b, internals in enumerate(batch_internals):
            internals["query_source_indices"] = query_source_indices[:, b]
    else:
        x, y, target_y = reorder_result

    normalize = config.get("normalize_features", True)
    if normalize:
        if cached_norm_stats is not None:
            # Use fixed normalization stats for stationary targets
            x, target_y = _normalize_with_cached_stats(
                x, target_y, cached_norm_stats[0], cached_norm_stats[1]
            )
        else:
            x, target_y = _normalize_per_batch(x, target_y)

    if return_internals:
        return x, y, target_y, batch_internals
    return x, y, target_y


class FixedSCMDataLoader:
    """Data loader that pre-creates a single SCM and reuses it for every batch.

    At initialization:
    1. Builds one SCM with random weights
    2. Runs forward_with_internals_fixed_mapping to get a fixed node permutation
    3. Calibrates a FixedThresholdBinarize from a large calibration batch
    4. Computes and caches normalization stats (mean/std) from calibration data

    Every __iter__ call generates fresh samples from the same SCM structure,
    using cached normalization stats for consistency across batches.
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
        self._n_outputs = num_outputs  # avoid "num_outputs" attr (check_compatibility)
        self.num_steps = num_steps
        self.return_internals = False

        # Build config
        config = dict(get_default_counterfactual_config())
        config.update(self.hyperparameters)

        # Build SCM
        gen = CounterfactualSCMGenerator(config, device=device)
        self.scm = gen._sample_scm(seq_len, num_features, self._n_outputs)

        # Get fixed node permutation
        with torch.no_grad():
            _, y_cal, _, self.fixed_perm = self.scm.forward_with_internals_fixed_mapping()

        # Calibrate class boundary from a large batch
        with torch.no_grad():
            y_values = []
            x_values = []
            remaining = calibration_size
            while remaining > 0:
                chunk = min(remaining, seq_len)
                x_chunk, y_chunk, _, _ = self.scm.forward_with_internals_fixed_mapping(
                    self.fixed_perm
                )
                y_values.append(y_chunk.squeeze(-1) if y_chunk.dim() > 2 else y_chunk)
                x_values.append(x_chunk)
                remaining -= chunk
            y_all = torch.cat(y_values, dim=0)
            threshold = torch.median(y_all).item()

            # Cache normalization stats from calibration data
            x_all = torch.cat(x_values, dim=0)  # (total_samples, 1, num_features)
            self._cached_mean = x_all.mean(dim=0, keepdim=True)  # (1, 1, num_features)
            self._cached_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)

        self.class_assigner = FixedThresholdBinarize(threshold)

    def __iter__(self):
        for _ in range(self.num_steps):
            result = get_batch_fixed_scm(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                num_features=self.num_features,
                hyperparameters=self.hyperparameters,
                device=self.device,
                single_eval_pos=self.single_eval_pos,
                num_outputs=self._n_outputs,
                scm=self.scm,
                fixed_perm=self.fixed_perm,
                class_assigner=self.class_assigner,
                cached_norm_stats=(self._cached_mean, self._cached_std),
                return_internals=self.return_internals,
            )
            if self.return_internals:
                x, y, target_y, batch_internals = result
                yield (None, x, y), target_y, self.single_eval_pos, batch_internals
            else:
                x, y, target_y = result
                # Match the standard DataLoader output format:
                # ((style, x, y), target_y, single_eval_pos)
                yield (None, x, y), target_y, self.single_eval_pos

    def __len__(self):
        return self.num_steps


class SCMFamilyDataLoader:
    """Data loader that pre-creates a family of N SCMs and randomly selects one per batch element.

    Used for in-context learning experiments where the model must identify
    which SCM generated the context data and predict correct counterfactuals.

    At initialization:
    1. Builds N SCMs, each with its own fixed node permutation
    2. Calibrates a FixedThresholdBinarize per SCM from calibration data
    3. Computes shared normalization stats across all SCMs

    Every batch randomly assigns each element to one of the N SCMs.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        batch_size: int,
        num_scms: int = 10,
        hyperparameters: dict = None,
        device: str = "cpu",
        single_eval_pos: int = 64,
        num_outputs: int = 1,
        num_steps: int = 100,
        calibration_size: int = 2000,
    ):
        import random as pyrandom

        self.num_features = num_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_scms = num_scms
        self.hyperparameters = hyperparameters or {}
        self.device = device
        self.single_eval_pos = single_eval_pos
        self._n_outputs = num_outputs
        self.num_steps = num_steps
        self.return_internals = False

        # Build config
        config = dict(get_default_counterfactual_config())
        config.update(self.hyperparameters)

        gen = CounterfactualSCMGenerator(config, device=device)

        # Pre-generate family of SCMs
        self.scms = []
        self.fixed_perms = []
        self.class_assigners = []

        all_x_cal = []
        for i in range(num_scms):
            scm = gen._sample_scm(seq_len, num_features, self._n_outputs)

            # Get fixed node permutation
            with torch.no_grad():
                _, y_cal, _, fixed_perm = scm.forward_with_internals_fixed_mapping()

            # Calibrate class boundary
            with torch.no_grad():
                y_values = []
                x_values = []
                remaining = calibration_size
                while remaining > 0:
                    chunk = min(remaining, seq_len)
                    x_chunk, y_chunk, _, _ = scm.forward_with_internals_fixed_mapping(
                        fixed_perm
                    )
                    y_values.append(y_chunk.squeeze(-1) if y_chunk.dim() > 2 else y_chunk)
                    x_values.append(x_chunk)
                    remaining -= chunk
                y_all = torch.cat(y_values, dim=0)
                threshold = torch.median(y_all).item()

                x_cal = torch.cat(x_values, dim=0)
                all_x_cal.append(x_cal)

            self.scms.append(scm)
            self.fixed_perms.append(fixed_perm)
            self.class_assigners.append(FixedThresholdBinarize(threshold))

        # Compute shared normalization stats across all SCMs
        all_x = torch.cat(all_x_cal, dim=0)  # (total, 1, num_features)
        self._cached_mean = all_x.mean(dim=0, keepdim=True)  # (1, 1, num_features)
        self._cached_std = all_x.std(dim=0, keepdim=True).clamp(min=1e-6)

        print(f"  SCMFamilyDataLoader: {num_scms} SCMs initialized")

    def __iter__(self):
        import random as pyrandom

        for _ in range(self.num_steps):
            # Each batch element gets a randomly selected SCM
            all_x, all_y, all_target_y = [], [], []
            all_internals = [] if self.return_internals else None
            scm_indices = []

            for b in range(self.batch_size):
                scm_idx = pyrandom.randint(0, self.num_scms - 1)
                scm_indices.append(scm_idx)

                result = get_batch_fixed_scm(
                    batch_size=1,
                    seq_len=self.seq_len,
                    num_features=self.num_features,
                    hyperparameters=self.hyperparameters,
                    device=self.device,
                    single_eval_pos=self.single_eval_pos,
                    num_outputs=self._n_outputs,
                    scm=self.scms[scm_idx],
                    fixed_perm=self.fixed_perms[scm_idx],
                    class_assigner=self.class_assigners[scm_idx],
                    cached_norm_stats=(self._cached_mean, self._cached_std),
                    return_internals=self.return_internals,
                )
                if self.return_internals:
                    x, y, target_y, batch_int = result
                    all_internals.extend(batch_int)
                else:
                    x, y, target_y = result

                all_x.append(x)
                all_y.append(y)
                all_target_y.append(target_y)

            # Concatenate along batch dimension
            x = torch.cat(all_x, dim=1)  # (seq_len, batch_size, nf)
            y = torch.cat(all_y, dim=1)  # (seq_len, batch_size)
            target_y = torch.cat(all_target_y, dim=1)  # (seq_len, batch_size, channels)

            if self.return_internals:
                # Attach SCM index to each internals dict
                for i, internals in enumerate(all_internals):
                    internals["scm_idx"] = scm_indices[i]
                yield (None, x, y), target_y, self.single_eval_pos, all_internals
            else:
                yield (None, x, y), target_y, self.single_eval_pos

    def __len__(self):
        return self.num_steps


class DiverseSCMDataLoader:
    """Data loader that creates a new random SCM per batch element with diverse structure.

    Unlike the standard DataLoader which uses fixed SCM config, this varies:
    - num_layers: randomly sampled from a range
    - prior_mlp_hidden_dim: randomly sampled from a range
    - prior_mlp_activations: randomly chosen from a set

    Used for Experiment 4 (meta-learning across diverse causal structures).
    Each batch element gets a completely new SCM with potentially different
    architecture, so the model must do in-context causal inference.
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
        layer_range: tuple = (2, 5),
        hidden_dim_range: tuple = (8, 33),
        activations: list = None,
    ):
        from torch import nn

        self.num_features = num_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hyperparameters = hyperparameters or {}
        self.device = device
        self.single_eval_pos = single_eval_pos
        self._n_outputs = num_outputs
        self.num_steps = num_steps
        self.return_internals = False
        self.layer_range = layer_range
        self.hidden_dim_range = hidden_dim_range
        self.activations = activations or [nn.Tanh, nn.ReLU]

        print(f"  DiverseSCMDataLoader: layers={layer_range}, hidden={hidden_dim_range}, "
              f"activations={[a.__name__ for a in self.activations]}")

    def _random_hp(self):
        """Sample random SCM hyperparameters for one batch element."""
        import random as pyrandom

        hp = dict(self.hyperparameters)
        hp["num_layers"] = pyrandom.randint(self.layer_range[0], self.layer_range[1])
        hp["prior_mlp_hidden_dim"] = pyrandom.randint(
            self.hidden_dim_range[0], self.hidden_dim_range[1] - 1
        )
        hp["prior_mlp_activations"] = pyrandom.choice(self.activations)
        return hp

    def __iter__(self):
        for _ in range(self.num_steps):
            all_x, all_y, all_target_y = [], [], []
            all_internals = [] if self.return_internals else None

            for b in range(self.batch_size):
                hp = self._random_hp()

                if self.return_internals:
                    x, y, target_y, scm_data_list = get_batch_with_scm(
                        batch_size=1,
                        seq_len=self.seq_len,
                        num_features=self.num_features,
                        hyperparameters=hp,
                        device=self.device,
                        single_eval_pos=self.single_eval_pos,
                        num_outputs=self._n_outputs,
                    )
                    all_internals.extend(scm_data_list)
                else:
                    x, y, target_y = get_batch(
                        batch_size=1,
                        seq_len=self.seq_len,
                        num_features=self.num_features,
                        hyperparameters=hp,
                        device=self.device,
                        single_eval_pos=self.single_eval_pos,
                        num_outputs=self._n_outputs,
                    )

                all_x.append(x)
                all_y.append(y)
                all_target_y.append(target_y)

            x = torch.cat(all_x, dim=1)
            y = torch.cat(all_y, dim=1)
            target_y = torch.cat(all_target_y, dim=1)

            if self.return_internals:
                yield (None, x, y), target_y, self.single_eval_pos, all_internals
            else:
                yield (None, x, y), target_y, self.single_eval_pos

    def __len__(self):
        return self.num_steps
