"""
Counterfactual pair generation from TabPFN's SCM (MLP) prior.

Generates factual-counterfactual pairs by:
1. Sampling a random SCM (MLP with random weights/structure)
2. Running a factual forward pass, capturing all exogenous noise
3. Applying feature-level interventions (do-operator)
4. Re-propagating through the SCM with the same noise but intervened values
5. Pairing factual and counterfactual outputs

The causal downstream propagation is always enforced: when a feature at layer L
is intervened on, all features at deeper layers update causally, while features
at shallower layers remain unchanged.
"""

import random as pyrandom
import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Union

import torch
from torch import nn, Tensor
import numpy as np

from tabpfn.utils import default_device
from tabpfn.priors.mlp import GaussianNoise, causes_sampler_f
from tabpfn.priors.flexible_categorical import (
    BalancedBinarize,
    MulticlassRank,
    MulticlassValue,
    MulticlassMultiNode,
)


class PerturbationStrategy(Enum):
    """Available perturbation strategies for counterfactual generation."""

    ADDITIVE_NOISE = "additive_noise"
    MARGINAL_REPLACEMENT = "marginal_replacement"
    GRADIENT_GUIDED = "gradient_guided"
    FIXED_MAGNITUDE = "fixed_magnitude"
    UNIFORM_RANDOM = "uniform_random"


@dataclass
class CounterfactualBatch:
    """Output container for a batch of factual-counterfactual pairs.

    All tensors have shape (seq_len, batch_size, ...) following TabPFN convention.
    """

    x_factual: Tensor  # (seq_len, batch, num_features) - factual feature values
    y_factual: Tensor  # (seq_len, batch) - factual targets (continuous)
    y_factual_class: Tensor  # (seq_len, batch) - factual targets (classified)
    x_counterfactual: Tensor  # (seq_len, batch, num_features) - counterfactual features
    y_counterfactual: Tensor  # (seq_len, batch) - counterfactual targets (continuous)
    y_counterfactual_class: (
        Tensor  # (seq_len, batch) - counterfactual targets (classified)
    )
    label_flipped: Tensor  # (seq_len, batch) - boolean: did the class label change?
    intervention_mask: Tensor  # (seq_len, batch, num_features) - boolean: which features were intervened on
    perturbation_delta: (
        Tensor  # (seq_len, batch, num_features) - actual delta applied to features
    )


def get_default_counterfactual_config():
    """Returns default configuration for counterfactual pair generation."""
    return {
        # Perturbation settings
        "perturbation_strategy": "additive_noise",
        "perturbation_prob": 0.3,  # probability each feature is perturbed
        "perturbation_magnitude": 1.0,  # scale for additive noise / fixed magnitude
        "gradient_step_size": 0.1,  # step size for gradient-guided strategy
        "fixed_magnitude_k": 1.0,  # number of std devs for fixed_magnitude
        "num_counterfactuals_per_factual": 1,
        # Classification settings
        "num_classes": 2,
        "balanced": True,
        "multiclass_type": "rank",
        "output_multiclass_ordered_p": 0.0,
        # SCM settings (overrides for MLP prior)
        "force_is_causal": True,
        # MLP prior defaults (can be overridden)
        "num_layers": 4,
        "prior_mlp_hidden_dim": 64,
        "prior_mlp_dropout_prob": 0.3,
        "noise_std": 0.1,
        "init_std": 1.0,
        "num_causes": 5,
        "is_causal": True,
        "pre_sample_weights": True,
        "pre_sample_causes": True,
        "y_is_effect": True,
        "sampling": "normal",
        "prior_mlp_activations": torch.nn.Tanh,
        "block_wise_dropout": False,
        "sort_features": False,
        "in_clique": False,
        "random_feature_rotation": False,
        "prior_mlp_scale_weights_sqrt": True,
        "new_mlp_per_example": False,
        "mix_activations": False,
        "multiclass_type": "rank",
        "verbose": False,
    }


class CounterfactualSCMGenerator:
    """Generates factual-counterfactual pairs from TabPFN's SCM prior.

    Usage:
        config = get_default_counterfactual_config()
        gen = CounterfactualSCMGenerator(config, device='cpu')
        batch = gen.generate_batch(
            batch_size=8,
            seq_len=100,
            num_features=10,
        )
        # batch.x_factual, batch.x_counterfactual, batch.label_flipped, etc.
    """

    def __init__(self, config: Optional[Dict] = None, device: str = default_device):
        self.config = get_default_counterfactual_config()
        if config:
            self.config.update(config)
        self.device = device

        # Force causal mode if configured
        if self.config.get("force_is_causal", True):
            self.config["is_causal"] = True

    def generate_batch(
        self,
        batch_size: int,
        seq_len: int,
        num_features: int,
        num_outputs: int = 1,
        perturbation_strategy: Optional[Union[str, PerturbationStrategy]] = None,
    ) -> CounterfactualBatch:
        """Generate a batch of factual-counterfactual paired datasets.

        Each element in the batch uses a separately sampled SCM.

        Args:
            batch_size: number of datasets in the batch
            seq_len: number of samples per dataset
            num_features: number of observed features
            num_outputs: number of target dimensions (usually 1)
            perturbation_strategy: override the config strategy for this batch

        Returns:
            CounterfactualBatch with all factual and counterfactual data
        """
        if perturbation_strategy is not None:
            if isinstance(perturbation_strategy, PerturbationStrategy):
                perturbation_strategy = perturbation_strategy.value
        else:
            perturbation_strategy = self.config["perturbation_strategy"]

        # Build shared class assigner for consistent boundaries
        class_assigner = self._build_class_assigner()

        all_x_f, all_y_f, all_x_cf, all_y_cf = [], [], [], []
        all_intervention_masks, all_deltas = [], []

        for _ in range(batch_size):
            scm = self._sample_scm(seq_len, num_features, num_outputs)
            x_f, y_f, internals = scm.forward_with_internals()

            # Determine which features to perturb (random subset)
            perturb_mask = self._select_perturbation_targets(
                seq_len, num_features
            )  # (seq_len, num_features) boolean

            # Compute intervention values
            interventions, deltas = self._compute_interventions(
                x_f, perturb_mask, internals, perturbation_strategy, scm
            )

            # Re-propagate with interventions
            if interventions:
                x_cf, y_cf = scm.forward_with_intervention(internals, interventions)
            else:
                # No interventions applied — counterfactual equals factual
                x_cf, y_cf = x_f.clone(), y_f.clone()

            all_x_f.append(x_f)
            all_y_f.append(y_f)
            all_x_cf.append(x_cf)
            all_y_cf.append(y_cf)
            all_intervention_masks.append(
                perturb_mask.unsqueeze(1)
            )  # (seq_len, 1, num_features)
            all_deltas.append(deltas.unsqueeze(1))  # (seq_len, 1, num_features)

        # Concatenate along batch dimension
        x_factual = torch.cat(all_x_f, dim=1).detach()  # (seq_len, batch, num_features)
        y_factual = torch.cat(all_y_f, dim=1).detach()  # (seq_len, batch, num_outputs)
        x_cf = torch.cat(all_x_cf, dim=1).detach()  # (seq_len, batch, num_features)
        y_cf = torch.cat(all_y_cf, dim=1).detach()  # (seq_len, batch, num_outputs)
        intervention_mask = torch.cat(all_intervention_masks, dim=1).detach()
        perturbation_delta = torch.cat(all_deltas, dim=1).detach()

        # Squeeze target dim if single output
        if num_outputs == 1:
            y_factual = y_factual.squeeze(-1)
            y_cf = y_cf.squeeze(-1)

        # Apply shared classification
        y_factual_class = class_assigner(
            y_factual.unsqueeze(-1) if y_factual.dim() == 2 else y_factual
        ).float()
        y_cf_class = class_assigner(
            y_cf.unsqueeze(-1) if y_cf.dim() == 2 else y_cf
        ).float()

        # Squeeze back if needed
        if y_factual_class.dim() > 2:
            y_factual_class = y_factual_class.squeeze(-1)
        if y_cf_class.dim() > 2:
            y_cf_class = y_cf_class.squeeze(-1)

        label_flipped = y_factual_class != y_cf_class

        return CounterfactualBatch(
            x_factual=x_factual,
            y_factual=y_factual,
            y_factual_class=y_factual_class,
            x_counterfactual=x_cf,
            y_counterfactual=y_cf,
            y_counterfactual_class=y_cf_class,
            label_flipped=label_flipped,
            intervention_mask=intervention_mask,
            perturbation_delta=perturbation_delta,
        )

    def _build_class_assigner(self) -> nn.Module:
        """Build a class assigner that will be shared between factual and counterfactual."""
        num_classes = self.config["num_classes"]
        balanced = self.config["balanced"]
        multiclass_type = self.config["multiclass_type"]
        ordered_p = self.config.get("output_multiclass_ordered_p", 0.0)

        if num_classes == 0:
            # Regression mode: identity
            return nn.Identity()
        elif num_classes == 2 and balanced:
            return BalancedBinarize()
        elif num_classes > 2 and balanced:
            raise NotImplementedError("Balanced multiclass not supported")
        else:
            if multiclass_type == "rank":
                return MulticlassRank(num_classes, ordered_p=ordered_p)
            elif multiclass_type == "value":
                return MulticlassValue(num_classes, ordered_p=ordered_p)
            elif multiclass_type == "multi_node":
                return MulticlassMultiNode(num_classes, ordered_p=ordered_p)
            else:
                raise ValueError(f"Unknown multiclass_type: {multiclass_type}")

    def _sample_scm(self, seq_len: int, num_features: int, num_outputs: int):
        """Sample a random SCM by instantiating an MLP with random structure/weights.

        We import the MLP class definition from mlp.py's get_batch, but instantiate
        it directly with our config to get access to internal methods.
        """
        from tabpfn.priors.mlp import (
            get_batch as _get_batch_unused,
        )  # ensure module loaded

        hyperparameters = dict(self.config)

        # Sample activation if it's a class (not instance)
        act = hyperparameters.get("prior_mlp_activations", torch.nn.Tanh)
        if not callable(act) or isinstance(act, type):
            hyperparameters["prior_mlp_activations"] = lambda: act()
        elif not isinstance(act(), nn.Module):
            pass  # already a factory

        # Import the MLP class by creating it through the module's mechanism
        # We need to reconstruct the MLP class with access to seq_len, num_features, etc.
        # The MLP class is defined inside get_batch as a closure — we recreate it here.
        return self._build_mlp(hyperparameters, seq_len, num_features, num_outputs)

    def _build_mlp(self, hyperparameters, seq_len, num_features, num_outputs):
        """Build an MLP instance with the counterfactual-aware methods.

        This recreates the MLP class from mlp.py but with access to
        forward_with_internals and forward_with_intervention.
        """
        device = self.device

        # The MLP class in mlp.py is a closure over seq_len, num_features, etc.
        # We need to import and instantiate it properly.
        # Since it's defined inside get_batch(), we replicate the construction here
        # using the same logic but calling it as a standalone class.

        from tabpfn.priors import mlp as mlp_module

        # Temporarily override the activation to be a consistent factory
        if not (hyperparameters.get("mix_activations", False)):
            act_cls = hyperparameters.get("prior_mlp_activations", torch.nn.Tanh)
            if isinstance(act_cls, type):
                s = act_cls()
                hyperparameters["prior_mlp_activations"] = lambda: s
            elif callable(act_cls):
                s = act_cls()
                if isinstance(s, nn.Module):
                    hyperparameters["prior_mlp_activations"] = lambda: s
                else:
                    # act_cls is already a factory returning modules
                    pass

        # The MLP class is defined as an inner class of get_batch.
        # We call get_batch's MLP class construction by leveraging the module.
        # To avoid code duplication, we use a trick: call get_batch with batch_size=1
        # but we need the model object. However, get_batch doesn't expose it.
        # Instead, we must replicate the MLP construction.

        class _MLP(torch.nn.Module):
            """Replica of mlp.py's MLP class with counterfactual support."""

            def __init__(self, hp):
                super().__init__()

                with torch.no_grad():
                    for key in hp:
                        setattr(self, key, hp[key])

                    assert self.num_layers >= 2

                    if self.is_causal:
                        self.prior_mlp_hidden_dim = max(
                            self.prior_mlp_hidden_dim, num_outputs + 2 * num_features
                        )
                    else:
                        self.num_causes = num_features

                    if self.pre_sample_causes:
                        self.causes_mean, self.causes_std = causes_sampler_f(
                            self.num_causes
                        )
                        self.causes_mean = (
                            torch.tensor(self.causes_mean, device=device)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .tile((seq_len, 1, 1))
                        )
                        self.causes_std = (
                            torch.tensor(self.causes_std, device=device)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .tile((seq_len, 1, 1))
                        )

                    def generate_module(layer_idx, out_dim):
                        noise = (
                            GaussianNoise(
                                torch.abs(
                                    torch.normal(
                                        torch.zeros(size=(1, out_dim), device=device),
                                        float(self.noise_std),
                                    )
                                ),
                                device=device,
                            )
                            if self.pre_sample_weights
                            else GaussianNoise(float(self.noise_std), device=device)
                        )
                        return [
                            nn.Sequential(
                                *[
                                    self.prior_mlp_activations(),
                                    nn.Linear(
                                        self.prior_mlp_hidden_dim,
                                        out_dim,
                                        device=device,
                                    ),
                                    noise,
                                ]
                            )
                        ]

                    self.layers = [
                        nn.Linear(
                            self.num_causes, self.prior_mlp_hidden_dim, device=device
                        )
                    ]
                    self.layers += [
                        module
                        for layer_idx in range(self.num_layers - 1)
                        for module in generate_module(
                            layer_idx, self.prior_mlp_hidden_dim
                        )
                    ]
                    if not self.is_causal:
                        self.layers += generate_module(-1, num_outputs)
                    self.layers = nn.Sequential(*self.layers)

                    # Initialize parameters (same as mlp.py)
                    for i, (n, p) in enumerate(self.layers.named_parameters()):
                        if self.block_wise_dropout:
                            if len(p.shape) == 2:
                                nn.init.zeros_(p)
                                n_blocks = pyrandom.randint(
                                    1, math.ceil(math.sqrt(min(p.shape[0], p.shape[1])))
                                )
                                w, h = p.shape[0] // n_blocks, p.shape[1] // n_blocks
                                keep_prob = (n_blocks * w * h) / p.numel()
                                for block in range(0, n_blocks):
                                    nn.init.normal_(
                                        p[
                                            w * block : w * (block + 1),
                                            h * block : h * (block + 1),
                                        ],
                                        std=self.init_std
                                        / keep_prob
                                        ** (
                                            1 / 2
                                            if self.prior_mlp_scale_weights_sqrt
                                            else 1
                                        ),
                                    )
                        else:
                            if len(p.shape) == 2:
                                dropout_prob = (
                                    self.prior_mlp_dropout_prob if i > 0 else 0.0
                                )
                                dropout_prob = min(dropout_prob, 0.99)
                                nn.init.normal_(
                                    p,
                                    std=self.init_std
                                    / (
                                        1.0
                                        - dropout_prob
                                        ** (
                                            1 / 2
                                            if self.prior_mlp_scale_weights_sqrt
                                            else 1
                                        )
                                    ),
                                )
                                p *= torch.bernoulli(
                                    torch.zeros_like(p) + 1.0 - dropout_prob
                                )

            def forward(self):
                """Standard forward pass (same as mlp.py)."""
                x, y, _ = self.forward_with_internals()
                return x, y

            def forward_with_internals(self):
                """Forward pass returning all SCM internals for counterfactual generation."""

                # --- Sample root causes ---
                def sample_normal():
                    if self.pre_sample_causes:
                        return torch.normal(
                            self.causes_mean, self.causes_std.abs()
                        ).float()
                    else:
                        return torch.normal(
                            0.0, 1.0, (seq_len, 1, self.num_causes), device=device
                        ).float()

                if self.sampling == "normal":
                    causes = sample_normal()
                elif self.sampling == "mixed":
                    zipf_p = pyrandom.random() * 0.66
                    multi_p = pyrandom.random() * 0.66
                    normal_p = pyrandom.random() * 0.66

                    def sample_cause(n):
                        if pyrandom.random() > normal_p:
                            if self.pre_sample_causes:
                                return torch.normal(
                                    self.causes_mean[:, :, n],
                                    self.causes_std[:, :, n].abs(),
                                ).float()
                            else:
                                return torch.normal(
                                    0.0, 1.0, (seq_len, 1), device=device
                                ).float()
                        elif pyrandom.random() > multi_p:
                            x = (
                                torch.multinomial(
                                    torch.rand((pyrandom.randint(2, 10))),
                                    seq_len,
                                    replacement=True,
                                )
                                .to(device)
                                .unsqueeze(-1)
                                .float()
                            )
                            x = (x - torch.mean(x)) / torch.std(x)
                            return x
                        else:
                            x = torch.minimum(
                                torch.tensor(
                                    np.random.zipf(
                                        2.0 + pyrandom.random() * 2, size=(seq_len)
                                    ),
                                    device=device,
                                )
                                .unsqueeze(-1)
                                .float(),
                                torch.tensor(10.0, device=device),
                            )
                            return x - torch.mean(x)

                    causes = torch.cat(
                        [sample_cause(n).unsqueeze(-1) for n in range(self.num_causes)],
                        -1,
                    )
                elif self.sampling == "uniform":
                    causes = torch.rand((seq_len, 1, self.num_causes), device=device)
                else:
                    raise ValueError(f"Invalid sampling: {self.sampling}")

                # --- Forward propagation with noise capture ---
                layer_noises = []
                outputs = [causes]

                for layer in self.layers:
                    layer_input = outputs[-1]
                    if isinstance(layer, nn.Sequential):
                        pre_noise_out = layer_input
                        for sublayer in layer:
                            if isinstance(sublayer, GaussianNoise):
                                noise_tensor = sublayer.sample_noise(pre_noise_out)
                                layer_noises.append(noise_tensor)
                                pre_noise_out = pre_noise_out + noise_tensor
                            else:
                                pre_noise_out = sublayer(pre_noise_out)
                        outputs.append(pre_noise_out)
                    else:
                        outputs.append(layer(layer_input))
                        layer_noises.append(None)

                outputs_for_selection = outputs[2:]  # skip causes + first hidden

                if self.is_causal:
                    outputs_flat = torch.cat(outputs_for_selection, -1)

                    # Build layer boundaries
                    layer_boundaries = []
                    offset = 0
                    for i, out in enumerate(outputs_for_selection):
                        dim = out.shape[-1]
                        layer_boundaries.append((offset, offset + dim, i))
                        offset += dim

                    if self.in_clique:
                        random_perm = pyrandom.randint(
                            0, outputs_flat.shape[-1] - num_outputs - num_features
                        ) + torch.randperm(num_outputs + num_features, device=device)
                    else:
                        random_perm = torch.randperm(
                            outputs_flat.shape[-1] - 1, device=device
                        )

                    random_idx_y = (
                        list(range(-num_outputs, 0))
                        if self.y_is_effect
                        else random_perm[0:num_outputs]
                    )
                    random_idx = random_perm[num_outputs : num_outputs + num_features]

                    if self.sort_features:
                        random_idx, _ = torch.sort(random_idx)

                    y = outputs_flat[:, :, random_idx_y]
                    x = outputs_flat[:, :, random_idx]
                else:
                    y = outputs_for_selection[-1][:, :, :]
                    x = causes
                    outputs_flat = (
                        torch.cat(outputs_for_selection, -1)
                        if outputs_for_selection
                        else causes
                    )
                    random_idx = torch.arange(num_features, device=device)
                    random_idx_y = list(range(-num_outputs, 0))
                    layer_boundaries = []
                    offset = 0
                    for i, out in enumerate(outputs_for_selection):
                        dim = out.shape[-1]
                        layer_boundaries.append((offset, offset + dim, i))
                        offset += dim

                internals = {
                    "cause_noise": causes,
                    "layer_noises": layer_noises,
                    "layer_outputs": outputs,
                    "outputs_flat": outputs_flat,
                    "node_mapping": {
                        "feature_indices": random_idx,
                        "target_indices": random_idx_y,
                    },
                    "layer_boundaries": layer_boundaries,
                }

                return x, y, internals

            def forward_with_intervention(self, internals, interventions):
                """Re-propagate through SCM with fixed noise and feature-level interventions.

                Args:
                    internals: dict from forward_with_internals()
                    interventions: dict mapping outputs_flat_index -> value tensor (seq_len, 1)

                Returns:
                    x_cf, y_cf: counterfactual features and target
                """
                causes = internals["cause_noise"]
                layer_noises = internals["layer_noises"]
                layer_boundaries = internals["layer_boundaries"]
                node_mapping = internals["node_mapping"]

                # Map flat indices to per-layer interventions
                intervention_by_layer = {}
                for flat_idx, value in interventions.items():
                    for start, end, layer_idx in layer_boundaries:
                        if start <= flat_idx < end:
                            if layer_idx not in intervention_by_layer:
                                intervention_by_layer[layer_idx] = {}
                            intervention_by_layer[layer_idx][flat_idx - start] = value
                            break

                # Re-propagate layer by layer with fixed noise
                outputs = [causes]
                noise_idx = 0

                for i, layer in enumerate(self.layers):
                    layer_input = outputs[-1]
                    if isinstance(layer, nn.Sequential):
                        pre_noise_out = layer_input
                        for sublayer in layer:
                            if isinstance(sublayer, GaussianNoise):
                                pre_noise_out = sublayer(
                                    pre_noise_out,
                                    pre_sampled_noise=layer_noises[noise_idx],
                                )
                            else:
                                pre_noise_out = sublayer(pre_noise_out)
                        outputs.append(pre_noise_out)
                        noise_idx += 1
                    else:
                        outputs.append(layer(layer_input))
                        noise_idx += 1

                    # Apply interventions at correct layer depth
                    selection_layer_idx = len(outputs) - 1 - 2
                    if selection_layer_idx in intervention_by_layer:
                        for within_idx, value in intervention_by_layer[
                            selection_layer_idx
                        ].items():
                            outputs[-1] = outputs[-1].clone()
                            outputs[-1][:, :, within_idx] = value

                outputs_for_selection = outputs[2:]

                if self.is_causal:
                    outputs_flat_cf = torch.cat(outputs_for_selection, -1)
                    feature_indices = node_mapping["feature_indices"]
                    target_indices = node_mapping["target_indices"]
                    y_cf = outputs_flat_cf[:, :, target_indices]
                    x_cf = outputs_flat_cf[:, :, feature_indices]
                else:
                    y_cf = outputs_for_selection[-1][:, :, :]
                    x_cf = causes

                return x_cf, y_cf

        model = _MLP(hyperparameters).to(device)
        return model

    def _select_perturbation_targets(self, seq_len: int, num_features: int) -> Tensor:
        """Select which features to perturb for each sample (random subset).

        Returns:
            Boolean tensor of shape (seq_len, num_features)
        """
        perturb_prob = self.config["perturbation_prob"]
        mask = torch.rand((seq_len, num_features), device=self.device) < perturb_prob
        # Ensure at least one feature is perturbed per sample
        no_perturbation = ~mask.any(dim=1)
        if no_perturbation.any():
            random_features = torch.randint(
                0, num_features, (no_perturbation.sum(),), device=self.device
            )
            mask[no_perturbation, random_features] = True
        return mask

    def _compute_interventions(
        self,
        x_factual: Tensor,
        perturb_mask: Tensor,
        internals: Dict,
        strategy: str,
        scm,
    ) -> tuple:
        """Compute intervention values for perturbed features.

        Args:
            x_factual: (seq_len, 1, num_features) factual feature values
            perturb_mask: (seq_len, num_features) boolean mask
            internals: SCM internals from forward_with_internals
            strategy: perturbation strategy name
            scm: the MLP model (needed for gradient-guided)

        Returns:
            interventions: dict mapping outputs_flat_index -> value tensor
            deltas: (seq_len, num_features) tensor of perturbation deltas
        """
        seq_len = x_factual.shape[0]
        num_features = x_factual.shape[2]
        feature_indices = internals["node_mapping"]["feature_indices"]
        outputs_flat = internals["outputs_flat"]

        deltas = torch.zeros((seq_len, num_features), device=self.device)
        interventions = {}

        strategy_fn = {
            "additive_noise": self._perturb_additive_noise,
            "marginal_replacement": self._perturb_marginal_replacement,
            "gradient_guided": self._perturb_gradient_guided,
            "fixed_magnitude": self._perturb_fixed_magnitude,
            "uniform_random": self._perturb_uniform_random,
        }[strategy]

        delta_per_feature = strategy_fn(x_factual, perturb_mask, internals, scm)

        # Build interventions dict: for each feature that has any perturbation,
        # set the intervention at the corresponding outputs_flat index
        for feat_idx in range(num_features):
            feat_mask = perturb_mask[:, feat_idx]  # (seq_len,)
            if not feat_mask.any():
                continue

            flat_idx = feature_indices[feat_idx].item()
            # Get the current value at this node and add the delta
            current_val = outputs_flat[:, :, flat_idx]  # (seq_len, 1)
            new_val = current_val.clone()
            new_val[feat_mask, :] = current_val[feat_mask, :] + delta_per_feature[
                feat_mask, feat_idx
            ].unsqueeze(-1)

            interventions[flat_idx] = new_val
            deltas[:, feat_idx] = delta_per_feature[:, feat_idx]

        return interventions, deltas

    # -------------------------------------------------------------------------
    # Perturbation strategy implementations
    # -------------------------------------------------------------------------

    def _perturb_additive_noise(
        self, x_factual: Tensor, perturb_mask: Tensor, internals: Dict, scm
    ) -> Tensor:
        """Add Gaussian noise scaled by feature std.

        Returns delta tensor (seq_len, num_features).
        """
        magnitude = self.config["perturbation_magnitude"]
        # Compute per-feature std across the sequence
        feat_std = x_factual[:, 0, :].std(dim=0, keepdim=True)  # (1, num_features)
        feat_std = feat_std.clamp(min=1e-6)

        noise = torch.randn(
            (x_factual.shape[0], x_factual.shape[2]), device=self.device
        )
        delta = noise * feat_std * magnitude
        # Zero out non-perturbed features
        delta[~perturb_mask] = 0.0
        return delta

    def _perturb_marginal_replacement(
        self, x_factual: Tensor, perturb_mask: Tensor, internals: Dict, scm
    ) -> Tensor:
        """Replace feature with value from another sample (marginal distribution).

        Returns delta tensor (seq_len, num_features).
        """
        seq_len = x_factual.shape[0]
        num_features = x_factual.shape[2]

        # For each sample, pick a random other sample's feature value
        random_indices = torch.randint(0, seq_len, (seq_len,), device=self.device)
        replacement_values = x_factual[random_indices, 0, :]  # (seq_len, num_features)
        original_values = x_factual[:, 0, :]  # (seq_len, num_features)

        delta = replacement_values - original_values
        delta[~perturb_mask] = 0.0
        return delta

    def _perturb_gradient_guided(
        self, x_factual: Tensor, perturb_mask: Tensor, internals: Dict, scm
    ) -> Tensor:
        """Use numerical gradient of target w.r.t. features to find effective perturbations.

        Estimates the gradient via finite differences at each feature node,
        then steps in the direction that changes y most.
        Returns delta tensor (seq_len, num_features).
        """
        step_size = self.config["gradient_step_size"]
        magnitude = self.config["perturbation_magnitude"]
        feature_indices = internals["node_mapping"]["feature_indices"]
        target_indices = internals["node_mapping"]["target_indices"]
        layer_noises = internals["layer_noises"]
        causes = internals["cause_noise"]
        node_mapping = internals["node_mapping"]
        layer_boundaries = internals["layer_boundaries"]

        eps = 1e-3  # finite difference step
        num_features = x_factual.shape[2]
        seq_len = x_factual.shape[0]

        # Get baseline y
        outputs_flat = internals["outputs_flat"]
        y_base = outputs_flat[:, :, target_indices].squeeze(-1)  # (seq_len, 1)

        # Estimate gradient for each feature via finite differences
        feature_grads = torch.zeros((seq_len, num_features), device=self.device)
        for feat_i in range(num_features):
            flat_idx = feature_indices[feat_i].item()
            # Create a small intervention: +eps at this feature
            intervention = {flat_idx: outputs_flat[:, :, flat_idx] + eps}
            _, y_perturbed = scm.forward_with_intervention(internals, intervention)
            y_perturbed = y_perturbed.squeeze(-1)  # (seq_len, 1)
            feature_grads[:, feat_i] = ((y_perturbed - y_base) / eps).squeeze(-1)

        # Step in the gradient direction (sign * magnitude * feature_std)
        feat_std = x_factual[:, 0, :].std(dim=0, keepdim=True).clamp(min=1e-6)
        delta = torch.sign(feature_grads) * step_size * feat_std * magnitude

        # Where gradient is zero, use random direction instead
        zero_grad = feature_grads == 0
        random_dir = (torch.rand_like(delta) > 0.5).float() * 2 - 1
        delta[zero_grad] = (
            random_dir[zero_grad]
            * step_size
            * feat_std.expand_as(delta)[zero_grad]
            * magnitude
        )

        # Randomly flip direction for diversity (50% chance)
        random_flip = (
            torch.rand((seq_len, num_features), device=self.device) > 0.5
        ).float() * 2 - 1
        delta = delta * random_flip

        delta[~perturb_mask] = 0.0
        return delta

    def _perturb_fixed_magnitude(
        self, x_factual: Tensor, perturb_mask: Tensor, internals: Dict, scm
    ) -> Tensor:
        """Shift features by k standard deviations in a random direction.

        Returns delta tensor (seq_len, num_features).
        """
        k = self.config["fixed_magnitude_k"]
        feat_std = x_factual[:, 0, :].std(dim=0, keepdim=True).clamp(min=1e-6)

        # Random direction: +1 or -1
        direction = (
            torch.rand((x_factual.shape[0], x_factual.shape[2]), device=self.device)
            > 0.5
        ).float() * 2 - 1
        delta = direction * k * feat_std
        delta[~perturb_mask] = 0.0
        return delta

    def _perturb_uniform_random(
        self, x_factual: Tensor, perturb_mask: Tensor, internals: Dict, scm
    ) -> Tensor:
        """Replace feature with uniform sample from [min, max] of that feature.

        Returns delta tensor (seq_len, num_features).
        """
        feat_vals = x_factual[:, 0, :]  # (seq_len, num_features)
        feat_min = feat_vals.min(dim=0, keepdim=True).values
        feat_max = feat_vals.max(dim=0, keepdim=True).values

        # Sample uniformly in [min, max]
        uniform_vals = torch.rand_like(feat_vals) * (feat_max - feat_min) + feat_min
        delta = uniform_vals - feat_vals
        delta[~perturb_mask] = 0.0
        return delta
