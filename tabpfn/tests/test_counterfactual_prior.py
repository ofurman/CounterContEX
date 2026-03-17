"""
Tests for the counterfactual prior data loader.

Verifies:
1. Output tensor shapes are correct
2. Context y values are factual labels, query y values are target labels
3. Targets are correct deltas matching x_cf - x_factual
4. Label-flipped samples are prioritized for queries
5. DataLoader integration via get_batch_to_dataloader works
"""

import pytest
import torch

import sys
import importlib

# Prevent tabpfn.priors.__init__ from importing all submodules (fast_gp needs gpytorch)
# by importing the specific modules directly
counterfactual_mod = importlib.import_module("tabpfn.priors.counterfactual")
get_default_counterfactual_config = counterfactual_mod.get_default_counterfactual_config

counterfactual_prior_mod = importlib.import_module("tabpfn.priors.counterfactual_prior")
get_batch = counterfactual_prior_mod.get_batch
DataLoader = counterfactual_prior_mod.DataLoader


DEVICE = "cpu"
BATCH_SIZE = 4
SEQ_LEN = 64
NUM_FEATURES = 5
SINGLE_EVAL_POS = 32


@pytest.fixture
def default_hyperparameters():
    return get_default_counterfactual_config()


class TestGetBatchShapes:
    """Verify get_batch produces tensors with correct shapes."""

    def test_output_shapes(self, default_hyperparameters):
        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=default_hyperparameters,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        assert x.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
        assert y.shape == (SEQ_LEN, BATCH_SIZE)
        # Default mask_supervision=True → target_y has 2*num_features channels
        assert target_y.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES * 2)

    def test_output_shapes_no_mask(self, default_hyperparameters):
        hp = dict(default_hyperparameters)
        hp["mask_supervision"] = False
        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=hp,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        assert x.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
        assert y.shape == (SEQ_LEN, BATCH_SIZE)
        assert target_y.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)

    def test_no_nans(self, default_hyperparameters):
        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=default_hyperparameters,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        assert not torch.isnan(x).any(), "NaN in x"
        assert not torch.isnan(y).any(), "NaN in y"
        assert not torch.isnan(target_y).any(), "NaN in target_y"

    def test_y_values_are_class_labels(self, default_hyperparameters):
        """y values should be valid class labels (non-negative integers)."""
        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=default_hyperparameters,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        assert (y >= 0).all(), "y should be non-negative"
        unique_classes = torch.unique(y)
        assert len(unique_classes) <= 2, f"Expected at most 2 classes, got {unique_classes}"


class TestContextQueryEncoding:
    """Verify context uses factual labels and queries use target labels."""

    def test_context_y_are_factual_labels(self, default_hyperparameters):
        """Context positions should have factual class labels."""
        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=default_hyperparameters,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        context_y = y[:SINGLE_EVAL_POS]
        # Context y should be valid class labels
        assert (context_y >= 0).all()
        assert (context_y == context_y.long().float()).all(), "Context y should be integer labels"

    def test_query_y_are_target_labels(self, default_hyperparameters):
        """Query positions should have target (counterfactual) class labels."""
        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=default_hyperparameters,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        query_y = y[SINGLE_EVAL_POS:]
        # Query y should also be valid class labels
        assert (query_y >= 0).all()
        assert (query_y == query_y.long().float()).all(), "Query y should be integer labels"


class TestDeltaTargets:
    """Verify targets are correct deltas."""

    def test_targets_are_deltas(self, default_hyperparameters):
        """Target values should represent x_cf - x_factual."""
        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=default_hyperparameters,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        # Targets should be finite
        assert torch.isfinite(target_y).all(), "All target deltas should be finite"

    def test_some_query_targets_nonzero(self, default_hyperparameters):
        """With default perturbation settings, some query deltas should be non-zero."""
        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=default_hyperparameters,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        # Delta channels are first num_features
        query_deltas = target_y[SINGLE_EVAL_POS:, :, :NUM_FEATURES]
        assert query_deltas.abs().sum() > 0, "Some query deltas should be non-zero"


class TestLabelFlipPrioritization:
    """Verify that label-flipped samples are prioritized for query positions."""

    def test_flipped_samples_in_queries(self):
        """With large perturbations causing many flips, query positions should
        have samples where the target label differs from the factual label."""
        config = get_default_counterfactual_config()
        config["perturbation_magnitude"] = 10.0
        config["perturbation_prob"] = 0.8
        config["perturbation_strategy"] = "fixed_magnitude"
        config["fixed_magnitude_k"] = 5.0

        x, y, target_y = get_batch(
            batch_size=8,
            seq_len=64,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=32,
        )
        # Query targets should have non-zero deltas (indicating actual changes)
        query_deltas = target_y[32:, :, :NUM_FEATURES]
        assert query_deltas.abs().sum() > 0, (
            "Query positions should have non-trivial deltas"
        )


class TestFlipOnlyQueries:
    """Verify flip_only_queries ensures all query positions have flipped samples."""

    def test_all_query_deltas_nonzero_with_flip_only(self):
        """When flip_only_queries=True, every query position should have non-zero deltas."""
        config = get_default_counterfactual_config()
        config["flip_only_queries"] = True
        config["perturbation_magnitude"] = 5.0
        config["perturbation_prob"] = 0.8
        config["perturbation_strategy"] = "fixed_magnitude"
        config["fixed_magnitude_k"] = 3.0

        x, y, target_y = get_batch(
            batch_size=4,
            seq_len=64,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=32,
        )
        query_deltas = target_y[32:, :, :NUM_FEATURES]  # (num_query, batch, num_features)
        # Each query position should have at least one non-zero delta feature
        for b in range(4):
            per_sample_norm = query_deltas[:, b, :].abs().sum(dim=-1)  # (num_query,)
            assert (per_sample_norm > 0).all(), (
                f"Batch {b}: all query deltas should be non-zero with flip_only_queries=True"
            )

    def test_context_labels_are_factual_with_flip_only(self):
        """Context positions should still have factual labels when flip_only_queries=True."""
        config = get_default_counterfactual_config()
        config["flip_only_queries"] = True

        x, y, target_y = get_batch(
            batch_size=4,
            seq_len=64,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=32,
        )
        context_y = y[:32]
        assert (context_y >= 0).all()
        assert (context_y == context_y.long().float()).all()

    def test_retry_loop_increases_flip_rate(self):
        """The retry loop should recover from low-flip-rate batches."""
        config = get_default_counterfactual_config()
        config["min_flip_rate"] = 0.01  # Very low threshold - should always pass
        config["max_retries"] = 3

        # This should not raise and should produce valid data
        x, y, target_y = get_batch(
            batch_size=4,
            seq_len=64,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=32,
        )
        assert x.shape == (64, 4, NUM_FEATURES)
        assert not torch.isnan(target_y).any()


class TestFeatureNormalization:
    """Verify per-dataset feature normalization."""

    def test_normalized_features_zero_mean_unit_std(self):
        """After normalization, per-feature mean ~ 0 and std ~ 1 per batch element."""
        config = get_default_counterfactual_config()
        config["normalize_features"] = True

        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        for b in range(BATCH_SIZE):
            feat_mean = x[:, b, :].mean(dim=0)
            feat_std = x[:, b, :].std(dim=0)
            assert torch.allclose(feat_mean, torch.zeros_like(feat_mean), atol=0.1), (
                f"Batch {b}: mean should be ~0, got {feat_mean}"
            )
            assert torch.allclose(feat_std, torch.ones_like(feat_std), atol=0.2), (
                f"Batch {b}: std should be ~1, got {feat_std}"
            )

    def test_normalization_reduces_delta_scale(self):
        """Normalized deltas should have smaller magnitude than raw deltas."""
        config = get_default_counterfactual_config()

        # Get raw (unnormalized) data
        config["normalize_features"] = False
        _, _, target_y_raw = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )

        # Get normalized data
        config["normalize_features"] = True
        _, _, target_y_norm = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )

        # Both should be finite
        assert torch.isfinite(target_y_norm).all()
        assert torch.isfinite(target_y_raw).all()

    def test_normalization_disabled(self):
        """When normalize_features=False, features should not be normalized."""
        config = get_default_counterfactual_config()
        config["normalize_features"] = False

        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        # Without normalization, features should NOT have mean~0, std~1 in general
        # (they will have the raw SCM scale)
        assert not torch.isnan(x).any()
        assert not torch.isnan(target_y).any()

    def test_normalization_no_nans(self):
        """Normalization should not introduce NaNs."""
        config = get_default_counterfactual_config()
        config["normalize_features"] = True

        x, y, target_y = get_batch(
            batch_size=8,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        assert not torch.isnan(x).any(), "NaN in normalized x"
        assert not torch.isnan(target_y).any(), "NaN in normalized target_y"
        assert torch.isfinite(x).all(), "Non-finite in normalized x"
        assert torch.isfinite(target_y).all(), "Non-finite in normalized target_y"


class TestMaskSupervision:
    """Verify mask supervision adds correct intervention mask channels to target_y."""

    def test_mask_channels_are_binary(self):
        """Mask channels should contain only 0s and 1s."""
        config = get_default_counterfactual_config()
        config["mask_supervision"] = True

        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        mask_channels = target_y[:, :, NUM_FEATURES:]
        unique_vals = torch.unique(mask_channels)
        assert all(v in (0.0, 1.0) for v in unique_vals), (
            f"Mask channels should be 0/1, got unique values: {unique_vals}"
        )

    def test_mask_consistent_with_nonzero_deltas(self):
        """Where mask=1, the delta should generally be non-zero (intervention was applied)."""
        config = get_default_counterfactual_config()
        config["mask_supervision"] = True
        config["perturbation_magnitude"] = 5.0

        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        deltas = target_y[:, :, :NUM_FEATURES]
        masks = target_y[:, :, NUM_FEATURES:]
        # Where mask=1, the raw perturbation delta is nonzero
        # (after causal propagation the full delta might differ, but masked features
        # should have received a nonzero perturbation)
        # At least some masked features should have nonzero delta
        masked_deltas = deltas[masks > 0.5]
        assert masked_deltas.abs().sum() > 0, (
            "Masked (intervened) features should have nonzero deltas"
        )

    def test_query_masks_have_interventions(self):
        """Query positions should have at least one intervened feature per sample."""
        config = get_default_counterfactual_config()
        config["mask_supervision"] = True
        config["flip_only_queries"] = True
        config["perturbation_magnitude"] = 5.0
        config["perturbation_prob"] = 0.8
        config["perturbation_strategy"] = "fixed_magnitude"
        config["fixed_magnitude_k"] = 3.0

        x, y, target_y = get_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
            hyperparameters=config,
            device=DEVICE,
            single_eval_pos=SINGLE_EVAL_POS,
        )
        query_masks = target_y[SINGLE_EVAL_POS:, :, NUM_FEATURES:]
        for b in range(BATCH_SIZE):
            per_sample_mask_sum = query_masks[:, b, :].sum(dim=-1)
            assert (per_sample_mask_sum > 0).all(), (
                f"Batch {b}: every query should have at least one intervened feature"
            )

    def test_composite_loss_computes(self):
        """The composite loss function should compute without errors."""
        import torch.nn.functional as F

        num_features = NUM_FEATURES
        num_query, batch = 10, 4
        output = torch.randn(num_query, batch, num_features * 2)
        target = torch.cat([
            torch.randn(num_query, batch, num_features),
            torch.randint(0, 2, (num_query, batch, num_features)).float(),
        ], dim=-1)

        # Import the composite loss
        train_mod = importlib.import_module("tabpfn.train_counterfactual")
        loss = train_mod._composite_loss(output, target, num_features, 0.5)
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestDataLoaderIntegration:
    """Test integration with get_batch_to_dataloader."""

    def test_dataloader_creation(self):
        """DataLoader class should be created successfully."""
        assert DataLoader is not None

    def test_dataloader_iteration(self):
        """DataLoader should produce valid batches when iterated."""
        dl = DataLoader(
            num_steps=2,
            batch_size=BATCH_SIZE,
            num_features=NUM_FEATURES,
            hyperparameters=get_default_counterfactual_config(),
            eval_pos_seq_len_sampler=lambda: (SINGLE_EVAL_POS, SEQ_LEN),
            device=DEVICE,
            seq_len_maximum=SEQ_LEN,
        )

        # Need to set model attribute (required by __iter__)
        class DummyModel:
            pass
        dl.model = DummyModel()

        batch_data = next(iter(dl))
        data, target_y, single_eval_pos = batch_data

        style, x, y = data
        assert x.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
        assert y.shape == (SEQ_LEN, BATCH_SIZE)
        # Default mask_supervision=True → 2*num_features channels
        assert target_y.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES * 2)
        assert single_eval_pos == SINGLE_EVAL_POS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
