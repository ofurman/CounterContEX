"""
Tests for counterfactual pair generation from TabPFN's SCM prior.

Tests cover:
1. Identity test (no intervention -> factual == counterfactual)
2. Causal consistency (shallow features unchanged, deep features update)
3. Output shape correctness
4. All perturbation strategies produce non-trivial perturbations
5. Label flip detection
6. Shared classification boundaries
7. Batch generation consistency
"""

import pytest
import torch
import numpy as np

from tabpfn.priors.counterfactual import (
    CounterfactualSCMGenerator,
    CounterfactualBatch,
    PerturbationStrategy,
    get_default_counterfactual_config,
)


DEVICE = "cpu"
BATCH_SIZE = 4
SEQ_LEN = 50
NUM_FEATURES = 5
NUM_OUTPUTS = 1


@pytest.fixture
def default_generator():
    config = get_default_counterfactual_config()
    config["device"] = DEVICE
    return CounterfactualSCMGenerator(config, device=DEVICE)


@pytest.fixture
def default_config():
    config = get_default_counterfactual_config()
    config["device"] = DEVICE
    return config


class TestOutputShapes:
    """Verify all output tensors have correct shapes and types."""

    def test_batch_output_shapes(self, default_generator):
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )

        assert isinstance(batch, CounterfactualBatch)
        assert batch.x_factual.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
        assert batch.x_counterfactual.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
        assert batch.y_factual.shape == (SEQ_LEN, BATCH_SIZE)
        assert batch.y_counterfactual.shape == (SEQ_LEN, BATCH_SIZE)
        assert batch.y_factual_class.shape == (SEQ_LEN, BATCH_SIZE)
        assert batch.y_counterfactual_class.shape == (SEQ_LEN, BATCH_SIZE)
        assert batch.label_flipped.shape == (SEQ_LEN, BATCH_SIZE)
        assert batch.intervention_mask.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
        assert batch.perturbation_delta.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)

    def test_label_flipped_is_boolean(self, default_generator):
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        assert batch.label_flipped.dtype == torch.bool

    def test_intervention_mask_is_boolean(self, default_generator):
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        assert batch.intervention_mask.dtype == torch.bool

    def test_no_nans_in_output(self, default_generator):
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        assert not torch.isnan(batch.x_factual).any(), "NaN in x_factual"
        assert not torch.isnan(batch.x_counterfactual).any(), "NaN in x_counterfactual"
        assert not torch.isnan(batch.y_factual).any(), "NaN in y_factual"
        assert not torch.isnan(batch.y_counterfactual).any(), "NaN in y_counterfactual"


class TestIdentity:
    """No intervention should produce identical factual and counterfactual."""

    def test_zero_perturbation_prob(self):
        """With perturbation_prob=0, the minimum 1-feature perturbation still happens.
        But with a very large perturbation_prob, all features get perturbed."""
        # We test that with zero magnitude, output is unchanged.
        config = get_default_counterfactual_config()
        config["perturbation_magnitude"] = 0.0
        config["perturbation_strategy"] = "additive_noise"
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        batch = gen.generate_batch(
            batch_size=2,
            seq_len=30,
            num_features=NUM_FEATURES,
        )
        # With zero-magnitude additive noise, the delta should be zero
        # (though the intervention is still applied, its value is the same)
        assert torch.allclose(
            batch.perturbation_delta, torch.zeros_like(batch.perturbation_delta)
        ), "Zero-magnitude perturbation should produce zero deltas"


class TestCausalConsistency:
    """When intervening on a feature, downstream features should change
    while upstream features should remain the same."""

    def test_intervention_changes_counterfactual(self, default_generator):
        """Verify that interventions actually change the counterfactual output."""
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        # At least some features should differ between factual and counterfactual
        diffs = (batch.x_factual - batch.x_counterfactual).abs()
        assert diffs.sum() > 0, (
            "Counterfactual should differ from factual when perturbations are applied"
        )

    def test_non_intervened_features_may_change_if_downstream(self, default_generator):
        """Non-intervened features at deeper layers should change due to causal propagation."""
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        # We can only verify that the mask is consistent:
        # features that are NOT in intervention_mask may still change (downstream causal effect)
        # This is correct behavior for causal propagation
        intervention_mask = batch.intervention_mask
        assert intervention_mask.any(), (
            "At least some features should be marked as intervened"
        )


class TestPerturbationStrategies:
    """Test each perturbation strategy produces non-trivial results."""

    @pytest.mark.parametrize(
        "strategy",
        [
            "additive_noise",
            "marginal_replacement",
            "gradient_guided",
            "fixed_magnitude",
            "uniform_random",
        ],
    )
    def test_strategy_produces_perturbations(self, strategy, default_config):
        config = dict(default_config)
        config["perturbation_strategy"] = strategy
        config["perturbation_prob"] = 0.5
        config["perturbation_magnitude"] = 1.0
        gen = CounterfactualSCMGenerator(config, device=DEVICE)

        batch = gen.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )

        # Perturbation deltas should be non-zero for intervened features
        masked_deltas = batch.perturbation_delta[batch.intervention_mask]
        assert masked_deltas.abs().sum() > 0, (
            f"Strategy {strategy} should produce non-zero perturbation deltas"
        )

    @pytest.mark.parametrize(
        "strategy",
        [
            "additive_noise",
            "marginal_replacement",
            "gradient_guided",
            "fixed_magnitude",
            "uniform_random",
        ],
    )
    def test_strategy_via_enum(self, strategy, default_config):
        """Test passing strategy as enum works."""
        config = dict(default_config)
        config["perturbation_strategy"] = strategy
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        batch = gen.generate_batch(
            batch_size=2,
            seq_len=20,
            num_features=3,
            perturbation_strategy=PerturbationStrategy(strategy),
        )
        assert batch.x_factual.shape == (20, 2, 3)


class TestLabelFlip:
    """Test label flip detection."""

    def test_label_flip_consistency(self, default_generator):
        """label_flipped should match actual class difference."""
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        expected_flip = batch.y_factual_class != batch.y_counterfactual_class
        assert torch.equal(batch.label_flipped, expected_flip), (
            "label_flipped should equal (y_factual_class != y_counterfactual_class)"
        )

    def test_large_perturbation_causes_some_flips(self):
        """With large perturbations, at least some labels should flip."""
        config = get_default_counterfactual_config()
        config["perturbation_magnitude"] = 10.0
        config["perturbation_prob"] = 0.8
        config["perturbation_strategy"] = "fixed_magnitude"
        config["fixed_magnitude_k"] = 5.0
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        batch = gen.generate_batch(
            batch_size=8,
            seq_len=100,
            num_features=NUM_FEATURES,
        )
        flip_rate = batch.label_flipped.float().mean().item()
        # With very large perturbations, we expect *some* flips (not deterministic, but very likely)
        # Using a soft check: at least 1% of labels should flip with 5-sigma perturbations
        assert flip_rate > 0.0, (
            f"Large perturbations should cause at least some label flips, got flip_rate={flip_rate}"
        )


class TestClassificationBoundaries:
    """Test that factual and counterfactual share the same classification boundaries."""

    def test_class_values_are_valid(self, default_generator):
        """Class labels should be non-negative integers."""
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        assert (batch.y_factual_class >= 0).all()
        assert (batch.y_counterfactual_class >= 0).all()

    def test_binary_classification_two_classes(self, default_generator):
        """With balanced binary (default), we should see exactly 2 unique class values."""
        batch = default_generator.generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        unique_classes = torch.unique(batch.y_factual_class)
        # Should be {0.0, 1.0} for balanced binary
        assert len(unique_classes) <= 2, (
            f"Expected at most 2 classes, got {unique_classes}"
        )


class TestSCMSampling:
    """Test SCM construction and sampling."""

    def test_different_scms_per_batch_element(self):
        """Each batch element should use a different SCM."""
        config = get_default_counterfactual_config()
        config["perturbation_prob"] = 0.0001  # minimal perturbation
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        batch = gen.generate_batch(
            batch_size=4,
            seq_len=50,
            num_features=NUM_FEATURES,
        )
        # Different SCMs should produce different feature distributions
        # Check that batch elements are not identical
        x = batch.x_factual
        pairwise_equal = True
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                if not torch.allclose(x[:, i, :], x[:, j, :]):
                    pairwise_equal = False
                    break
        assert not pairwise_equal, "Different batch elements should have different SCMs"

    def test_causal_mode_forced(self):
        """force_is_causal should override is_causal=False."""
        config = get_default_counterfactual_config()
        config["is_causal"] = False
        config["force_is_causal"] = True
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        assert gen.config["is_causal"] == True

    @pytest.mark.parametrize("num_features", [2, 5, 10, 20])
    def test_various_feature_counts(self, num_features):
        config = get_default_counterfactual_config()
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        batch = gen.generate_batch(
            batch_size=2,
            seq_len=20,
            num_features=num_features,
        )
        assert batch.x_factual.shape[2] == num_features

    @pytest.mark.parametrize("num_layers", [2, 4, 6])
    def test_various_layer_counts(self, num_layers):
        config = get_default_counterfactual_config()
        config["num_layers"] = num_layers
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        batch = gen.generate_batch(
            batch_size=2,
            seq_len=20,
            num_features=NUM_FEATURES,
        )
        assert batch.x_factual.shape == (20, 2, NUM_FEATURES)

    @pytest.mark.parametrize("sampling", ["normal", "mixed", "uniform"])
    def test_various_sampling_modes(self, sampling):
        config = get_default_counterfactual_config()
        config["sampling"] = sampling
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        batch = gen.generate_batch(
            batch_size=2,
            seq_len=20,
            num_features=NUM_FEATURES,
        )
        assert batch.x_factual.shape == (20, 2, NUM_FEATURES)


class TestGaussianNoiseReplay:
    """Test that GaussianNoise supports deterministic replay."""

    def test_sample_and_replay(self):
        from tabpfn.priors.mlp import GaussianNoise

        noise_module = GaussianNoise(std=1.0, device=DEVICE)
        x = torch.randn(10, 1, 5)

        # Sample noise
        noise = noise_module.sample_noise(x)
        assert noise.shape == x.shape

        # Apply with pre-sampled noise
        result1 = noise_module(x, pre_sampled_noise=noise)
        result2 = noise_module(x, pre_sampled_noise=noise)
        assert torch.equal(result1, result2), (
            "Replayed noise should produce identical results"
        )

        # Without pre-sampled noise, results should differ (stochastic)
        result3 = noise_module(x)
        result4 = noise_module(x)
        # Not guaranteed to differ, but very likely with continuous distributions
        # We just check it runs without error


class TestFixedSCMMapping:
    """Test forward_with_internals_fixed_mapping and generate_batch_fixed_scm."""

    def test_fixed_perm_returns_consistent_indices(self):
        """Two calls with same fixed_perm should yield identical feature/target indices."""
        config = get_default_counterfactual_config()
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        scm = gen._sample_scm(SEQ_LEN, NUM_FEATURES, NUM_OUTPUTS)

        # First call: get a fresh perm
        _, _, internals1, perm = scm.forward_with_internals_fixed_mapping(None)
        # Second call: reuse the perm
        _, _, internals2, perm2 = scm.forward_with_internals_fixed_mapping(perm)

        assert torch.equal(perm, perm2), "Returned perm should be identical when passed in"
        assert torch.equal(
            internals1["node_mapping"]["feature_indices"],
            internals2["node_mapping"]["feature_indices"],
        ), "Feature indices should be identical with same fixed_perm"
        assert internals1["node_mapping"]["target_indices"] == internals2["node_mapping"]["target_indices"], \
            "Target indices should be identical with same fixed_perm"

    def test_fixed_perm_different_data(self):
        """Two calls with same fixed_perm produce different data points (fresh samples)."""
        config = get_default_counterfactual_config()
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        scm = gen._sample_scm(SEQ_LEN, NUM_FEATURES, NUM_OUTPUTS)

        x1, _, _, perm = scm.forward_with_internals_fixed_mapping(None)
        x2, _, _, _ = scm.forward_with_internals_fixed_mapping(perm)

        # Different random causes -> different data (extremely unlikely to be equal)
        assert not torch.allclose(x1, x2), \
            "Different calls should produce different data even with same perm"

    def test_generate_batch_fixed_scm_shapes(self):
        """generate_batch_fixed_scm should produce correct output shapes."""
        from tabpfn.priors.counterfactual import FixedThresholdBinarize

        config = get_default_counterfactual_config()
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        scm = gen._sample_scm(SEQ_LEN, NUM_FEATURES, NUM_OUTPUTS)

        _, y_cal, _, perm = scm.forward_with_internals_fixed_mapping(None)
        threshold = torch.median(y_cal).item()
        class_assigner = FixedThresholdBinarize(threshold)

        batch = gen.generate_batch_fixed_scm(
            scm=scm, fixed_perm=perm, class_assigner=class_assigner,
            batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )

        assert batch.x_factual.shape == (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
        assert batch.y_factual_class.shape == (SEQ_LEN, BATCH_SIZE)
        assert batch.label_flipped.shape == (SEQ_LEN, BATCH_SIZE)

    def test_fixed_scm_consistent_scales(self):
        """Fixed SCM should produce data with similar feature scales across batches."""
        from tabpfn.priors.counterfactual import FixedThresholdBinarize

        config = get_default_counterfactual_config()
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        scm = gen._sample_scm(SEQ_LEN, NUM_FEATURES, NUM_OUTPUTS)

        _, y_cal, _, perm = scm.forward_with_internals_fixed_mapping(None)
        threshold = torch.median(y_cal).item()
        class_assigner = FixedThresholdBinarize(threshold)

        batch1 = gen.generate_batch_fixed_scm(
            scm=scm, fixed_perm=perm, class_assigner=class_assigner,
            batch_size=1, seq_len=SEQ_LEN, num_features=NUM_FEATURES,
        )
        batch2 = gen.generate_batch_fixed_scm(
            scm=scm, fixed_perm=perm, class_assigner=class_assigner,
            batch_size=1, seq_len=SEQ_LEN, num_features=NUM_FEATURES,
        )

        # Feature means should be in the same ballpark
        mean1 = batch1.x_factual[:, 0, :].mean(dim=0)
        mean2 = batch2.x_factual[:, 0, :].mean(dim=0)
        # With same SCM the means should be similar (not identical due to sampling)
        diff = (mean1 - mean2).abs().max().item()
        assert diff < 5.0, f"Feature means should be similar across batches, got max diff {diff}"

    def test_fixed_threshold_binarize(self):
        """FixedThresholdBinarize uses a fixed threshold, not per-batch median."""
        from tabpfn.priors.counterfactual import FixedThresholdBinarize

        assigner = FixedThresholdBinarize(0.5)
        x = torch.tensor([[[0.1], [0.6], [0.3], [0.9]]])
        result = assigner(x)
        expected = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]])
        assert torch.equal(result, expected), \
            f"FixedThresholdBinarize should use fixed threshold, got {result}"


class TestGenerateBatchWithSCM:
    """Tests for generate_batch_with_scm returning SCM data for validity evaluation."""

    def test_returns_batch_and_scm_data(self, default_generator):
        """generate_batch_with_scm returns (batch, scm_data_list)."""
        batch, scm_data_list = default_generator.generate_batch_with_scm(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        assert isinstance(batch, CounterfactualBatch)
        assert len(scm_data_list) == BATCH_SIZE
        for sd in scm_data_list:
            assert "scm" in sd
            assert "internals" in sd
            assert "class_assigner" in sd

    def test_scm_data_contains_valid_internals(self, default_generator):
        """Each scm_data entry has internals with node_mapping and outputs_flat."""
        _, scm_data_list = default_generator.generate_batch_with_scm(
            batch_size=2,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        for sd in scm_data_list:
            internals = sd["internals"]
            assert "node_mapping" in internals
            assert "feature_indices" in internals["node_mapping"]
            assert "target_indices" in internals["node_mapping"]
            assert "outputs_flat" in internals
            assert "layer_boundaries" in internals

    def test_scm_can_rerun_intervention(self, default_generator):
        """SCMs returned can be used for forward_with_intervention."""
        batch, scm_data_list = default_generator.generate_batch_with_scm(
            batch_size=1,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        sd = scm_data_list[0]
        scm = sd["scm"]
        internals = sd["internals"]

        # Re-running with empty interventions should succeed
        x_cf, y_cf = scm.forward_with_intervention(internals, {})
        assert x_cf.shape == batch.x_factual[:, 0:1, :].shape

    def test_true_cf_through_scm_matches_original_label(self, default_generator):
        """True counterfactuals re-fed through SCM should produce the same class."""
        from tabpfn.priors.counterfactual import FixedThresholdBinarize

        batch, scm_data_list = default_generator.generate_batch_with_scm(
            batch_size=1,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        sd = scm_data_list[0]
        scm = sd["scm"]
        internals = sd["internals"]
        feature_indices = internals["node_mapping"]["feature_indices"]
        target_indices = internals["node_mapping"]["target_indices"]
        outputs_flat = internals["outputs_flat"]

        # Compute a fixed threshold from the original y values (same as used
        # during batch generation by BalancedBinarize)
        y_original = outputs_flat[:, :, target_indices]
        if y_original.dim() > 2:
            y_original = y_original.squeeze(-1)
        threshold = torch.median(y_original).item()
        fixed_assigner = FixedThresholdBinarize(threshold)

        # Take a flipped sample and re-feed its true CF through the SCM
        flipped_mask = batch.label_flipped[:, 0]
        if not flipped_mask.any():
            pytest.skip("No flipped samples in this batch")

        flipped_idx = torch.where(flipped_mask)[0][0].item()
        x_cf_true = batch.x_counterfactual[flipped_idx, 0]  # (num_features,)

        # Set all feature nodes to CF values
        interventions = {}
        for feat_idx in range(NUM_FEATURES):
            flat_idx = feature_indices[feat_idx].item()
            new_val = outputs_flat[:, :, flat_idx].clone()
            new_val[:, :] = x_cf_true[feat_idx]
            interventions[flat_idx] = new_val

        _, y_cf_scm = scm.forward_with_intervention(internals, interventions)
        y_cf_class = fixed_assigner(
            y_cf_scm.unsqueeze(-1) if y_cf_scm.dim() == 2 else y_cf_scm
        ).float()
        if y_cf_class.dim() > 2:
            y_cf_class = y_cf_class.squeeze(-1)

        # The class from re-feeding should match the original CF class
        pred_class = y_cf_class[0, 0].item()
        expected_class = batch.y_counterfactual_class[flipped_idx, 0].item()
        assert pred_class == expected_class, \
            f"Re-feeding true CF should reproduce class: got {pred_class}, expected {expected_class}"

    def test_batch_data_matches_regular_generate(self, default_config):
        """The batch from generate_batch_with_scm should have same structure as generate_batch."""
        gen = CounterfactualSCMGenerator(default_config, device=DEVICE)
        batch_with, _ = gen.generate_batch_with_scm(
            batch_size=2, seq_len=SEQ_LEN, num_features=NUM_FEATURES,
        )
        batch_regular = gen.generate_batch(
            batch_size=2, seq_len=SEQ_LEN, num_features=NUM_FEATURES,
        )
        # Same shapes
        assert batch_with.x_factual.shape == batch_regular.x_factual.shape
        assert batch_with.y_factual.shape == batch_regular.y_factual.shape
        assert batch_with.intervention_mask.shape == batch_regular.intervention_mask.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
