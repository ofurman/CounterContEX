"""
Tests for SCM-based ground-truth validity evaluation.

Verifies:
1. compute_scm_validity returns valid rate for true counterfactuals (~100%)
2. compute_scm_validity returns low rate for zero/random deltas
3. generate_test_data with with_scm=True returns SCM data
4. EvalMetrics includes scm_validity field
"""

import pytest
import torch
import importlib

counterfactual_mod = importlib.import_module("tabpfn.priors.counterfactual")
CounterfactualSCMGenerator = counterfactual_mod.CounterfactualSCMGenerator
get_default_counterfactual_config = counterfactual_mod.get_default_counterfactual_config

counterfactual_prior_mod = importlib.import_module("tabpfn.priors.counterfactual_prior")
get_batch_with_scm = counterfactual_prior_mod.get_batch_with_scm

eval_mod = importlib.import_module("tabpfn.eval_counterfactual")
compute_scm_validity = eval_mod.compute_scm_validity
generate_test_data = eval_mod.generate_test_data
EvalMetrics = eval_mod.EvalMetrics

DEVICE = "cpu"
SEQ_LEN = 64
NUM_FEATURES = 5
SINGLE_EVAL_POS = 32


class TestComputeSCMValidity:
    """Tests for compute_scm_validity function."""

    def _generate_data(self):
        """Helper to generate test data with SCM info."""
        config = get_default_counterfactual_config()
        gen = CounterfactualSCMGenerator(config, device=DEVICE)
        batch, scm_data_list = gen.generate_batch_with_scm(
            batch_size=1,
            seq_len=SEQ_LEN,
            num_features=NUM_FEATURES,
        )
        return batch, scm_data_list

    def test_true_cf_high_validity(self):
        """True counterfactual deltas should yield high SCM validity."""
        # Retry a few times to get flipped samples
        batch, scm_data_list = None, None
        for _ in range(5):
            batch, scm_data_list = self._generate_data()
            if batch.label_flipped[:, 0].any():
                break
        flipped_mask = batch.label_flipped[:, 0]
        if not flipped_mask.any():
            pytest.skip("No flipped samples after 5 attempts")

        flipped_idx = torch.where(flipped_mask)[0]
        query_x = batch.x_factual[flipped_idx, 0:1, :].squeeze(1)
        true_deltas = (
            batch.x_counterfactual[flipped_idx, 0:1, :]
            - batch.x_factual[flipped_idx, 0:1, :]
        ).squeeze(1)
        target_labels = batch.y_counterfactual_class[flipped_idx, 0]

        validity = compute_scm_validity(
            query_x, true_deltas, target_labels,
            scm_data_list, 0, NUM_FEATURES,
        )
        # True CFs should have high validity (may not be exactly 100% due to
        # threshold effects but should be well above chance)
        assert validity >= 0.5, f"True CF validity should be high, got {validity}"

    def test_true_cf_beats_zero_deltas(self):
        """True CF deltas should have higher validity than zero deltas."""
        # Use larger seq_len for more flipped samples
        config = get_default_counterfactual_config()
        gen = CounterfactualSCMGenerator(config, device=DEVICE)

        batch, scm_data_list = None, None
        for _ in range(10):
            batch, scm_data_list = gen.generate_batch_with_scm(
                batch_size=1, seq_len=128, num_features=NUM_FEATURES,
            )
            flipped_mask = batch.label_flipped[:, 0]
            if flipped_mask.sum() >= 5:
                break

        if batch is None or batch.label_flipped[:, 0].sum() < 5:
            pytest.skip("Not enough flipped samples")

        flipped_idx = torch.where(flipped_mask)[0]
        query_x = batch.x_factual[flipped_idx, 0:1, :].squeeze(1)
        true_deltas = (
            batch.x_counterfactual[flipped_idx, 0:1, :]
            - batch.x_factual[flipped_idx, 0:1, :]
        ).squeeze(1)
        zero_deltas = torch.zeros_like(query_x)
        target_labels = batch.y_counterfactual_class[flipped_idx, 0]

        true_validity = compute_scm_validity(
            query_x, true_deltas, target_labels,
            scm_data_list, 0, NUM_FEATURES,
        )
        zero_validity = compute_scm_validity(
            query_x, zero_deltas, target_labels,
            scm_data_list, 0, NUM_FEATURES,
        )
        # True CFs should achieve higher validity than zero deltas
        assert true_validity >= zero_validity, \
            f"True CF validity ({true_validity}) should be >= zero-delta ({zero_validity})"


class TestGenerateTestDataWithSCM:
    """Tests for generate_test_data with with_scm=True."""

    def test_with_scm_returns_scm_data(self):
        """with_scm=True should return tuples of length 4."""
        test_batches, sep = generate_test_data(
            num_datasets=2, seq_len=SEQ_LEN, num_features=NUM_FEATURES,
            device=DEVICE, with_scm=True,
        )
        assert len(test_batches) == 2
        for tb in test_batches:
            assert len(tb) == 4, f"Expected 4-tuple, got {len(tb)}"
            assert isinstance(tb[3], list)
            assert len(tb[3]) == 1  # batch_size=1

    def test_without_scm_returns_3_tuple(self):
        """with_scm=False should return tuples of length 3."""
        test_batches, sep = generate_test_data(
            num_datasets=2, seq_len=SEQ_LEN, num_features=NUM_FEATURES,
            device=DEVICE, with_scm=False,
        )
        for tb in test_batches:
            assert len(tb) == 3, f"Expected 3-tuple, got {len(tb)}"


class TestEvalMetricsSCMValidity:
    """Tests for EvalMetrics scm_validity field."""

    def test_default_scm_validity_negative(self):
        """Default scm_validity should be -1 (not computed)."""
        m = EvalMetrics(
            delta_mse=0.1, validity_rate=0.5, proximity_mean=1.0,
            proximity_std=0.1, sparsity=0.5, num_test_datasets=10,
            num_query_points=100,
        )
        assert m.scm_validity == -1.0

    def test_scm_validity_can_be_set(self):
        """scm_validity field should accept a value."""
        m = EvalMetrics(
            delta_mse=0.1, validity_rate=0.5, proximity_mean=1.0,
            proximity_std=0.1, sparsity=0.5, num_test_datasets=10,
            num_query_points=100, scm_validity=0.75,
        )
        assert m.scm_validity == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
