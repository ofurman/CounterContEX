"""
Tests for counterfactual generation training script.

Verification checks from Stage 2:
1. Model instantiates without errors
2. Forward pass produces output of shape (num_query, batch, num_features)
3. Loss computes and backpropagation works
4. Training runs for at least 5 epochs without NaN
"""

import pytest
import torch
from torch import nn

from tabpfn.transformer import TransformerModel
from tabpfn.priors.counterfactual_prior import DataLoader as CounterfactualDataLoader
import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings
from tabpfn.train_counterfactual import train_counterfactual


DEVICE = "cpu"
NUM_FEATURES = 5
EMSIZE = 32
NLAYERS = 2
NHEAD = 2
NHID = 64
SEQ_LEN = 32
BATCH_SIZE = 4
SINGLE_EVAL_POS = 16


@pytest.fixture
def small_model():
    encoder = encoders.Linear(NUM_FEATURES, EMSIZE)
    y_encoder = encoders.Linear(1, EMSIZE)
    pos_encoder = positional_encodings.NoPositionalEncoding(EMSIZE, SEQ_LEN * 2)
    model = TransformerModel(
        encoder=encoder,
        n_out=NUM_FEATURES,
        ninp=EMSIZE,
        nhead=NHEAD,
        nhid=NHID,
        nlayers=NLAYERS,
        dropout=0.0,
        y_encoder=y_encoder,
        pos_encoder=pos_encoder,
        efficient_eval_masking=True,
    )
    return model


@pytest.fixture
def sample_batch():
    """Create a synthetic batch matching the counterfactual data format."""
    x = torch.randn(SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
    y = torch.randint(0, 2, (SEQ_LEN, BATCH_SIZE)).float()
    target_deltas = torch.randn(SEQ_LEN, BATCH_SIZE, NUM_FEATURES) * 0.1
    return (None, x, y), target_deltas, SINGLE_EVAL_POS


class TestModelInstantiation:
    def test_model_creates(self, small_model):
        assert isinstance(small_model, TransformerModel)
        assert small_model.n_out == NUM_FEATURES

    def test_model_parameter_count(self, small_model):
        n_params = sum(p.numel() for p in small_model.parameters())
        assert n_params > 0


class TestForwardPass:
    def test_output_shape(self, small_model, sample_batch):
        data, targets, sep = sample_batch
        output = small_model(data, single_eval_pos=sep)
        num_query = SEQ_LEN - SINGLE_EVAL_POS
        assert output.shape == (num_query, BATCH_SIZE, NUM_FEATURES)

    def test_output_no_nan(self, small_model, sample_batch):
        data, targets, sep = sample_batch
        output = small_model(data, single_eval_pos=sep)
        assert not torch.isnan(output).any()


class TestLossAndBackprop:
    def test_mse_loss_computes(self, small_model, sample_batch):
        data, targets, sep = sample_batch
        output = small_model(data, single_eval_pos=sep)
        query_targets = targets[sep:]
        criterion = nn.MSELoss(reduction="none")
        losses = criterion(output, query_targets)
        assert losses.shape == output.shape
        loss = losses.mean()
        assert not torch.isnan(loss)

    def test_backward_pass(self, small_model, sample_batch):
        data, targets, sep = sample_batch
        output = small_model(data, single_eval_pos=sep)
        query_targets = targets[sep:]
        loss = nn.functional.mse_loss(output, query_targets)
        loss.backward()
        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_model.parameters()
        )
        assert has_grad, "At least some parameters should have gradients"


class TestTrainingLoop:
    def test_short_training_runs(self):
        """Training runs for 5 epochs without NaN loss."""
        total_loss, model, dl, history = train_counterfactual(
            num_features=3,
            seq_len=32,
            batch_size=4,
            emsize=32,
            nlayers=2,
            nhead=2,
            nhid=64,
            dropout=0.0,
            epochs=5,
            steps_per_epoch=5,
            lr=0.001,
            bptt=32,
            warmup_epochs=1,
            gpu_device="cpu:0",
            verbose=False,
        )
        assert len(history) == 5
        for loss_val in history:
            assert not (loss_val != loss_val), f"NaN loss at some epoch"  # NaN check
        assert total_loss < float("inf")

    def test_loss_is_finite(self):
        """Loss values should be finite."""
        total_loss, model, dl, history = train_counterfactual(
            num_features=3,
            seq_len=32,
            batch_size=4,
            emsize=32,
            nlayers=2,
            nhead=2,
            nhid=64,
            epochs=3,
            steps_per_epoch=3,
            bptt=32,
            warmup_epochs=1,
            gpu_device="cpu:0",
            verbose=False,
        )
        for h in history:
            assert torch.isfinite(torch.tensor(h)), f"Non-finite loss: {h}"
