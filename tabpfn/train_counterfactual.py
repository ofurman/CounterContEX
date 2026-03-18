"""
Training script for counterfactual generation with TabPFN.

Adapts the TabPFN transformer to output continuous delta vectors instead of
class logits. The model receives (context_x, context_y, query_x, target_y)
and predicts delta = x_counterfactual - x_factual for query positions.

Usage:
    python -m tabpfn.train_counterfactual
"""

import time
import itertools
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from tabpfn.transformer import TransformerModel
from tabpfn.priors.counterfactual_prior import DataLoader as CounterfactualDataLoader
import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings
import tabpfn.utils as utils
from tabpfn.utils import get_cosine_schedule_with_warmup, init_dist


# ---------- default toy-experiment config ----------

TOY_CONFIG = dict(
    # SCM settings
    num_features=5,
    num_classes=2,
    seq_len=128,
    batch_size=16,
    # Model settings
    emsize=64,
    nlayers=4,
    nhead=2,
    nhid=128,
    dropout=0.0,
    # Training settings
    epochs=50,
    steps_per_epoch=100,
    lr=0.001,
    bptt=128,
    warmup_epochs=5,
    weight_decay=0.0,
)


def train_counterfactual(
    num_features=5,
    seq_len=128,
    batch_size=16,
    emsize=64,
    nlayers=4,
    nhead=2,
    nhid=128,
    dropout=0.0,
    epochs=50,
    steps_per_epoch=100,
    lr=0.001,
    bptt=128,
    warmup_epochs=5,
    weight_decay=0.0,
    gpu_device="cuda:0",
    extra_prior_kwargs=None,
    verbose=True,
    epoch_callback=None,
    mask_supervision=True,
    mask_loss_weight=0.5,
    dataloader=None,
    loss_type="mse",
):
    """Train a counterfactual generation model.

    Returns:
        (total_loss, model, dataloader)
    """
    device = gpu_device if torch.cuda.is_available() else "cpu:0"
    print(f"Using {device} device")
    using_dist, rank, device = init_dist(device)

    # Distributional loss overrides mask_supervision (output is mean + log_var)
    if loss_type == "distributional":
        mask_supervision = False
        n_out = num_features * 2  # first half = mean, second half = log_var
    elif mask_supervision:
        n_out = num_features * 2  # first half = delta, second half = mask logits
    else:
        n_out = num_features

    # --- data loader ---
    single_eval_pos = seq_len // 2

    if dataloader is not None:
        dl = dataloader
    else:
        def eval_pos_seq_len_sampler():
            return single_eval_pos, bptt

        hp = dict(extra_prior_kwargs or {})
        hp["mask_supervision"] = mask_supervision
        prior_kwargs = dict(
            num_features=num_features,
            hyperparameters=hp,
        )
        dl = CounterfactualDataLoader(
            num_steps=steps_per_epoch,
            batch_size=batch_size,
            eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
            seq_len_maximum=bptt,
            device=device,
            **prior_kwargs,
        )

    # --- model ---
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
    criterion = nn.MSELoss(reduction="none")  # used as fallback / inside composite
    model.criterion = criterion
    _mask_supervision = mask_supervision
    _mask_loss_weight = mask_loss_weight
    _num_features = num_features

    print(
        f"Using a Transformer with "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters"
    )

    model.to(device)
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, broadcast_buffers=False
        )
    dl.model = model
    utils.check_compatibility(dl)

    # --- optimizer & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_epochs, epochs if epochs else 100
    )

    # --- training loop ---
    total_loss = float("inf")
    loss_history = []

    try:
        for epoch in range(1, epochs + 1) if epochs else itertools.count(1):
            epoch_start = time.time()
            epoch_loss = _train_one_epoch(
                model, dl, criterion, optimizer, single_eval_pos,
                steps_per_epoch, device, n_out,
                mask_supervision=_mask_supervision,
                mask_loss_weight=_mask_loss_weight,
                num_features=_num_features,
                loss_type=loss_type,
            )
            total_loss = epoch_loss
            loss_history.append(epoch_loss)

            if verbose:
                elapsed = time.time() - epoch_start
                loss_label = {"mse": "MSE", "distributional": "NLL", "validity": "VAL"}.get(loss_type, "MSE")
                print(
                    f"| epoch {epoch:3d} | time {elapsed:5.2f}s "
                    f"| {loss_label} loss {epoch_loss:.4f} "
                    f"| lr {scheduler.get_last_lr()[0]:.6f}"
                )

            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch / epochs)

            scheduler.step()
    except KeyboardInterrupt:
        pass

    if rank == 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        return total_loss, model.to("cpu"), dl, loss_history

    return total_loss, model, dl, loss_history


def _train_one_epoch(model, dl, criterion, optimizer, single_eval_pos,
                     steps_per_epoch, device, n_out,
                     mask_supervision=False, mask_loss_weight=0.5,
                     num_features=None, loss_type="mse"):
    """Run one training epoch, return mean loss."""
    model.train()
    total_loss = 0.0

    for batch_idx, (data, targets, sep) in enumerate(dl):
        # data = (style, x, y), targets = (seq_len, batch, target_channels)
        data_device = tuple(
            e.to(device) if torch.is_tensor(e) else e for e in data
        )
        output = model(data_device, single_eval_pos=single_eval_pos)
        # output: (num_query, batch, n_out)

        # Slice targets to query positions only
        query_targets = targets[single_eval_pos:].to(device)

        if loss_type == "distributional":
            # Targets are plain deltas (num_query, batch, num_features)
            loss = _distributional_loss(output, query_targets, num_features)
        elif mask_supervision and num_features is not None:
            loss = _composite_loss(
                output, query_targets, num_features, mask_loss_weight,
            )
        else:
            # Plain MSE on all features
            losses = criterion(output, query_targets)
            loss = losses.mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / steps_per_epoch


def _distributional_loss(output, query_targets, num_features):
    """Gaussian NLL loss: model outputs mean + log_var per feature.

    output shape: (num_query, batch, num_features * 2)
      first num_features = predicted mean (delta)
      last num_features = predicted log_variance

    query_targets shape: (num_query, batch, num_features) — true deltas
    """
    mu = output[..., :num_features]
    log_var = output[..., num_features:]

    # Clamp log_var for numerical stability
    log_var = log_var.clamp(-10, 10)

    # Gaussian NLL: 0.5 * (log_var + (target - mu)^2 / exp(log_var))
    nll = 0.5 * (log_var + (query_targets - mu) ** 2 / log_var.exp())
    return nll.mean()


def _composite_loss(output, query_targets, num_features, mask_loss_weight):
    """Compute composite loss: masked delta MSE + mask BCE.

    Args:
        output: (num_query, batch, num_features * 2) — first half = pred deltas, second = mask logits
        query_targets: (num_query, batch, num_features * 2) — first half = true deltas, second = true mask
        num_features: number of features
        mask_loss_weight: weight for mask BCE loss

    Returns:
        scalar loss
    """
    pred_delta = output[..., :num_features]
    pred_mask_logits = output[..., num_features:]
    true_delta = query_targets[..., :num_features]
    true_mask = query_targets[..., num_features:]

    # Delta MSE — weighted by true mask (only penalize on actually changed features)
    delta_sq_err = (pred_delta - true_delta) ** 2  # (num_query, batch, num_features)
    # Per-sample: mean over perturbed features only
    mask_sum = true_mask.sum(dim=-1).clamp(min=1)  # (num_query, batch)
    delta_loss = (delta_sq_err * true_mask).sum(dim=-1) / mask_sum  # (num_query, batch)

    # Mask BCE — learn which features to perturb
    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, true_mask, reduction='none',
    ).mean(dim=-1)  # (num_query, batch)

    # Combined
    loss = (delta_loss + mask_loss_weight * mask_loss).mean()
    return loss


if __name__ == "__main__":
    total_loss, model, dl, history = train_counterfactual(**TOY_CONFIG)
    print(f"\nFinal MSE loss: {total_loss:.4f}")

    # Save model
    save_path = "counterfactual_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
