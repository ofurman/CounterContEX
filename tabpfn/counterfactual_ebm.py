import copy
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EBMConfig:
    lambda_val: float = 1.0
    lambda_clf: float = 5.0
    clf_uses_logits: bool = False
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    sgld_steps: int = 400
    sgld_lr: float = 0.001
    sgld_lr_decay: float = 1.0
    sgld_noise: float = 0.1
    sgld_grad_clip: float | None = 10.0
    init_noise_std: float = 0.01
    tab_validity_thresh: float = 0.5
    clf_validity_thresh: float = 0.5


def _ensure_2d(x):
    return x.unsqueeze(0) if x.ndim == 1 else x


def _to_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device="cpu", dtype=dtype)
    return torch.as_tensor(x, device="cpu", dtype=dtype)


def _resolve_target_index(tabpfn, target_class):
    if not hasattr(tabpfn, "classes_"):
        return int(target_class)

    classes = np.asarray(tabpfn.classes_)
    match = np.where(classes == target_class)[0]
    if match.size == 0:
        raise ValueError(
            f"target_class={target_class} is not in tabpfn.classes_={classes.tolist()}"
        )
    return int(match[0])


def _get_tabpfn_logits_fn(tabpfn):
    diff_tabpfn = copy.copy(tabpfn)
    diff_tabpfn.X_ = _to_tensor(tabpfn.X_)
    diff_tabpfn.no_grad = False
    diff_tabpfn.no_preprocess_mode = True

    def logits_fn(x):
        return diff_tabpfn.predict_proba(_ensure_2d(x), return_logits=True)

    return logits_fn


def debug_tabpfn_gradient(tabpfn, x, target_class):
    target_index = _resolve_target_index(tabpfn, target_class)
    logits_fn = _get_tabpfn_logits_fn(tabpfn)
    x = _ensure_2d(_to_tensor(x)).detach().requires_grad_(True)
    logits = _ensure_2d(logits_fn(x))
    loss = -logits[:, target_index].mean()
    grad = torch.autograd.grad(loss, x)[0]
    return {
        "target_index": int(target_index),
        "logits": logits.detach().cpu().numpy(),
        "loss": float(loss.detach().item()),
        "grad": grad.detach().cpu().numpy(),
        "grad_norm": float(grad.norm().detach().item()),
        "has_grad": bool(grad is not None and torch.isfinite(grad).all()),
    }


def compute_energy(
    x,
    x_0,
    target_index,
    tabpfn_logits_fn,
    clf_proba_fn,
    config,
    return_components=False,
):
    logits = _ensure_2d(tabpfn_logits_fn(x))
    tab_log_probs = torch.log_softmax(logits, dim=-1)
    tab_target_probs = torch.exp(tab_log_probs[:, target_index])
    e_tab = (-logits[:, target_index]).mean()

    clf_output = _ensure_2d(_to_tensor(clf_proba_fn(x), dtype=x.dtype))
    if config.clf_uses_logits:
        clf_log_probs = torch.log_softmax(clf_output, dim=-1)
        clf_target_probs = torch.exp(clf_log_probs[:, target_index])
        e_clf = (-clf_log_probs[:, target_index]).mean()
    else:
        clf_target_probs = torch.clamp(clf_output[:, target_index], min=1e-8, max=1.0)
        e_clf = (-torch.log(clf_target_probs)).mean()

    e_l1 = torch.abs(x - x_0).sum(dim=-1).mean()
    e_l2 = ((x - x_0) ** 2).sum(dim=-1).mean()

    energy = (
        config.lambda_val * e_tab
        + config.lambda_clf * e_clf
        + config.lambda_l1 * e_l1
        + config.lambda_l2 * e_l2
    )

    if not return_components:
        return energy

    return energy, {
        "tab": float(e_tab.detach().item()),
        "clf": float(e_clf.detach().item()),
        "l1": float((config.lambda_l1 * e_l1).detach().item()),
        "l2": float((config.lambda_l2 * e_l2).detach().item()),
        "tab_p": float(tab_target_probs.mean().detach().item()),
        "clf_p": float(clf_target_probs.mean().detach().item()),
    }


def generate_counterfactuals(
    x_0,
    target_class,
    tabpfn,
    clf_proba_fn=None,
    config=None,
    return_trajectory=False,
):
    if clf_proba_fn is None:
        raise ValueError("`clf_proba_fn` is mandatory for counterfactual generation.")

    config = config or EBMConfig()
    if config.lambda_clf <= 0:
        raise ValueError("`config.lambda_clf` must be > 0.")

    target_index = _resolve_target_index(tabpfn, target_class)
    tabpfn_logits_fn = _get_tabpfn_logits_fn(tabpfn)
    x_0 = _to_tensor(x_0)
    single_input = x_0.ndim == 1
    x_0 = _ensure_2d(x_0)

    x = x_0.clone() + config.init_noise_std * torch.randn_like(x_0)
    energy_history = []
    trajectory = [x.detach().numpy().copy()] if return_trajectory else None
    best_x = x.detach().clone()
    best_score = None
    best_step = -1

    for step in range(config.sgld_steps):
        x = x.detach().requires_grad_(True)
        energy, components = compute_energy(
            x,
            x_0,
            target_index,
            tabpfn_logits_fn,
            clf_proba_fn,
            config,
            return_components=True,
        )
        x_current = x.detach().clone()
        grad = torch.autograd.grad(energy, x)[0]

        components["total"] = float(energy.detach().item())
        energy_history.append(components)
        is_valid = (
            components["tab_p"] >= config.tab_validity_thresh
            and components["clf_p"] >= config.clf_validity_thresh
        )
        score = (0 if is_valid else 1, components["total"])
        if best_score is None or score < best_score:
            best_score = score
            best_x = x_current
            best_step = step

        with torch.no_grad():
            if config.sgld_grad_clip is not None and config.sgld_grad_clip > 0:
                grad = torch.clamp(grad, -config.sgld_grad_clip, config.sgld_grad_clip)
            lr = config.sgld_lr * (config.sgld_lr_decay ** step)
            noise = config.sgld_noise * (2.0 * lr) ** 0.5 * torch.randn_like(x)
            x = x - lr * grad + noise
        if return_trajectory and (step + 1) % 10 == 0:
            trajectory.append(x_current.numpy().copy())

    x_cf = best_x.detach().numpy()
    x_0_np = x_0.detach().numpy()

    if single_input:
        x_cf = x_cf[0]
        x_0_np = x_0_np[0]
        if return_trajectory:
            trajectory = [t[0] for t in trajectory]

    result = {
        "x_cf": x_cf,
        "x_0": x_0_np,
        "target_class": target_class,
        "energy_history": energy_history,
        "num_steps": len(energy_history),
        "best_step": best_step,
    }
    if return_trajectory:
        if not np.allclose(trajectory[-1], x_cf):
            trajectory.append(x_cf.copy())
        result["trajectory"] = trajectory
    return result
