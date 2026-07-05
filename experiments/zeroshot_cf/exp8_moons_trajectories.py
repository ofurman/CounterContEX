"""Experiment 8: MOONS per-step greedy trajectory visualization.

Builds the Stage-6 meeting figures for the 2-D MOONS dataset:

  - train scatter + logistic-regression discriminator boundary,
  - axis-aligned factual -> intermediate -> CF trajectories,
  - per-step labels and status coloring,
  - blocked-slice panels showing TabPFN conditional density along each
    single-feature cut through a representative stalled point.

The run is fully offline: models are loaded only through checkpoints.get_models().
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.zeroshot_cf.exp4_greedy_cf import (  # noqa: E402
    N_ESTIMATORS,
    N_PERMUTATIONS,
    TAU,
    TEMPERATURE,
)

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SELECTOR = "prob_ascent"
MAX_CONTEXT = 512
BUDGET = 64
STALL_EPS = 1e-6
RANDOM_STATE_BASE = 42


@dataclass
class TrajectoryRecord:
    row_idx: int
    x0: np.ndarray
    target: int
    states: np.ndarray
    history: List[tuple]
    flipped: bool
    final_p_target: float


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _decision_grid(disc_model, n_grid: int = 220) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.linspace(0.0, 1.0, n_grid)
    gy = np.linspace(0.0, 1.0, n_grid)
    xx, yy = np.meshgrid(gx, gy)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    proba = disc_model.predict_proba(points)[:, 1].reshape(xx.shape)
    return xx, yy, proba


def _boundary_rank(disc_model, X: np.ndarray) -> np.ndarray:
    proba = disc_model.predict_proba(X)
    pred = disc_model.predict(X)
    margin = np.abs(proba[np.arange(len(X)), pred] - 0.5)
    return np.argsort(margin)


def _replay_states(x0: np.ndarray, history: Sequence[tuple]) -> np.ndarray:
    states = [np.asarray(x0, dtype=float).copy()]
    current = states[0].copy()
    for feature_idx, committed_value, *_ in history:
        current = current.copy()
        current[int(feature_idx)] = float(committed_value)
        states.append(current.copy())
    return np.vstack(states)


def _fit_sampler(clf, reg, X_train: np.ndarray, y_train: np.ndarray, target: int):
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    sampler = ConditionalDensitySampler(
        clf=clf,
        reg=reg,
        append_target=True,
        n_permutations=N_PERMUTATIONS,
        temperature=TEMPERATURE,
        random_state=RANDOM_STATE_BASE + int(target),
    )
    sampler.set_context(
        X_train,
        y_context=y_train,
        target_class=None,
        max_context=MAX_CONTEXT,
        selection="random",
    )
    return sampler


def _generate_records(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    ranked_indices: np.ndarray,
    disc_model,
    actionable_idx: List[int],
    clf,
    reg,
    max_test: int,
    fallback_pool: int,
) -> List[TrajectoryRecord]:
    from experiments.zeroshot_cf.greedy import greedy_counterfactual

    selected = list(ranked_indices[:max_test])
    records: Dict[int, TrajectoryRecord] = {}
    samplers: Dict[int, object] = {}

    def _run_one(i: int) -> TrajectoryRecord:
        x0 = X_test[i]
        pred = int(disc_model.predict(x0.reshape(1, -1))[0])
        target = 1 - pred
        if target not in samplers:
            samplers[target] = _fit_sampler(clf, reg, X_train, y_train, target)
        x_cf, _changed, info = greedy_counterfactual(
            samplers[target],
            disc_model,
            x0,
            target,
            actionable_idx,
            SELECTOR,
            tau=TAU,
            budget=BUDGET,
            temperature=TEMPERATURE,
            stall_eps=STALL_EPS,
        )
        p_target = float(disc_model.predict_proba(x_cf.reshape(1, -1))[0, target])
        return TrajectoryRecord(
            row_idx=int(i),
            x0=x0.copy(),
            target=int(target),
            states=_replay_states(x0, info["history"]),
            history=list(info["history"]),
            flipped=bool(info["flipped"]),
            final_p_target=p_target,
        )

    for i in selected:
        records[int(i)] = _run_one(int(i))

    if all(r.flipped for r in records.values()):
        for i in ranked_indices[max_test:fallback_pool]:
            rec = _run_one(int(i))
            records[int(i)] = rec
            if not rec.flipped:
                break

    return [records[i] for i in records]


def _choose_blocked(records: Sequence[TrajectoryRecord]) -> TrajectoryRecord:
    stalled = [r for r in records if not r.flipped]
    if stalled:
        return min(stalled, key=lambda r: (r.final_p_target, -len(r.history)))
    return max(records, key=lambda r: (len(r.history), r.final_p_target))


def _feature_label(feature_idx: int) -> str:
    return f"x{feature_idx}"


def _draw_trajectory_panel(ax, bundle, disc_model, records: Sequence[TrajectoryRecord]) -> None:
    xx, yy, proba = _decision_grid(disc_model)
    ax.contourf(xx, yy, proba, levels=[0.0, 0.5, 1.0], colors=["#f5d5d0", "#d7ead8"], alpha=0.35)
    ax.contour(xx, yy, proba, levels=[0.5], colors="#1f1f1f", linewidths=1.2)

    y_train = bundle.y_train
    palette = np.array(["#b94a48", "#3b7f45"])
    ax.scatter(
        bundle.X_train[:, 0],
        bundle.X_train[:, 1],
        c=palette[y_train],
        s=12,
        alpha=0.32,
        linewidths=0,
        label="train",
    )

    feature_colors = {0: "#1f77b4", 1: "#d62728"}
    for rec in records:
        states = rec.states
        status_color = "#1f8f3a" if rec.flipped else "#bd2d2d"
        ax.scatter(states[0, 0], states[0, 1], marker="o", s=45, color=status_color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.scatter(states[-1, 0], states[-1, 1], marker="^", s=62, color=status_color, edgecolor="white", linewidth=0.7, zorder=6)
        if len(states) > 2:
            ax.scatter(states[1:-1, 0], states[1:-1, 1], marker=".", s=35, color=status_color, alpha=0.85, zorder=5)

        for step_idx, (start, end, hist) in enumerate(zip(states[:-1], states[1:], rec.history), start=1):
            feature_idx = int(hist[0])
            dx, dy = end - start
            ax.arrow(
                start[0],
                start[1],
                dx,
                dy,
                color=feature_colors[feature_idx],
                width=0.0025,
                head_width=0.018,
                head_length=0.018,
                length_includes_head=True,
                alpha=0.88,
                zorder=4,
            )
            mid = (start + end) / 2.0
            ax.text(
                mid[0],
                mid[1],
                str(step_idx),
                fontsize=7,
                color="#111111",
                ha="center",
                va="center",
                bbox={"boxstyle": "circle,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
                zorder=7,
            )

    ax.plot([], [], color=feature_colors[0], lw=2, label="feature 0 move")
    ax.plot([], [], color=feature_colors[1], lw=2, label="feature 1 move")
    ax.scatter([], [], marker="o", s=45, color="#1f8f3a", label="flipped")
    ax.scatter([], [], marker="o", s=45, color="#bd2d2d", label="stalled")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(bundle.feature_names[0])
    ax.set_ylabel(bundle.feature_names[1])
    ax.set_title("MOONS greedy trajectories")
    ax.legend(loc="upper right", fontsize=8, frameon=True)


def _distribution_curve(dist: dict) -> Tuple[np.ndarray, np.ndarray, float]:
    import torch

    if "logits" not in dist:
        classes = np.asarray(dist["classes"], dtype=float)
        proba = np.asarray(dist["proba"], dtype=float)[0]
        if classes.size == 1:
            return classes, proba, float(classes[0])
        return classes, proba, float(classes[int(np.argmax(proba))])

    logits = dist["logits"][0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    criterion = dist["criterion"]
    borders = getattr(criterion, "borders", None)
    if borders is not None:
        borders_np = borders.detach().cpu().numpy() if hasattr(borders, "detach") else np.asarray(borders)
        centers = (borders_np[:-1] + borders_np[1:]) / 2.0
        if centers.shape[0] != probs.shape[0]:
            centers = np.linspace(0.0, 1.0, probs.shape[0])
    else:
        centers = np.linspace(0.0, 1.0, probs.shape[0])
    mode = float(centers[int(np.argmax(probs))])
    return centers, probs, mode


def _axis_slice(
    disc_model,
    state: np.ndarray,
    target: int,
    feature_idx: int,
    grid: np.ndarray,
) -> np.ndarray:
    rows = np.repeat(state.reshape(1, -1), len(grid), axis=0)
    rows[:, feature_idx] = grid
    return disc_model.predict_proba(rows)[:, target]


def _draw_blocked_panel(
    ax,
    sampler,
    disc_model,
    blocked: TrajectoryRecord,
    feature_idx: int,
    step_state: np.ndarray,
) -> None:
    target = blocked.target
    dist = sampler.predictive_distribution(
        step_state.reshape(1, -1),
        target_col=feature_idx,
        fixed_target=target,
    )
    x_density, mass, mode = _distribution_curve(dist)
    grid = np.linspace(0.0, 1.0, 240)
    p_slice = _axis_slice(disc_model, step_state, target, feature_idx, grid)

    if mass.max() > 0:
        mass_plot = mass / mass.max()
    else:
        mass_plot = mass
    ax.plot(grid, p_slice, color="#202020", lw=1.4, label=f"LR p(target={target})")
    ax.axhline(0.5, color="#202020", lw=0.9, ls="--")
    ax.fill_between(grid, 0.0, 1.0, where=p_slice >= 0.5, color="#d7ead8", alpha=0.4, label="flip side")
    ax.plot(x_density, mass_plot, color="#6f3fb5", lw=1.8, label="TabPFN density")
    ax.fill_between(x_density, 0.0, mass_plot, color="#6f3fb5", alpha=0.22)
    ax.axvline(step_state[feature_idx], color="#666666", lw=1.0, ls=":", label="current")
    ax.axvline(mode, color="#6f3fb5", lw=1.2, ls="-.", label="mode")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    fixed_feature = 1 - feature_idx
    ax.set_title(
        f"Blocked slice: vary {_feature_label(feature_idx)}, "
        f"fix {_feature_label(fixed_feature)}={step_state[fixed_feature]:.2f}"
    )
    ax.set_xlabel(_feature_label(feature_idx))
    ax.set_ylabel("scaled probability / density")
    ax.legend(loc="best", fontsize=8)


def _write_figures(bundle, disc_model, records: Sequence[TrajectoryRecord], blocked: TrajectoryRecord, sampler) -> None:
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_trajectory_panel(ax, bundle, disc_model, records)
    fig.tight_layout()
    path = FIGURES_DIR / "moons_trajectories.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")

    step_state = blocked.states[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for feature_idx, ax in enumerate(axes):
        _draw_blocked_panel(ax, sampler, disc_model, blocked, feature_idx, step_state)
    fig.suptitle(
        f"Representative {'stalled' if not blocked.flipped else 'hard'} point "
        f"#{blocked.row_idx}: final p(target={blocked.target})={blocked.final_p_target:.3f}"
    )
    fig.tight_layout()
    path = FIGURES_DIR / "moons_blocked_slice.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}")


def _write_readme(
    records: Sequence[TrajectoryRecord],
    blocked: TrajectoryRecord,
    plotted_count: int,
) -> None:
    n_flipped = sum(r.flipped for r in records)
    readme = FIGURES_DIR / "README.md"
    readme.write_text(
        "# MOONS Greedy Trajectory Figures\n\n"
        "`moons_trajectories.png` shows the LR discriminator boundary, MOONS train "
        "scatter, and the per-step greedy counterfactual paths for near-boundary "
        "test points. Each arrow is axis-aligned because one feature is committed "
        "per step; blue arrows move feature 0 and red arrows move feature 1. Green "
        "markers reached the target class, red markers stalled before the flip.\n\n"
        "`moons_blocked_slice.png` zooms into a representative "
        f"{'stalled' if not blocked.flipped else 'hard'} point (test row "
        f"{blocked.row_idx}). The panels hold one coordinate fixed at the final "
        "state and vary the other coordinate, overlaying the LR target probability "
        "with the TabPFN class-conditional density. When the density mode remains "
        "outside the shaded flip side, a single-feature MAP commit lands on the "
        "wrong side of the boundary and the greedy path stalls.\n\n"
        f"Generated from {plotted_count} plotted near-boundary trajectories; "
        f"{len(records)} rows were evaluated including the fallback scan, with "
        f"{n_flipped} flipped and {len(records) - n_flipped} stalled.\n",
        encoding="utf-8",
    )
    print(f"Wrote {readme}")


def run(max_test: int, fallback_pool: int) -> None:
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator

    print("\n########## Exp8 MOONS trajectories ##########")
    print(
        f"selector={SELECTOR}, context=random_both@{MAX_CONTEXT}, budget={BUDGET}, "
        f"max_test={max_test}, fallback_pool={fallback_pool}"
    )

    bundle = load_dataset("moons")
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test, y_test = bundle.X_test, bundle.y_test
    actionable_idx, _immutable_idx = get_actionable_immutable("moons", bundle)

    disc_model = train_discriminator(X_train, y_train, X_test, y_test, "moons")
    ranked = _boundary_rank(disc_model, X_test)
    fallback_pool = min(max(fallback_pool, max_test), len(ranked))

    print("Loading TabPFN models ...")
    clf, reg = get_models(n_estimators=N_ESTIMATORS)

    records = _generate_records(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        ranked_indices=ranked,
        disc_model=disc_model,
        actionable_idx=actionable_idx,
        clf=clf,
        reg=reg,
        max_test=max_test,
        fallback_pool=fallback_pool,
    )
    blocked = _choose_blocked(records)
    blocked_sampler = _fit_sampler(clf, reg, X_train, y_train, blocked.target)

    plotted = records[:max_test]
    _write_figures(bundle, disc_model, plotted, blocked, blocked_sampler)
    _write_readme(records, blocked, plotted_count=len(plotted))


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 8: MOONS greedy trajectory plots")
    parser.add_argument("--max-test", type=int, default=30, help="Number of near-boundary trajectories to plot")
    parser.add_argument(
        "--fallback-pool",
        type=int,
        default=100,
        help="Bounded near-boundary pool to scan if the plotted rows contain no stalled point",
    )
    args = parser.parse_args()
    run(max_test=args.max_test, fallback_pool=args.fallback_pool)


if __name__ == "__main__":
    main()
