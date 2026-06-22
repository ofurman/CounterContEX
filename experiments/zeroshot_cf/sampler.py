"""ConditionalDensitySampler: TabPFNUnsupervisedModel wrapper for CF generation.

Wraps TabPFNUnsupervisedModel (tabpfn-extensions) to handle:
- Class-conditional context selection (optionally filter context by target class)
- Y-as-appended-categorical-column trick for class conditioning
- Masked imputation for feature reconstruction (Exp 1) and CF generation (Exp 2)

Decision: all public methods accept/return numpy float64 arrays to match
the cel metrics contract, even though the inner model works in float32 tensors.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel


def _knn_indices(X: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` rows of ``X`` closest to ``query``.

    Distance is plain Euclidean (L2) over the ``d`` columns of ``X``, computed
    in whatever feature space ``X`` is already in (callers pass MinMax-[0,1]
    features before the categorical-Y column is appended). The returned index
    array is **sorted ascending** for determinism, so callers can slice ``X``
    and a parallel ``y`` in lockstep.

    Args:
        X: ndarray of shape (n, d). The candidate pool.
        query: ndarray of shape (d,) or (1, d). The anchor point.
        k: number of neighbours to return. Assumes ``k <= n`` (callers gate on
            ``len(X) > max_context`` before calling).

    Returns:
        Sorted ndarray of ``k`` integer indices into ``X``.
    """
    q = np.asarray(query, dtype=X.dtype).reshape(-1)
    diff = X - q[None, :]
    dist2 = np.einsum("ij,ij->i", diff, diff)
    # argpartition gives the k smallest (unordered); sort for determinism.
    nearest = np.argpartition(dist2, k - 1)[:k]
    return np.sort(nearest)


def build_chain_dag(
    ordered_actionable: List[int],
    immutable_idx: List[int],
    y_idx: int,
) -> Dict[int, List[int]]:
    """Build a chain DAG for ordered autoregressive CF generation.

    Each actionable feature a_i gets parents: [y_idx] + immutable_idx + earlier actionables.
    Y and immutables are roots (not in the dict) — observed columns skipped by impute().

    Args:
        ordered_actionable: Actionable column indices in desired generation order.
        immutable_idx: Immutable column indices (always observed parents).
        y_idx: Augmented index of the appended Y column (= n_original_features).

    Returns:
        DAG dict compatible with TabPFNUnsupervisedModel.impute(dag=...).
    """
    dag: Dict[int, List[int]] = {}
    for i, a in enumerate(ordered_actionable):
        dag[a] = [y_idx] + list(immutable_idx) + list(ordered_actionable[:i])
    return dag


def mean_of_prediction(logits, criterion) -> np.ndarray:
    """Expected value of a bar-distribution prediction.

    ``logits``/``criterion`` are the pair returned by
    ``ConditionalDensitySampler.predictive_distribution`` for a regressor column
    (``criterion`` is a ``FullSupportBarDistribution``). Returns the per-row
    distribution mean as a numpy array. This is the bar-distribution *mean*, NOT
    the near-MAP (mode) value a ``t≈1e-9`` draw produces — they differ for
    skewed distributions.
    """
    mean = criterion.mean(logits)
    return mean.detach().cpu().numpy()


def symmetric_kl(logits_a, logits_b, criterion) -> np.ndarray:
    """Symmetric KL between two bar distributions that share the same borders.

    Both ``logits_a`` and ``logits_b`` are scored against the same ``criterion``
    (shared bucket borders), so the per-bucket probabilities are directly
    comparable. Returns ``KL(P_a || P_b) + KL(P_b || P_a)`` per row as a numpy
    array.
    """
    log_pa = torch.log_softmax(logits_a, dim=-1)
    log_pb = torch.log_softmax(logits_b, dim=-1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    kl_ab = (pa * (log_pa - log_pb)).sum(dim=-1)
    kl_ba = (pb * (log_pb - log_pa)).sum(dim=-1)
    return (kl_ab + kl_ba).detach().cpu().numpy()


def class_conditional_shift(dist_tgt: dict, dist_cur: dict) -> np.ndarray:
    """Per-row magnitude of the class-conditional shift between two predictive
    distributions of the *same* masked feature (one conditioned on ``Y=target``,
    the other on ``Y=current``).

    Uniform across TabPFN's two per-column routings; both branches return a
    value in ``[0, 1]`` so the class-divergence selector's argmax stays in
    comparable units regardless of whether a column is regressor- or
    classifier-routed:

    - **Regressor column** (``{"logits", "criterion"}``): absolute difference of
      the bar-distribution means, ``|E[x_j|Y=target] - E[x_j|Y=current]|``. The
      feature is MinMax-[0,1] so this is already in ``[0, 1]``.
    - **Classifier column** (``{"proba", "classes"}``): total-variation distance
      ``½·Σ_k |p_target,k - p_current,k|`` between the two class-probability
      vectors, aligned on the union of their ``classes_``. We deliberately do
      NOT compute an expected value ``Σ_k p_k·support_k`` here: ``density_`` fits
      the classifier on ``y.astype(int)``, so for MinMax-[0,1] features the
      ``classes_`` support collapses to ``{0}`` (a few columns to ``{0, 1}``) and
      carries no real feature-value information — the true 8-10 distinct MinMax
      levels are destroyed by the int-cast and are not recoverable from the
      fitted model. TV distance is the principled, in-[0,1] stand-in: it directly
      measures how much the class-conditional distribution moves between the two
      target conditions, which is exactly what the selector ranks. (Its scale is
      not identical to the regressor mean-shift, but both live in ``[0, 1]`` and
      the selector only needs a per-step argmax over candidate columns.)
    """
    if "logits" in dist_tgt:
        mean_tgt = mean_of_prediction(dist_tgt["logits"], dist_tgt["criterion"])
        mean_cur = mean_of_prediction(dist_cur["logits"], dist_cur["criterion"])
        return np.abs(mean_tgt - mean_cur)

    # Classifier column: TV distance over the union of the two class supports.
    classes = np.union1d(
        np.asarray(dist_tgt["classes"]), np.asarray(dist_cur["classes"])
    )

    def _align(dist: dict) -> np.ndarray:
        proba = np.atleast_2d(np.asarray(dist["proba"], dtype=float))
        src = np.asarray(dist["classes"])
        aligned = np.zeros((proba.shape[0], classes.shape[0]), dtype=float)
        col_for = {c: k for k, c in enumerate(classes)}
        for s, c in enumerate(src):
            aligned[:, col_for[c]] = proba[:, s]
        return aligned

    p_tgt = _align(dist_tgt)
    p_cur = _align(dist_cur)
    return 0.5 * np.abs(p_tgt - p_cur).sum(axis=1)


class ConditionalDensitySampler:
    """Conditional density sampler built on top of TabPFNUnsupervisedModel.

    Parameters
    ----------
    clf : TabPFNClassifier
        Pre-loaded classifier (from checkpoints.get_models).
    reg : TabPFNRegressor
        Pre-loaded regressor (from checkpoints.get_models).
    append_target : bool
        If True, append the target label as an extra categorical column during
        fit and imputation so that generation is conditioned on Y=target_class.
    n_permutations : int
        Monte-Carlo permutations for the imputation inner loop.
    temperature : float
        Sampling temperature. Near-0 (1e-9) = MAP/deterministic; 1.0 = full
        posterior sampling. Only affects numerical (regressor) columns.
    random_state : int
        Seed for numpy and torch RNGs to make subsampling deterministic.
    """

    def __init__(
        self,
        clf,
        reg,
        append_target: bool = False,
        n_permutations: int = 10,
        temperature: float = 1e-9,
        random_state: int = 0,
    ) -> None:
        self.clf = clf
        self.reg = reg
        self.append_target = append_target
        self.n_permutations = n_permutations
        self.temperature = temperature
        self.random_state = random_state

        self.model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
        self._n_original_features: Optional[int] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Context setup
    # ------------------------------------------------------------------

    def set_context(
        self,
        X_context: np.ndarray,
        y_context: Optional[np.ndarray] = None,
        target_class: Optional[int] = None,
        max_context: Optional[int] = None,
        selection: str = "random",
        query: Optional[np.ndarray] = None,
    ) -> "ConditionalDensitySampler":
        """Build the context and call model.fit().

        Parameters
        ----------
        X_context : ndarray of shape (n, d)
            Feature matrix used as the in-context training set.
        y_context : ndarray of shape (n,) or None
            Labels for each context row. Required when append_target=True or
            target_class is given.
        target_class : int or None
            If provided, filter X_context to rows whose label equals target_class
            before fitting (class-conditional context).
        max_context : int or None
            Cap on context size. If set and context exceeds this, select
            ``max_context`` rows according to ``selection``.
        selection : {"random", "knn"}
            How to subsample the (optionally class-filtered) pool down to
            ``max_context`` rows.
            - ``"random"`` (default): deterministic ``rng.choice`` subsample
              seeded by ``self.random_state``. This path is byte-identical to
              the pre-Stage-3 behaviour.
            - ``"knn"``: keep the ``max_context`` rows with smallest Euclidean
              distance to ``query`` over the original ``d`` features (the
              MinMax-[0,1] feature space, before the Y column is appended).
              Requires ``query`` (raises ``ValueError`` otherwise). Chosen
              indices are sorted for determinism.
        query : ndarray of shape (d,) or (1, d), optional
            The factual point used as the kNN anchor. Required when
            ``selection="knn"``; ignored when ``selection="random"``.

        Notes
        -----
        The four context strategies used by the ablation map onto the two
        orthogonal choices (class pool via ``target_class`` × selection):

        - ``random_target`` ≡ (``target_class=<t>``, ``selection="random"``)
        - ``random_both``   ≡ (``target_class=None``, ``selection="random"``)
        - ``knn_target``    ≡ (``target_class=<t>``, ``selection="knn"``)
        - ``knn_both``      ≡ (``target_class=None``, ``selection="knn"``)
        """
        if selection not in ("random", "knn"):
            raise ValueError(
                f"selection must be 'random' or 'knn', got {selection!r}"
            )
        if selection == "knn" and query is None:
            raise ValueError("query is required when selection='knn'")
        rng = np.random.default_rng(self.random_state)
        # Seed all RNGs so imputation permutations and posterior sampling are
        # reproducible. MPS float nondeterminism may persist on Apple Silicon.
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X_context, dtype=np.float32)
        y = np.asarray(y_context) if y_context is not None else None

        # Filter to target class if requested
        if target_class is not None:
            if y is None:
                raise ValueError("y_context required when target_class is given")
            mask = np.asarray(y) == target_class
            X = X[mask]
            y = y[mask]

        # Subsample if over max_context
        if max_context is not None and len(X) > max_context:
            if selection == "knn":
                idx = _knn_indices(X, query, max_context)
            else:
                idx = rng.choice(len(X), size=max_context, replace=False)
                idx.sort()
            X = X[idx]
            if y is not None:
                y = y[idx]

        self._n_original_features = X.shape[1]

        # Optionally append Y as the last (categorical) column
        if self.append_target:
            if y is None:
                raise ValueError("y_context required when append_target=True")
            y_col = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            X_aug = np.concatenate([X, y_col], axis=1)
            last_idx = X_aug.shape[1] - 1
            self.model.set_categorical_features([last_idx])
        else:
            X_aug = X

        self.model.fit(X_aug)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Masked imputation
    # ------------------------------------------------------------------

    def impute_masked(
        self,
        X_query: np.ndarray,
        mask_cols: List[int],
        fixed_target: Optional[int] = None,
        dag: Optional[Dict[int, List[int]]] = None,
    ) -> np.ndarray:
        """Fill masked columns via conditional density estimation.

        Parameters
        ----------
        X_query : ndarray of shape (m, d)
            Query points. The original (non-masked) feature values are preserved.
        mask_cols : list of int
            Column indices to NaN-mask (the values to be imputed).
        fixed_target : int or None
            Target class to condition on. Required (and only used) when
            append_target=True. The appended Y column is set to this value
            (observed, not NaN) so imputation is class-conditional.
        dag : dict[int, list[int]] or None
            Optional DAG in augmented index space (Y appended). When provided,
            imputation uses the DAG path (condition_on_all_features=False) and
            each actionable conditions only on its declared parents. When None
            (default), the standard random-permutation path is used.

        Returns
        -------
        X_filled : ndarray of shape (m, d)
            X_query with masked columns filled; non-masked columns are
            byte-identical to the input.
        """
        if not self._fitted:
            raise RuntimeError("Call set_context() before impute_masked().")

        # Re-seed before each impute call so n_permutations draws are reproducible.
        # MPS float nondeterminism may persist on Apple Silicon.
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X_query, dtype=np.float32).copy()
        X[:, mask_cols] = np.nan

        if self.append_target:
            if fixed_target is None:
                raise ValueError("fixed_target required when append_target=True")
            target_col = np.full((len(X), 1), float(fixed_target), dtype=np.float32)
            X_aug = np.concatenate([X, target_col], axis=1)
        else:
            X_aug = X

        if dag is not None:
            assert all(idx < X_aug.shape[1] for k, v in dag.items() for idx in [k] + v), (
                f"DAG index out of bounds for augmented matrix of shape {X_aug.shape}"
            )

        t0 = time.perf_counter()
        X_filled_tensor = self.model.impute(
            X_aug,
            t=self.temperature,
            n_permutations=self.n_permutations,
            dag=dag,
        )
        elapsed = time.perf_counter() - t0
        print(f"[sampler] impute_masked: {len(X_query)} rows, "
              f"{len(mask_cols)} masked cols, "
              f"t={self.temperature}, n_perm={self.n_permutations} "
              f"→ {elapsed:.2f}s")

        X_filled = X_filled_tensor.cpu().numpy().astype(np.float64)

        # Drop the appended target column if it was added
        if self.append_target:
            X_filled = X_filled[:, : self._n_original_features]

        # Restore non-masked columns exactly from the original query
        original = np.asarray(X_query, dtype=np.float64)
        non_masked = [c for c in range(original.shape[1]) if c not in mask_cols]
        X_filled[:, non_masked] = original[:, non_masked]

        return X_filled

    # ------------------------------------------------------------------
    # Single-feature sampling (Experiment 1)
    # ------------------------------------------------------------------

    def sample_feature(
        self,
        X_query: np.ndarray,
        target_col: int,
        n_samples: int = 1,
        sample_temperature: Optional[float] = None,
        fixed_target: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct a single feature column via conditional density estimation.

        Parameters
        ----------
        X_query : ndarray of shape (m, d)
            Query rows; target_col values are ignored (replaced by NaN before
            imputation).
        target_col : int
            Column index to reconstruct.
        n_samples : int
            Number of independent samples to draw per query row. Returns shape
            (m,) when n_samples==1, else (n_samples, m).
        sample_temperature : float or None
            Temperature to use for all draws in this call. Overrides
            self.temperature when provided. Use 1e-9 for near-MAP point
            estimates and 1.0 for posterior exploration. When None, falls
            back to self.temperature.
        fixed_target : int or None
            Target class to condition on, forwarded to ``impute_masked``.
            Required (and only used) when ``append_target=True`` — under that
            regime the single masked column is drawn class-conditionally,
            ``p(x_j | x_{-j}, Y=fixed_target)``. Default None preserves the
            existing ``append_target=False`` (class-agnostic) behaviour.

        Returns
        -------
        samples : ndarray of shape (m,) if n_samples==1, else (n_samples, m)
        """
        if not self._fitted:
            raise RuntimeError("Call set_context() before sample_feature().")

        effective_temp = sample_temperature if sample_temperature is not None else self.temperature
        original_temp = self.temperature
        self.temperature = effective_temp
        try:
            if n_samples == 1:
                X_filled = self.impute_masked(
                    X_query, mask_cols=[target_col], fixed_target=fixed_target
                )
                return X_filled[:, target_col]

            # Each sample must use a distinct seed so draws are truly independent.
            # impute_masked always re-seeds from self.random_state; we offset it
            # per iteration to avoid all n_samples producing the same value.
            original_rs = self.random_state
            results = []
            try:
                for i in range(n_samples):
                    self.random_state = original_rs + i
                    X_filled = self.impute_masked(
                        X_query, mask_cols=[target_col], fixed_target=fixed_target
                    )
                    results.append(X_filled[:, target_col])
            finally:
                self.random_state = original_rs
        finally:
            self.temperature = original_temp

        return np.stack(results, axis=0)  # shape (n_samples, m)

    # ------------------------------------------------------------------
    # Single-feature predictive distribution (Experiment 4, Strategy 2)
    # ------------------------------------------------------------------

    def predictive_distribution(
        self,
        X_query: np.ndarray,
        target_col: int,
        fixed_target: Optional[int] = None,
    ):
        """Return the conditional predictive distribution of a single masked feature.

        Unlike ``sample_feature`` this does NOT sample — it returns the raw
        per-row distribution of ``x_{target_col} | x_{-target_col}, Y=fixed_target``
        so callers (e.g. the class-divergence selector) can compute statistics
        (mean, KL) without drawing.

        Mirrors the augmented-matrix construction of ``impute_masked`` (NaN-mask
        ``target_col``, append the ``Y=fixed_target`` categorical column, same
        RNG re-seeding) and then calls the underlying model's conditional-density
        primitive ``density_`` directly — there is no public "impute minus the
        sample" path.

        Parameters
        ----------
        X_query : ndarray of shape (m, d)
            Query rows; ``target_col`` is masked (its value is ignored).
        target_col : int
            Column index whose conditional distribution is requested.
        fixed_target : int or None
            Class to condition on. Required when ``append_target=True``.

        Returns
        -------
        For a regressor (numerical) column — the case for all HELOC/MOONS
        features — a dict ``{"logits": Tensor, "criterion": FullSupportBarDistribution}``
        describing the per-row bar distribution. For a classifier (categorical)
        column — which DOES occur on HELOC, whose low-cardinality integer
        features ``infer_categorical_features`` routes to the classifier head —
        a dict ``{"proba": ndarray, "classes": ndarray}`` of class probabilities
        and the (int-cast) class labels. Use ``class_conditional_shift`` to get a
        comparable [0,1] divergence across both shapes (see that helper for why
        the classifier branch can't yield a true expected feature value).
        """
        if not self._fitted:
            raise RuntimeError("Call set_context() before predictive_distribution().")

        # Re-seed exactly as impute_masked does so the fit is reproducible.
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        X = np.asarray(X_query, dtype=np.float32).copy()
        X[:, target_col] = np.nan

        if self.append_target:
            if fixed_target is None:
                raise ValueError("fixed_target required when append_target=True")
            target_col_arr = np.full((len(X), 1), float(fixed_target), dtype=np.float32)
            X_aug = np.concatenate([X, target_col_arr], axis=1)
        else:
            X_aug = X

        # conditional_idx = every augmented column except the masked target_col
        # (all observed features + the appended Y column when present).
        conditional_idx = [c for c in range(X_aug.shape[1]) if c != target_col]

        X_predict_t = torch.tensor(X_aug, dtype=torch.float32)
        model_j, X_predict, _ = self.model.density_(
            X_predict_t,
            self.model.X_,
            conditional_idx,
            target_col,
        )

        if self.model.use_classifier_(target_col, self.model.X_[:, target_col]):
            proba = model_j.predict_proba(X_predict.numpy())
            # ``classes_`` are the *int-cast* feature labels density_ fit on
            # (``y_fit.astype(int)``), NOT the real MinMax-[0,1] feature values:
            # for MinMax features almost everything collapses to class 0, so the
            # support is unusable as an expected-value grid. We still return it so
            # callers can column-align two proba vectors (see ``class_divergence``
            # in greedy.py, which uses total-variation distance, not a mean).
            return {
                "proba": np.asarray(proba),
                "classes": np.asarray(model_j.classes_),
            }

        # Pass X_predict as the tensor returned by density_ (matches the verified
        # internal caller outliers_single_permutation_); do NOT .numpy() it.
        out = model_j.predict(X_predict, output_type="full")
        return {"logits": out["logits"], "criterion": out["criterion"]}
