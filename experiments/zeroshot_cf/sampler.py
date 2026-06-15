"""ConditionalDensitySampler: TabPFNUnsupervisedModel wrapper for CF generation.

Wraps TabPFNUnsupervisedModel (tabpfn-extensions) to handle:
- Class-conditional context selection (optionally filter context by target class)
- Y-as-appended-categorical-column trick for class conditioning
- Masked imputation for feature reconstruction (Exp 1) and CF generation (Exp 2)

Decision: all public methods accept/return numpy float64 arrays to match
the cel metrics contract, even though the inner model works in float32 tensors.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch

from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel


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
            Cap on context size. If set and context exceeds this, subsample
            deterministically using self.random_state.
        """
        rng = np.random.default_rng(self.random_state)

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

        Returns
        -------
        X_filled : ndarray of shape (m, d)
            X_query with masked columns filled; non-masked columns are
            byte-identical to the input.
        """
        if not self._fitted:
            raise RuntimeError("Call set_context() before impute_masked().")

        X = np.asarray(X_query, dtype=np.float32).copy()
        X[:, mask_cols] = np.nan

        if self.append_target:
            if fixed_target is None:
                raise ValueError("fixed_target required when append_target=True")
            target_col = np.full((len(X), 1), float(fixed_target), dtype=np.float32)
            X_aug = np.concatenate([X, target_col], axis=1)
        else:
            X_aug = X

        t0 = time.perf_counter()
        X_filled_tensor = self.model.impute(
            X_aug,
            t=self.temperature,
            n_permutations=self.n_permutations,
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
            Number of independent samples to draw per query row. When n_samples>1
            the sampler raises temperature to 1.0 for each additional draw so the
            samples explore the posterior rather than all collapsing to MAP.

        Returns
        -------
        samples : ndarray of shape (m,) if n_samples==1, else (n_samples, m)
        """
        if not self._fitted:
            raise RuntimeError("Call set_context() before sample_feature().")

        if n_samples == 1:
            X_filled = self.impute_masked(X_query, mask_cols=[target_col])
            return X_filled[:, target_col]

        # Multiple samples: first draw uses self.temperature; additional draws
        # use temperature=1.0 to explore the posterior.
        results = []
        original_temp = self.temperature

        # First sample at configured temperature
        X_filled = self.impute_masked(X_query, mask_cols=[target_col])
        results.append(X_filled[:, target_col])

        # Additional samples at t=1.0 for diversity
        self.temperature = 1.0
        try:
            for _ in range(n_samples - 1):
                X_filled = self.impute_masked(X_query, mask_cols=[target_col])
                results.append(X_filled[:, target_col])
        finally:
            self.temperature = original_temp

        return np.stack(results, axis=0)  # shape (n_samples, m)
