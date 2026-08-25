"""Offline checkpoint staging and loading helpers for TabPFN v2.

We use TabPFN model version "v2" throughout because:
- v2 models are freely downloadable from HuggingFace without license acceptance.
- v3 (default) requires a TABPFN_TOKEN license check that is not available offline.
- v2 still provides the full conditional-density API needed for CF generation.
"""

import os
from pathlib import Path

# Repo-local checkpoint cache: experiments/zeroshot_cf/models/
_DEFAULT_LOCAL_CACHE = Path(__file__).parent / "models"

# Allow override via env var; otherwise use the local cache
TABPFN_DEVICE = os.environ.get("TABPFN_DEVICE", "auto")
TABPFN_LOCAL_CACHE = Path(
    os.environ.get("TABPFN_LOCAL_CACHE", str(_DEFAULT_LOCAL_CACHE))
)

# v2 checkpoint filenames (must match model_loading.py default_filename values)
_V2_CLF_FILENAME = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
_V2_REG_FILENAME = "tabpfn-v2-regressor.ckpt"

# Use v2 to avoid the license gate on newer versions
_TABPFN_MODEL_VERSION = "v2"


def _set_env(cache_dir: Path) -> None:
    """Configure TabPFN env vars before importing models."""
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(cache_dir)
    os.environ["TABPFN_MODEL_VERSION"] = _TABPFN_MODEL_VERSION


def stage_checkpoints(cache_dir: Path | None = None) -> Path:
    """Download TabPFN v2 checkpoints into the local cache (requires network once).

    Call this once on a machine with internet access. Subsequent offline runs
    will find the files in cache_dir and skip the download.

    Returns the cache directory path.
    """
    cache_dir = cache_dir or TABPFN_LOCAL_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_env(cache_dir)

    from tabpfn import TabPFNClassifier, TabPFNRegressor

    print(f"Staging TabPFN v2 checkpoints in: {cache_dir}")
    dummy_X = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]
    dummy_y = [0.0, 1.0, 0.5, 0.2, 0.8]
    dummy_y_cls = [0, 1, 0, 1, 0]

    import numpy as np
    X_arr = np.array(dummy_X, dtype=float)
    reg = TabPFNRegressor(n_estimators=1, device="cpu")
    reg.fit(X_arr, dummy_y)

    clf = TabPFNClassifier(n_estimators=1, device="cpu")
    clf.fit(X_arr, dummy_y_cls)

    print("Checkpoints staged successfully.")
    return cache_dir


def get_models(
    device: str = "auto",
    n_estimators: int = 4,
    cache_dir: Path | None = None,
) -> tuple:
    """Return a (TabPFNClassifier, TabPFNRegressor) pair loaded from the local cache.

    Environment overrides:
      TABPFN_DEVICE      — device string passed to both models ("auto", "cpu", "mps")
      TABPFN_LOCAL_CACHE — path to directory containing .ckpt files
    """
    cache_dir = cache_dir or TABPFN_LOCAL_CACHE
    device = os.environ.get("TABPFN_DEVICE", device)

    # Point TabPFN at local cache before importing so it never reaches the network
    _set_env(cache_dir)

    from tabpfn import TabPFNClassifier, TabPFNRegressor

    # Pass explicit model_path so version is resolved from the filename, not from
    # settings.tabpfn.model_version which defaults to V3 at import time and is
    # not reliably updated by the TABPFN_MODEL_VERSION env var after import.
    clf_path = cache_dir / _V2_CLF_FILENAME
    reg_path = cache_dir / _V2_REG_FILENAME
    clf = TabPFNClassifier(n_estimators=n_estimators, device=device, model_path=clf_path)
    reg = TabPFNRegressor(n_estimators=n_estimators, device=device, model_path=reg_path)
    return clf, reg
