#  Copyright (c) Prior Labs GmbH 2026.

"""Offline checkpoint staging and loading helpers for TabPFN.

The experiments default to TabPFN v2 because those weights can be staged without
the newer license gate. Set ``TABPFN_MODEL_VERSION=v3`` to run against local v3
weights instead.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo-local checkpoint cache: experiments/zeroshot_cf/models/
_EXPERIMENT_LOCAL_CACHE = Path(__file__).parent / "models"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_MODELS_CACHE = _REPO_ROOT / "models"

_DEFAULT_MODEL_VERSION = "v2"


def _selected_model_version() -> str:
    version = os.environ.get("TABPFN_MODEL_VERSION", _DEFAULT_MODEL_VERSION)
    if version not in _MODEL_FILENAMES:
        supported = ", ".join(sorted(_MODEL_FILENAMES))
        raise ValueError(
            f"Unsupported TABPFN_MODEL_VERSION={version!r}; use {supported}"
        )
    return version


def _default_local_cache() -> Path:
    if (
        os.environ.get("TABPFN_MODEL_VERSION", _DEFAULT_MODEL_VERSION) == "v3"
        and _REPO_MODELS_CACHE.exists()
    ):
        return _REPO_MODELS_CACHE
    return _EXPERIMENT_LOCAL_CACHE


# Allow override via env var; otherwise use the local cache
TABPFN_DEVICE = os.environ.get("TABPFN_DEVICE", "auto")
TABPFN_LOCAL_CACHE = Path(
    os.environ.get("TABPFN_LOCAL_CACHE", str(_default_local_cache()))
)

# Checkpoint filenames. The first entry is the official/default name used if the
# model is not already present; additional entries support locally renamed files.
_MODEL_FILENAMES = {
    "v2": {
        "classifier": ("tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",),
        "regressor": ("tabpfn-v2-regressor.ckpt",),
    },
    "v3": {
        "classifier": (
            "tabpfn-v3-classifier-v3_default.ckpt",
            "tabpfn-v3-classifier-v3_20260417_binary.ckpt",
            "tabpfn3_binary.ckpt",
        ),
        "regressor": (
            "tabpfn-v3-regressor-v3_default.ckpt",
            "tabpfn3_regressor.ckpt",
        ),
    },
}


def _set_env(cache_dir: Path, version: str | None = None) -> None:
    """Configure TabPFN env vars before importing models."""
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(cache_dir)
    os.environ["TABPFN_MODEL_VERSION"] = version or _selected_model_version()


def _checkpoint_path(cache_dir: Path, version: str, model_type: str) -> Path:
    candidates = _MODEL_FILENAMES[version][model_type]
    for filename in candidates:
        path = cache_dir / filename
        if path.exists():
            return path
    return cache_dir / candidates[0]


def stage_checkpoints(cache_dir: Path | None = None) -> Path:
    """Download TabPFN checkpoints into the local cache (requires network once).

    Call this once on a machine with internet access. Subsequent offline runs
    will find the files in cache_dir and skip the download.

    Returns the cache directory path.
    """
    version = _selected_model_version()
    cache_dir = cache_dir or TABPFN_LOCAL_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_env(cache_dir, version)

    from tabpfn import TabPFNClassifier, TabPFNRegressor  # noqa: PLC0415

    print(f"Staging TabPFN {version} checkpoints in: {cache_dir}")  # noqa: T201
    dummy_X = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]
    dummy_y = [0.0, 1.0, 0.5, 0.2, 0.8]
    dummy_y_cls = [0, 1, 0, 1, 0]

    import numpy as np  # noqa: PLC0415

    X_arr = np.array(dummy_X, dtype=float)
    reg = TabPFNRegressor(n_estimators=1, device="cpu")
    reg.fit(X_arr, dummy_y)

    clf = TabPFNClassifier(n_estimators=1, device="cpu")
    clf.fit(X_arr, dummy_y_cls)

    print("Checkpoints staged successfully.")  # noqa: T201
    return cache_dir


def get_models(
    device: str = "auto",
    n_estimators: int = 4,
    cache_dir: Path | None = None,
) -> tuple:
    """Return a (TabPFNClassifier, TabPFNRegressor) pair loaded from the local cache.

    Environment overrides:
      TABPFN_DEVICE      — device string passed to both models ("auto", "cpu", "mps")
      TABPFN_MODEL_VERSION — model version to load ("v2" by default, or "v3")
      TABPFN_LOCAL_CACHE — path to directory containing .ckpt files
    """
    version = _selected_model_version()
    cache_dir = cache_dir or TABPFN_LOCAL_CACHE
    device = os.environ.get("TABPFN_DEVICE", device)

    # Point TabPFN at local cache before importing so it never reaches the network
    _set_env(cache_dir, version)

    from tabpfn import TabPFNClassifier, TabPFNRegressor  # noqa: PLC0415

    # Pass explicit model_path so offline runs use the staged files and never
    # depend on settings.tabpfn.model_version after import time.
    clf_path = _checkpoint_path(cache_dir, version, "classifier")
    reg_path = _checkpoint_path(cache_dir, version, "regressor")
    clf = TabPFNClassifier(
        n_estimators=n_estimators,
        device=device,
        model_path=clf_path,
    )
    reg = TabPFNRegressor(
        n_estimators=n_estimators,
        device=device,
        model_path=reg_path,
    )
    return clf, reg


def get_v3_models(
    device: str = "auto",
    n_estimators: int = 4,
    cache_dir: Path | None = None,
) -> tuple:
    """Load the local TabPFNv3 pair used by the Athena comparison runs.

    This preserves the branch's flexible filename resolution while making the
    comparison explicitly v3 even if the surrounding shell has no version set.
    """
    cache_dir = Path(
        cache_dir
        or os.environ.get("TABPFN_V3_LOCAL_CACHE")
        or os.environ.get("TABPFN_LOCAL_CACHE", str(_REPO_MODELS_CACHE))
    )
    clf_path = _checkpoint_path(cache_dir, "v3", "classifier")
    reg_path = _checkpoint_path(cache_dir, "v3", "regressor")
    missing = [path for path in (clf_path, reg_path) if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "TabPFNv3 checkpoint(s) required for the Athena-matched comparison "
            f"are missing:\n{missing_text}"
        )

    os.environ["TABPFN_MODEL_VERSION"] = "v3"
    return get_models(device=device, n_estimators=n_estimators, cache_dir=cache_dir)
