"""Smoke test: offline conditional-density sampling with TabPFN v2.

Run with networking disabled to prove no download occurs once checkpoints are staged:
    HF_HUB_OFFLINE=1 python experiments/zeroshot_cf/smoke_test.py

Or from the repo root:
    HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/smoke_test.py
"""

import sys
import os
from pathlib import Path

# Ensure repo root is on sys.path for editable imports
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from experiments.zeroshot_cf.checkpoints import stage_checkpoints, get_models, TABPFN_LOCAL_CACHE

# ---- 1. Ensure checkpoints are staged ----
ckpt_files = list(TABPFN_LOCAL_CACHE.glob("*.ckpt"))
if not ckpt_files:
    print("Checkpoints not found locally — staging now (network required once).")
    stage_checkpoints()
else:
    print(f"Checkpoints already staged in: {TABPFN_LOCAL_CACHE}")
    for f in ckpt_files:
        print(f"  {f.name}")

# ---- 2. Load models offline ----
clf, reg = get_models(n_estimators=2)
print(f"Classifier: {clf.__class__.__name__} | Regressor: {reg.__class__.__name__}")

# ---- 3. Toy data: 20 points, 3 features; predict feature 2 from features 0,1 ----
rng = np.random.default_rng(42)
X = rng.standard_normal((20, 3)).astype(np.float32)
X_train, y_train = X[:15, :2], X[:15, 2]
X_query = X[15:, :2]

reg.fit(X_train, y_train)
out = reg.predict(X_query, output_type="full")

assert "criterion" in out, "'criterion' key missing from output_type='full'"
assert "logits" in out, "'logits' key missing from output_type='full'"

criterion = out["criterion"]
logits = out["logits"]

samples = criterion.sample(logits, t=1.0)

print(f"Samples shape: {samples.shape}  |  dtype: {samples.dtype}")
assert samples.shape[0] == X_query.shape[0], "sample count must match query count"

import torch
assert torch.all(torch.isfinite(samples)), "samples must be finite"

print("\n=== Smoke test PASSED ===")
print(f"  Sampled value (first): {samples[0].item():.4f}")
