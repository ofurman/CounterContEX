# API Reference — concrete call patterns

Captured during planning (2026-06-15) so the executing agent doesn't re-research. Verify signatures against the installed versions before relying on them.

---

## 1. Local TabPFN core (this repo, v8.0.8)

Files: `src/tabpfn/regressor.py`, `src/tabpfn/classifier.py`, `src/tabpfn/architectures/base/bar_distribution.py`, `src/tabpfn/model_loading.py`.

### Conditional density + sampling (the key path)
```python
from tabpfn import TabPFNRegressor
reg = TabPFNRegressor(n_estimators=4, device="auto")   # MPS on Apple Silicon
reg.fit(X_features, y_target)                            # any column can be the target
out = reg.predict(X_query, output_type="full")
# out: {"mean","median","mode","quantiles","criterion","logits"}
logits    = out["logits"]        # torch.Tensor (batch, num_bars)
criterion = out["criterion"]     # FullSupportBarDistribution
samples   = criterion.sample(logits, t=1.0)   # temperature t; t->0 = near-MAP
logdens   = -criterion.forward(logits, y)      # log-density (negative NLL) at y
```
`output_type` ∈ {`"mean","median","mode","quantiles","full","main"`}. `sample()` iterates over the batch (not vectorized) — fine for our sizes.

### Offline checkpoints
- Env var `TABPFN_MODEL_CACHE_DIR` sets cache location (default macOS `~/.cache/TabPFN`).
- `TabPFNRegressor(model_path="/abs/path/model.ckpt")` loads a local file with **no download** if present.
- First construction with network downloads `tabpfn-v2-classifier.ckpt` / `tabpfn-v2-regressor.ckpt` from HuggingFace (`Prior-Labs/TabPFN-v2-*`). Pre-stage once, then `HF_HUB_OFFLINE=1`.
- macOS MPS tuning: `TABPFN_MPS_MEMORY_FRACTION` (default 0.7). `device="auto"` → MPS on Apple Silicon, else CPU.

---

## 2. tabpfn-extensions — unsupervised module

Package: `tabpfn_extensions.unsupervised`; class `TabPFNUnsupervisedModel`. Examples: `examples/unsupervised/{generate_data,generate_data_following_dag,density_estimation_outlier_detection,imputation}.py`.

```python
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

clf = TabPFNClassifier(n_estimators=4); reg = TabPFNRegressor(n_estimators=4)
model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
model.set_categorical_features([y_col_idx])   # BEFORE fit, to mark appended Y categorical
model.fit(X_context)                           # stores the WHOLE matrix as conditioning set

# Imputation: fills ONLY NaN cells, conditioned on observed cells in the same row
X_q = X_query.copy(); X_q[:, mask_cols] = np.nan
X_filled = model.impute(X_q, t=1e-9, n_permutations=10)   # returns float32 torch tensor

# Generation (all-NaN matrix, autoregressive)
synth = model.generate_synthetic_data(n_samples=100, t=1.0, n_permutations=3)

# Joint log-density (outlier scoring)
logp = model.outliers(X, n_permutations=10)    # lower = more anomalous
```

Key facts:
- **No native Y-conditioning.** Workaround: append Y as last column, mark categorical via `set_categorical_features`, fit on augmented matrix, fix Y=target (observed) at impute time, NaN-mask the feature columns to generate. → class-conditional `p(features | observed, Y=target)`.
- `impute` default `t=1e-9` (near-MAP); `generate` default `t=1.0`. Temperature only affects **numerical/regressor** columns; categorical columns sample from `predict_proba`.
- Conditioning scope: `dag=None` → each masked col conditions on **all other observed cols**; `condition_on_all_features=False` (used internally by generate) → autoregressive prefix; `dag=dict[child→[parents]]` → parent-only, topological order.
- Internally re-`fit`s a fresh TabPFN per column/permutation — the expensive inner loop. Keep `n_estimators`, `n_permutations`, and context size modest.
- `FAST_TEST_MODE=1` clamps samples=5, permutations=1 (handy for smoke tests).
- Install: `pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"`. Works offline with local `tabpfn` (not `tabpfn-client`).

Sampling internals (reuse pattern):
```python
pred = reg.predict(X_predict, output_type="full")
pred_sampled = pred["criterion"].sample(torch.as_tensor(pred["logits"]), t=t)
```

---

## 3. counterfactuals repo (package `cel`)

Repo: `github.com/ofurman/counterfactuals`. **Package import is `cel`**, not `counterfactuals`. Data: `data/*.csv`; configs: `config/datasets/*.yaml`.

### Datasets
```python
from cel.datasets import FileDataset, MethodDataset
from cel.preprocessing import PreprocessingPipeline, MinMaxScalingStep, TorchDataTypeStep
fd = FileDataset(config_path="config/datasets/heloc.yaml")
ds = MethodDataset(fd, PreprocessingPipeline([("minmax", MinMaxScalingStep()),
                                              ("torch_dtype", TorchDataTypeStep())]))
# ds.X_train / X_test / y_train / y_test, ds.numerical_features_indices,
# ds.categorical_features_indices, ds.features, ds.actionable_features
```
- Split: 80/20 stratified, `random_state=42`. MinMax→[0,1] on continuous, fit on train; `inverse_transform` available.
- **HELOC**: 23 continuous features, target `RiskPerformance` {Bad:0, Good:1}. Feature order (idx 0..22): `ExternalRiskEstimate, MSinceOldestTradeOpen, MSinceMostRecentTradeOpen, AverageMInFile, NumSatisfactoryTrades, NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, PercentTradesNeverDelq, MSinceMostRecentDelq, MaxDelq2PublicRecLast12M, MaxDelqEver, NumTotalTrades, NumTradesOpeninLast12M, PercentInstallTrades, MSinceMostRecentInqexcl7days, NumInqLast6M, NumInqLast6Mexcl7days, NetFractionRevolvingBurden, NetFractionInstallBurden, NumRevolvingTradesWBalance, NumInstallTradesWBalance, NumBank2NatlTradesWHighUtilization, PercentTradesWBalance`. Config marks **all 23 actionable** → we define immutables ourselves (see index Decision #2).
- **MOONS**: features cols `"0"`,`"1"`, target col `"2"`; `samples_keep: 1000`. Both actionable.

### Metrics (registry path / `evaluate_cf`)
```python
from cel.metrics.metrics import evaluate_cf
res = evaluate_cf(disc_model=disc, gen_model=None, X_cf=X_cf,
                  model_returned=np.ones(len(X_cf), bool),
                  continuous_features=ds.numerical_features_indices,
                  categorical_features=[], X_train=X_train, y_train=y_train,
                  X_test=X_test, y_test=y_test, median_log_prob=None,
                  y_target=y_target)   # dict; metrics with unmet inputs are skipped
```
Our subset and where they live:
- `validity` (`cel/metrics/basic_metrics.py`): `(disc.predict(X_cf) != y_test).mean()`. Needs `disc_model.predict()` + `.eval()`.
- `lof_scores_cf` (`cel/metrics/plausibility.py`): `LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_train)`, `(-score_samples(X_cf)).mean()`. **Lower = more plausible.**
- `sparsity` (`basic_metrics.py`): `(X_test != X_cf).mean()`.
- `actionability` (`basic_metrics.py`): `np.all(X_test==X_cf, axis=1).mean()` — **mislabeled**; = fraction of unchanged CFs, NOT immutable compliance. Compute our own `true_actionability` (immutable cols unchanged).
- `proximity_l2_jaccard` (`cel/metrics/distance.py`): for all-continuous data reduces to **mean per-instance Euclidean (L2)** over valid CFs (`y_cf_pred == y_target`).

### Discriminator contract
`disc_model` must expose `.predict(X_np) -> array` and `.eval()` (no-op ok). cel classifiers in `cel/models/` (logistic_regression, MLP, NODE). `evaluate_cf` defaults `y_target = abs(1 - y_test)` if not passed.

### cel baseline numbers
TODO (fill during execution): read HELOC/MOONS validity / proximity / sparsity / LOF for PPCEF, DiCE, etc. from the cel repo's reported results or by running its pipelines, for side-by-side comparison in REPORT.md.
