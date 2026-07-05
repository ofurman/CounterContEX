# Continuous-Feature Binning and Routing Audit

## Answer to the meeting questions

TabPFN's continuous path preserves ordering. Continuous targets are represented by `BarDistribution` / `FullSupportBarDistribution`: the distribution is a softmax over bars, and the constructor requires one-dimensional sorted borders (`src/tabpfn/architectures/base/bar_distribution.py:18`, `src/tabpfn/architectures/base/bar_distribution.py:20`, `src/tabpfn/architectures/base/bar_distribution.py:28`, `src/tabpfn/architectures/base/bar_distribution.py:46`). The CDF uses the ordered `borders`, `bucket_widths`, and `searchsorted`-style bucket lookup (`src/tabpfn/architectures/base/bar_distribution.py:82`, `src/tabpfn/architectures/base/bar_distribution.py:88`, `src/tabpfn/architectures/base/bar_distribution.py:261`, `src/tabpfn/architectures/base/bar_distribution.py:270`).

Counterfactual commits recover a scalar through the same continuous distribution, not by choosing an unordered class label. The greedy loop calls `sampler.sample_feature(..., fixed_target=y_target)` when committing a selected feature (`experiments/zeroshot_cf/greedy.py:198`, `experiments/zeroshot_cf/greedy.py:200`, `experiments/zeroshot_cf/greedy.py:204`). That calls `impute_masked`, which appends the fixed target class, calls `model.impute(..., t=self.temperature)`, then drops the appended Y column (`experiments/zeroshot_cf/sampler.py:392`, `experiments/zeroshot_cf/sampler.py:406`, `experiments/zeroshot_cf/sampler.py:420`). For regressor-routed columns, sampling uses `criterion.sample`, whose implementation calls `icdf`; `icdf` interpolates within the selected ordered bin using the left/right borders and probability mass (`src/tabpfn/architectures/base/bar_distribution.py:583`, `src/tabpfn/architectures/base/bar_distribution.py:590`, `src/tabpfn/architectures/base/bar_distribution.py:282`, `src/tabpfn/architectures/base/bar_distribution.py:285`).

The bin layout is not class-aware. The fitted distribution object owns one set of borders; class conditioning enters through the observed appended Y column and changes the predicted logits/softmax weights for the masked feature, not the border locations. The predictive-distribution helper appends `Y=fixed_target`, builds `conditional_idx` over observed columns, and returns `{"logits", "criterion"}` for regressor-routed features (`experiments/zeroshot_cf/sampler.py:558`, `experiments/zeroshot_cf/sampler.py:566`, `experiments/zeroshot_cf/sampler.py:571`, `experiments/zeroshot_cf/sampler.py:593`). The `criterion` carries the shared ordered bars; the class condition affects model output logits.

The real ordering leak is classifier routing of low-cardinality integer columns. The unsupervised wrapper re-infers categorical features on every `fit` (`.venv/lib/python3.13/site-packages/tabpfn_extensions/unsupervised/unsupervised.py:298`) and routes a column to the classifier when the column is in the inferred categorical set and the class count is supported (`.venv/lib/python3.13/site-packages/tabpfn_extensions/unsupervised/unsupervised.py:536`, `.venv/lib/python3.13/site-packages/tabpfn_extensions/unsupervised/unsupervised.py:548`, `.venv/lib/python3.13/site-packages/tabpfn_extensions/unsupervised/unsupervised.py:551`). The classifier path int-casts the feature target before fitting (`.venv/lib/python3.13/site-packages/tabpfn_extensions/unsupervised/unsupervised.py:640`), and the sampler documents why those `classes_` are not the real MinMax support (`experiments/zeroshot_cf/sampler.py:578`, `experiments/zeroshot_cf/sampler.py:580`, `experiments/zeroshot_cf/sampler.py:586`). That is the proximity-risk path audited by Exp9.

## HELOC Routing Inventory

At `knn_both@256` with default routing, 5 of 23 original HELOC columns route to the classifier head:

| idx | feature |
|---|---|
| 5 | `NumTrades60Ever2DerogPubRec` |
| 6 | `NumTrades90Ever2DerogPubRec` |
| 9 | `MaxDelq2PublicRecLast12M` |
| 10 | `MaxDelqEver` |
| 12 | `NumTradesOpeninLast12M` |

The remaining 18 columns route through the regressor/bar-distribution path.

## Override

Stage 8 adds `--force-numeric-cols` to the experiment runner. `none` preserves current behavior; `all` forces all original feature columns numeric; otherwise the value is a comma-separated list of feature names or indices. The implementation filters only the requested original feature columns out of the unsupervised wrapper's inferred categorical list, preserving the appended Y column and any non-forced explicit categoricals.
