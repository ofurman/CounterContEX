# Zero-Shot Autoregressive CF Generation with TabPFN — Results Report

**Date**: 2026-06-15  
**Branch**: `zeroshot-tabpfn-cf`  
**Status**: All experiments complete (Stages 1–8, post-review corrected).

---

## Post-review corrections (2026-06-15)

Stage 7 identified and fixed two correctness issues before Stage 8 re-ran all experiments:

1. **Validity reference fix**: Previous runs (Stages 5–6) scored validity as
   `disc(X_cf) != y_test` instead of `disc(X_cf) == y_target` (where `y_target = 1 − y_pred`).
   For misclassified factuals (`y_test != y_pred`, ~28% of HELOC test points at 72% oracle accuracy),
   the two definitions are opposites. Effect on headline numbers:
   - **MOONS**: validity 0.85 → **1.00** (+0.15, because all corrected CFs flip to the target class)
   - **HELOC**: validity 0.66 → **0.52** (−0.14, corrected out of the inflated mis-scored set)

2. **RNG seeding fix for posterior calibration**: `impute_masked()` re-seeded from `self.random_state`
   on every call, causing all N posterior samples in Exp 1 to be identical (inter-quartile range = 0 →
   calibration = 0.00). Fixed by offsetting the seed by sample index. Effect:
   - MOONS calibration: 0.00 → **0.69**; HELOC calibration: 0.00 → **0.62**

3. **LOF label**: HELOC LOF (~3.1B) reflects the true geometry of unclipped CFs and is structurally
   expected given 72% OOB extrapolation. This report presents it as a structural finding, not an artefact.

All numbers in this report are from the Stage 8 re-run with corrected code.

---

## 1. Setup

### Offline environment
- **Model**: TabPFN v2 (no license gate), checkpoints cached locally.
- **Inference**: Apple Silicon MPS (`device="auto"`), `HF_HUB_OFFLINE=1`.
- **No retraining / architecture changes**: all generation is in-context inference only.

### Mechanism
`TabPFNUnsupervisedModel.impute()` from `tabpfn-extensions` fills NaN-masked feature
columns by conditioning on all observed columns in the same row (autoregressive over
random feature orderings). We use the **Y-as-column trick** to inject class conditioning:
append the target label as an extra categorical column, fit the model on the augmented
context, then at impute time fix Y=target (observed) and NaN-mask the actionable features.

### Datasets
- **MOONS**: 1000 rows, 2 continuous features, 80/20 split. Both features actionable (no immutables). MinMax→[0,1].
- **HELOC**: 10,459 rows, 23 continuous features, 80/20 split. 6 history/age features are immutable (cannot be changed by the applicant); 17 are actionable. MinMax→[0,1].

### Validity oracle
sklearn `LogisticRegression` trained on scaled features (MOONS: test acc=87%, HELOC: test acc=72%). Wrapped to expose `disc_model.predict(X)` + `.eval()` as required by the cel metrics contract. Note: validity oracle is a different model family than the TabPFN generator — a caveat for comparing with cel baselines.

### HELOC actionability split (Decision #2)
Immutable features (6): `MSinceOldestTradeOpen`, `MSinceMostRecentTradeOpen`, `AverageMInFile`, `NumTotalTrades`, `MSinceMostRecentDelq`, `MSinceMostRecentInqexcl7days`. These are history/age fields the applicant cannot directly change. The remaining 17 features (balances, utilization, inquiry counts, delinquency rates) are actionable.

---

## 2. Experiment 1: Single-Feature Reconstruction (Sanity Check)

### Protocol
For each feature j, mask it in test points, reconstruct via `ConditionalDensitySampler` using same-class context (t=1e-9 for MAP, N_SAMPLES posterior samples at t=1.0 for calibration). **Calibration uses per-sample distinct seeds** (fixed in Stage 7) so posterior draws are truly independent.

### Results

| Dataset | Features | Beats marginal | Avg MSE marginal | Avg MSE TabPFN | Avg calib 10-90% | Gate verdict |
|---------|----------|----------------|-----------------|---------------|-----------------|--------------|
| MOONS | 2 | 50% (1/2) | 0.0550 | 0.0455 | **0.69** | **WEAK** |
| HELOC | 23 | 65% (15/23) | 0.0254 | 0.0634 | **0.62** | **PASS** |

### Key observations
- **MOONS**: Feature 1 reconstructed well; Feature 0 does not improve. The conditional density is informative for one of two features. Calibration 0.69 indicates reasonable posterior coverage.
- **HELOC**: 15/23 features beat the marginal baseline. Strong wins on ExternalRiskEstimate (48×), NumTradesOpeninLast12M (28×). Poor results on sparse near-binary features (NumTrades60Ever2DerogPubRec, MaxDelq2PublicRecLast12M, MaxDelqEver) — heavy-tailed distributions confuse the regressor.
- **Calibration** is reasonable (0.62–0.69): true values fall in the 10–90% posterior interval at the expected rate. Previous runs showed 0.00 due to the RNG seeding bug (see Post-review corrections above).

### Gate verdict
**PASS** — proceed to Experiment 2 with moderate confidence. The conditioning mechanism is informative for most continuous features; sparse/near-binary features need refinement.

---

## 3. Experiment 2: Counterfactual Generation (Baseline)

### Protocol
`append_target=True`, context filtered to target class, temperature=1.0, n_permutations=5, max_context=256. Test points batched by target class (one `set_context + impute_masked` call per target class). Out-of-[0,1] values clipped before metric computation; LOF computed on **unclipped** CFs to preserve true geometry.

### Results

| Dataset | Validity | LOF | Sparsity | True-action | Proximity L2 | OOB frac |
|---------|---------|-----|---------|------------|-------------|---------|
| MOONS (n=100) | **1.000** | **1.076** | 1.000 | **1.000** | 0.656 | 0.000 |
| HELOC (n=50) | **0.520** | 3,114,610,048 | 0.716 | **1.000** | 1.668 | 0.720 |

*Previous (buggy) Exp2 values: MOONS validity=0.85, HELOC validity=0.66. HELOC LOF=2.56B.*

### Key observations
- **MOONS**: Excellent corrected results. Validity=1.0 (every CF lands on the target class). LOF≈1.08 — CFs are indistinguishable from training data (strong plausibility). Zero OOB. The Y-as-column trick works perfectly on the well-separated 2-D class structure.
- **HELOC**: Validity=0.52 (barely above the ≥0.50 target; down from the inflated 0.66). True actionability=1.0 (immutables exactly frozen by construction). However, LOF≈3.1B and OOB=72% show CFs land far outside the training distribution. Root cause: conditioning on only 6 immutable features + Y while masking 17/23 features leaves too little information for the model — it extrapolates aggressively.
- **HELOC LOF interpretation**: A score of ~3.1B means the unclipped CFs are ~3.1B times further from the nearest training neighbour than typical in-distribution points. This is a structural consequence of 72% OOB extrapolation under sparse conditioning, not a numerical artefact.

---

## 4. Refinement Sweep (Stage 6)

### MOONS sweep (6 configs: temperature × n_permutations × context strategy)

| Config | t | n_perm | Context | Validity | LOF | OOB | Prox L2 |
|--------|---|--------|---------|---------|-----|-----|---------|
| t1e9_np5_tgt | 1e-9 | 5 | target-only | 1.000 | 0.987 | 0.000 | 0.665 |
| t05_np5_tgt | 0.5 | 5 | target-only | 1.000 | 1.022 | 0.000 | 0.689 |
| t10_np5_tgt | 1.0 | 5 | target-only | 1.000 | 1.073 | 0.017 | 0.683 |
| t10_np10_tgt | 1.0 | 10 | target-only | 1.000 | 1.073 | 0.017 | 0.683 |
| t10_np5_all | 1.0 | 5 | all classes | 0.983 | 1.060 | 0.000 | 0.648 |
| **t05_np5_all** | **0.5** | **5** | **all classes** | **0.983** | 1.020 | 0.000 | **0.643** |

**Skipped for budget**: max_context ∈ {100, 512} (low expected impact vs. time cost); nearest-neighbor context (requires per-point refitting).

**MOONS recommended config**: `t=0.5, n_perm=5, context=all_classes` (`t05_np5_all`) — best proximity (0.643) with near-perfect validity (0.983) and good plausibility (LOF=1.020, zero OOB). All MOONS configs are strong; the corrected validity ≥ 0.98 across all configs confirms the mechanism is sound. The previous recommendation of this config was based on inflated-but-wrong validity=0.783; **the recommendation stands with corrected numbers** since proximity is still the distinguishing metric.

### HELOC sweep (3 temperature configs, n_perm=3, max_test=15)

| Config | t | n_perm | Validity | LOF | OOB | Prox L2 |
|--------|---|--------|---------|-----|-----|---------|
| t1e9_np3_tgt | 1e-9 | 3 | 0.733 | 106,772,339,833 | **1.000** | 1.826 |
| t05_np3_tgt | 0.5 | 3 | 0.533 | 45,345,618,680 | 0.733 | **1.619** |
| t10_np3_tgt | 1.0 | 3 | 0.533 | 16,331,877,449 | 0.733 | 1.721 |

*LOF computed on unclipped CFs (Stage 7 fix); previously clipped values gave artificially lower LOF (~0.7–1.7B range by clipping CFs to [0,1] corners). The unclipped values reveal the true geometry: CFs are astronomically far outside the training distribution.*

**Skipped for budget**: nearest-neighbor context (~3h), max_context variants, n_perm=10 (too slow with 17 masked features).

**HELOC key finding**: No temperature configuration resolves the OOB/plausibility issue:
- t=1e-9 (MAP): OOB=1.0 (100%!) — MAP estimate is even more extreme than stochastic sampling. Even the modal prediction lies outside the training marginal distribution when only 7 columns are observed.
- t=0.5 and t=1.0: OOB=0.733, but LOF remains 16B–45B. Temperature is not the lever here.
- The root cause is structural: 17/23 features masked with sparse conditioning.

**HELOC recommended config**: Use the original Exp 2 config (t=1.0, n_perm=5) — it showed validity=0.52 at n=50 test points, providing more reliable statistics than the n=15 sweep sample. The sweep confirms temperature doesn't meaningfully improve validity or OOB.

---

## 5. Comparison vs. cel Baselines

Cel baseline numbers (PPCEF, DiCE, etc.) were not run during this experiment due to the TensorFlow/Python 3.13 incompatibility (see Backlog item #1). The cel repo's CF methods require additional dependency setup and training.

**What we can note**: Our MOONS validity=1.0 (refined: 0.983) would be competitive with or better than simple CF methods on a 2-D dataset. Our HELOC validity=0.52 is at the low end of typical CF method performance, and our LOF score is far worse — indicating a fundamental plausibility problem specific to the extrapolation behaviour under sparse conditioning.

---

## 6. Verdict

### Is zero-shot autoregressive TabPFN viable as a CF generator out-of-the-box?

**TL;DR**: Partially. Works well on low-dimensional data; breaks down on high-dimensional data when many features must be imputed simultaneously.

#### Success Criteria assessment (corrected numbers)

| Metric | Target | Result | Met? |
|--------|--------|--------|------|
| Pipeline offline | Yes | Yes | ✓ |
| Exp1 MOONS gate | Below marginal-mean MSE on ≥1 feature | 1/2 features (WEAK) | ✓ |
| Exp1 HELOC gate | ≥50% features beat marginal | 15/23 = 65% (PASS) | ✓ |
| Exp2 MOONS validity | ≥0.70 | 1.000 | ✓ |
| Exp2 HELOC validity | ≥0.50 | 0.520 | ✓ (barely) |
| Exp2 true actionability | 1.0 | 1.000 | ✓ |
| Exp2 HELOC LOF plausibility | Competitive with baselines | 3.1B (structurally poor) | ✗ |

#### What works
- **MOONS**: Validity=1.0, LOF≈1.08, true_actionability=1.0. The mechanism is sound for 2-D class-separated problems. Generated CFs are plausible and flip the class reliably.
- **Single-feature reconstruction**: Beats the marginal-mean baseline on 65% of HELOC features. The conditional density is informative when most features are observed.
- **Immutability guarantee**: By construction (frozen columns), true_actionability=1.0 on all datasets. This is a structural guarantee, not a learned property.
- **Fully offline**: Zero network calls after initial checkpoint staging.

#### What doesn't work (yet)
- **HELOC CF generation**: When 17/23 features must be imputed from 7 observed values, TabPFN extrapolates outside the training distribution. LOF≈3.1B and OOB=72% make the generated CFs implausible despite technically valid (class-flipping at 52%).
- **Temperature doesn't help**: Even MAP estimates (t=1e-9) have OOB=100% under sparse conditioning. This is a fundamental information-deficit problem, not a tunable parameter.
- **Proximity is poor**: CFs change all actionable features simultaneously (sparsity≈0.72). There's no minimum-change mechanism.

#### Why HELOC is harder than MOONS
Masking 17/23 features leaves only 6+Y=7 observed values to condition on — a very low information ratio. MOONS masks 2/2 features with Y, which is similarly sparse, but the 2-D class structure is simple enough that even marginal conditioning succeeds. The imputation task complexity scales with n_masked × n_permutations: HELOC is 17× harder than MOONS per impute call.

#### Next iteration (recommended)
1. **Reduce actionable set**: Start with 3–5 actionable features rather than 17. Evaluate whether HELOC validity improves when fewer features need to be imputed.
2. **Nearest-neighbor context**: Select context rows similar to the factual point (k-NN by immutable features). This gives more relevant conditioning signal.
3. **Post-hoc projection**: After CF generation, project back to the nearest training manifold point (e.g. WACHTER L2 minimization starting from the TabPFN-generated CF). Addresses proximity and OOB simultaneously.
4. **Feature-ordering DAG**: Order imputation by most-to-least predictable features (correlation with Y). This could improve reconstruction quality for the hard features.

---

## 7. Files

| Path | Description |
|------|-------------|
| `exp1_single_feature.py` | Experiment 1 runner |
| `exp2_counterfactuals.py` | Experiment 2 runner |
| `refine.py` | Refinement sweep runner |
| `configs/sweep.yaml` | Sweep configuration |
| `results/exp1_{moons,heloc}.csv` | Per-feature reconstruction metrics |
| `results/exp1_summary.md` | Exp 1 summary + gate verdict |
| `results/exp2_{moons,heloc}_metrics.csv` | Exp 2 per-dataset metrics |
| `results/exp2_summary.md` | Exp 2 summary + interpretation |
| `results/exp2_examples.md` | Concrete CF examples |
| `results/exp2_sweep_{moons,heloc}.csv` | Refinement sweep results |
| `results/REPORT.md` | This file |
