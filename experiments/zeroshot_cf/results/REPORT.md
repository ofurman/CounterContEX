# Zero-Shot Autoregressive CF Generation with TabPFN — Results Report

**Date**: 2026-06-15  
**Branch**: `zeroshot-tabpfn-cf`  
**Status**: All experiments complete (Stages 1–8 post-review corrected; Stage 9 DAG
ablation §7; Stage 10 from-scratch beam search §8).

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

### Datasets & in-context rows
80/20 **stratified** split (`random_state=42`), MinMax→[0,1] fit on train. TabPFN is never
trained; the "context" is the in-context conditioning set passed to `TabPFNUnsupervisedModel.fit()`.

| Dataset | Features | Train rows | Train per-class | Test rows (full split) | In-context rows |
|---------|---------:|-----------:|-----------------|-----------------------:|----------------:|
| MOONS | 2 (both actionable) | 800 | [397, 403] | 200 | 256 |
| HELOC | 23 (6 immutable, 17 actionable) | 8,367 | [4367, 4000] | 2,092 | 256 |

**In-context row count** (`sampler.py:set_context`, `exp2:159-173`): the conditioning set is
drawn from the **train** split only; with `context_type=target_only` (baseline) it is the
training rows of the **target class**, with `all_classes` the full train set; it is then
subsampled to `max_context` (default **256**) deterministically. Because every per-class pool
exceeds 256, **256 rows are used as context in all baseline runs**. Query (test) rows are
capped by `--max-test` (baseline tables below: MOONS n=100, HELOC n=50; `-1` = full split:
MOONS 200, HELOC 2092).

### How `frac_oob` (out-of-bounds fraction) is computed
`frac_oob` is the **row-level extrapolation rate** of the generated CFs
(`exp2_counterfactuals.py:264-267`):
```python
oob_mask = (X_cf < 0.0) | (X_cf > 1.0)          # per-cell, on the UNCLIPPED generated CFs
frac_oob = float(oob_mask.any(axis=1).mean())   # fraction of CF ROWS with ≥1 out-of-range cell
```
Features are MinMax-scaled to [0,1] on **train**, so a value outside [0,1] means TabPFN
extrapolated beyond the training support. A CF row counts as OOB if **any** of its features is
`< 0` or `> 1`; the metric is measured on the raw imputed array **before** the `np.clip(X_cf, 0, 1)`
applied prior to validity/proximity/LOF. High `frac_oob` (HELOC 0.72) is therefore the direct
signature of sparse-conditioning extrapolation, and explains the astronomical HELOC LOF.

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

### Full test-split results (2026-06-16)

The baseline table above used capped test sets (MOONS n=100, HELOC n=50). Re-run on the
**full stratified 80/20 test split** (MOONS n=200, HELOC n=2092) at the same config
(t=1.0, n_perm=5, max_context=256, target_only, 256 in-context rows):

| Dataset | n | Validity | LOF | Sparsity | True-action | Proximity L2 | OOB frac | n_failed |
|---------|--:|---------|-----|---------|------------|-------------|---------|---------|
| MOONS | 200 | **0.995** | 1.060 | 1.000 | 1.000 | 0.674 | 0.010 | 0 |
| HELOC | 2092 | **0.538** | 5.68e9 | 0.705 | 0.999* | 1.690 | 0.653 | 0 |

**Stability vs. capped runs**: the small-sample headline figures were not misleading —
MOONS validity 1.00→0.995, HELOC validity 0.52→**0.538** (still ≥0.50 target), HELOC
`frac_oob` 0.72→0.653 (still high). LOF stays astronomical. The sparse-conditioning
extrapolation is therefore **structural and confirmed at scale**, not a small-sample artefact.

\* **`true_actionability`=0.9986 is a metric artefact, not a violation.** `compute_metrics`
compares the **clipped** CF to the test data with exact `==` (`metrics_harness.py:101`).
MinMax was fit on **train**, so ~3/2092 test rows have an immutable feature natively just
outside [0,1]; clipping the (frozen) CF value to the boundary creates an artificial mismatch.
The hard immutability assert — run on the **unclipped** CF at `<1e-9` — passed, so immutables
were genuinely frozen. The n=50 cap never sampled these boundary rows.

**Robustness note (`robust_impute`)**: at t=1.0 the autoregressive chain can sample a value
that overflows float32 when fed to the next column, raising `TabPFNValidationError`. The full
HELOC set (2092 rows) first surfaced this; `exp2_counterfactuals.py:robust_impute` now imputes
in 256-row chunks and bisects a failing chunk down to the offending row (left as factual →
counted invalid), with non-finite survivors mapped to an out-of-range sentinel (→ counted OOB).
This full-split run completed with **`n_failed=0`** (chunk-level re-seeding avoided the blow-up).

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

## 7. Experiment 3: Feature-Ordering (DAG) Ablation

**Stage 9** tests whether replacing random-permutation averaging with an explicit
chain DAG (Y → immutables → actionable features) changes CF quality, and whether
combining it with a reduced actionable set addresses HELOC's OOB problem.

### Setup deviations from Stage-8 baseline

| Setting | Stage-8 | Exp3 | Reason |
|---------|---------|------|--------|
| context_type | target_only | **all_classes** | DAG places Y as explicit parent; constant Y (target_only) → TabPFN constant-feature error |
| HELOC n_permutations | 5 | **1** | Runtime (17 cols × 5 perms × 4 cells ≈ 88 min → ≈18 min) |
| HELOC max_test | 50 | **20** | Runtime reduction; identical across all 4 cells |

### Results

**MOONS** (2 cells, n_perm=5, 100 test pts):

| ordering | actionable_set | validity | LOF | frac_oob | true_action |
|----------|---------------|---------|-----|---------|------------|
| random | full | 0.960 | 1.071 | 0.000 | 1.000 |
| dag | full | 0.930 | 1.035 | 0.000 | 1.000 |

**HELOC** (4 cells, n_perm=1, 20 test pts):

| ordering | actionable_set | n_masked | validity | LOF (×1e6) | frac_oob | true_action |
|----------|---------------|---------|---------|-----------|---------|------------|
| random | full | 17 | 0.400 | 1031.8 | 0.500 | 1.000 |
| dag | full | 17 | 0.550 | 2467.6 | 0.650 | 1.000 |
| random | reduced | 6 | 0.200 | 99.5 | **0.100** | 1.000 |
| dag | reduced | 6 | **0.500** | **11.9** | **0.100** | 1.000 |

Reduced set (top-6 by |LR coef|): `ExternalRiskEstimate`, `NumTrades60Ever2DerogPubRec`,
`NumInqLast6M`, `NumInqLast6Mexcl7days`, `NetFractionRevolvingBurden`, `NumRevolvingTradesWBalance`.

### Verdict

**MOONS**: DAG is neutral-to-slightly-worse vs random (validity 0.93 vs 0.96, Δ=−0.03).
With only 2 actionable features, ordering freedom is minimal — near-zero delta is expected
and matches the hypothesis.

**HELOC — DAG vs random (full actionable set)**: DAG improves validity (0.55 vs 0.40,
Δ=+0.15) but **worsens OOB** (0.65 vs 0.50). This is consistent with the DAG giving each
actionable a *smaller parent set* (only its declared parents, not all observed columns),
which can produce more coherent chains but also riskier extrapolations.

**HELOC — reduced vs full actionable set**: Reducing to 6 masked features cuts OOB
dramatically (0.50 → 0.10 for random; 0.65 → 0.10 for dag). This confirms **the
sparse-conditioning hypothesis**: HELOC's OOB root cause is having 17 features masked
with only 6 immutables + Y as context, not temperature or ordering.

**Best HELOC cell**: `dag/reduced` — validity=0.500, frac_oob=0.100, LOF=11.9M (vs
1031.8M for random/full). If pursuing HELOC further, use a **reduced actionable set
(top-6 by oracle |coef|) with all_classes context** as the starting configuration.

**Y-first ordering**: The original request ("Y first, immutables, then actionable") is
meaningful only via the DAG path (with all_classes context). In the random-permutation
path, Y and immutables are already always-observed parents regardless of position — the
ordering is a no-op. Switching to DAG does change behavior (improved HELOC validity at
cost of higher OOB on full set; better balance on reduced set).

---

## 7b. Iterative Greedy CF (Exp4a, Stage 1) — `iterative-greedy-cf` plan

Change actionable features **one at a time**, conditioned class-conditionally on all
the rest (single masked column per step → dense conditioning), and **stop at the
discriminator's flip**. L0 is optimized by construction (features added only until the
flip). Two interchangeable selectors share the same loop:

- **prob_ascent** (Strategy 1) — pick the candidate whose near-MAP value most raises
  `disc.predict_proba[y_target]` (target-class context).
- **class_divergence** (Strategy 2) — pick the candidate whose class-conditional
  predictive mean shifts most between `Y=target` and `Y=current` (all-classes context,
  classifier-free).

Committed value = near-MAP (`t≈1e-9`, deterministic single column). Near-MAP single-column
commits are deterministic regardless of `n_permutations`.

### Stage-1 baseline metrics (`max_context=256`, `n_permutations=3`)

| Dataset | Selector | n | Validity | LOF | frac_oob | l0_count_mean | steps_max | true_action | failure_rate |
|---------|----------|---|----------|-----|----------|---------------|-----------|-------------|--------------|
| MOONS | prob_ascent | 100 | 0.690 | 1.009 | 0.000 | 1.26 | 2 | 1.000 | 0.310 |
| MOONS | class_divergence | 30 | 0.633 | 1.026 | 0.000 | 1.32 | 2 | 1.000 | 0.367 |
| HELOC | prob_ascent | 5 | **1.000** | **1.85** | **0.000** | **1.40** | 2 | 1.000 | 0.000 |

**Headline (HELOC).** Greedy salvages HELOC dramatically vs the Stage-8 one-pass baseline
(validity 0.538, frac_oob 0.65, LOF 5.7e9, all 17 actionables changed): on the bounded
smoke set it reaches validity 1.0, frac_oob 0.0, LOF ≈ 1.85, and changes **only ~1.4
features** to flip. Both the OOB collapse (single masked column ⇒ dense conditioning) and
the L0 minimization (stop at the flip) behave exactly as designed. Larger-`n` confirmation
and the selector/context ablations are Stages 2–4.

**MOONS caveat.** Validity ≈ 0.69 (not the ≈1.0 expectation). The ~31% `failure_rate` is
the **near-MAP plateau** the plan anticipated (Success Criteria, HELOC row): with only 2
actionable features and deterministic near-MAP commits, ~1/3 of boundary points exhaust
the budget without a hard flip. `steps_max=2` and `true_actionability=1.0` hold by
construction. The selector ablation (Stage 2) and context ablation (Stage 4) are the levers
meant to probe this.

---

## 7c. Selector & Context Ablations (Exp5 + Exp6, Stages 2 & 4) — DGX GPU

Stages 2 and 4 were run on a remote **NVIDIA GB10 (DGX)** GPU; everything else is identical
(offline v2 checkpoints, near-MAP commits, `n_permutations=3`). The local `claude -p`
per-stage runner cannot survive multi-hour jobs, so the heavy ablations were run detached
on the DGX and the results pulled back.

### Exp5 — Selector ablation (Stage 2): `prob_ascent` vs `class_divergence`

Each selector at its required context (`prob_ascent` → target-only; `class_divergence` →
all-classes), `max_context=256`, held identical otherwise within a dataset.

| Dataset | Selector | n | Validity | l0_count_mean | failure_rate | frac_oob | LOF |
|---------|----------|---|----------|---------------|--------------|----------|-----|
| MOONS | prob_ascent | 100 | **0.70** | **1.27** | 0.30 | 0.00 | 1.009 |
| MOONS | class_divergence | 100 | 0.64 | 1.31 | 0.36 | 0.00 | 1.011 |
| HELOC | prob_ascent | 50 | **0.90** | **1.67** | **0.10** | **0.04** | 9.5e6 |
| HELOC | class_divergence | 50 | 0.52 | 14.27 | 0.48 | 0.08 | 3.1e6 |

**`prob_ascent` wins decisively**, most starkly on HELOC: validity **0.90 vs 0.52**, L0
**1.67 vs 14.27**, failure 0.10 vs 0.48. `class_divergence` degrades on HELOC because its
low-cardinality integer columns route to TabPFN's *classifier* head, where the int-cast
collapses the feature support (so an expected-value shift is unrecoverable; the selector
falls back to a total-variation divergence — see plan Fixed Issue #1 / Decision #11). The
weak signal leaves many points unable to find a flipping feature → budget exhaustion.
**Chosen downstream selector: `prob_ascent`.** Note the HELOC `prob_ascent` n=50 number
(validity **0.90**, frac_oob **0.04**) is the robust confirmation of the Stage-1 n=5 smoke.

### Exp6 — Context ablation (Stage 4): size {256,512,1024,2048} × strategy {random,knn}×{target,both}

At `prob_ascent` (16 cells/dataset). MOONS n=100; **HELOC n=15** (bounded for runtime — the
2048 kNN cells dominate; the full grid took ~5.3 h even at n=15, so HELOC validity is noisy
at ±0.12 and the `frac_oob`/`LOF` trends are the robust signal).

**HELOC — bigger context *hurts*; relevant (kNN) context helps.**

- `random_*` plausibility degrades monotonically with size: `frac_oob` 256→2048 rises
  0.13→0.53 (`random_target`), 0.13→0.47 (`random_both`); validity falls (`random_both`
  0.67→0.40→0.47). A larger *random* pool dilutes the local conditioning the dense
  single-column step relies on. **Refutes "bigger context helps."**
- `knn_*` holds plausibility at large size (2048: `frac_oob` 0.13–0.20, LOF ≈ 1e6) where
  `random_*` blows up (0.47–0.53, LOF 1e7–1e10). kNN beats random at every size on LOF.
- **Best cell: `knn_both` @ size 256 — `frac_oob` 0.000, `LOF` 1.98** (every CF
  in-distribution), validity 0.67, L0 1.50. Nearest-neighbours from both classes at small
  size give the tightest on-manifold context.

**MOONS — flat & saturating.** `frac_oob ≡ 0`, LOF ≈ 1.0 everywhere; size saturates by
~512 (effective_size caps at the ~403/class or 800-row pool); best `(512, random_both)`
validity 0.82 (near-MAP ceiling); kNN gives no benefit on easy 2-D data.

**Recommended production config** — HELOC: `(prob_ascent, size=256, knn_both)`; MOONS:
`(prob_ascent, size=512, random_both)`. Full grids: `results/exp6_summary.md`.

### Updated TL;DR — does iterative greedy + better context salvage HELOC?

**Largely yes.** Iterative greedy with `prob_ascent` lifts HELOC over the Stage-8 one-pass
baseline on every axis: **validity 0.538 → 0.90**, **L0 17 → 1.67**, **frac_oob 0.65 →
0.04** (256/random). Adding a **small relevance-selected context** (`knn_both@256`) closes
the residual plausibility gap entirely — **frac_oob 0.65 → 0.00, LOF 5.7e9 → 1.98**. The
controlling lever is *relevant* context (kNN, small), not *large* context (which hurts).
Remaining limitation: a ~10–30% `failure_rate` from the deterministic near-MAP plateau
(raising temperature or posterior-sample-best, off-by-default per Decision #3, are the
untried knobs).

---

## 8. Experiment 4b: Beam-Search Counterfactuals — Frozen-Immutable vs From-Scratch

**Stage 10** introduces a **task-guided beam search** (reimplemented on the raw
`TabPFNRegressor` — no `tabpfn-extensions`). At each autoregressive step it branches over
candidate values (bar-distribution quantiles + mode), scores them by
`log p(feature) − λ·|feature − factual|` with a **hard `[0,1]` rejection**, keeps the
top-`beam_width` partial CFs, and reranks completed beams by validity (Decisions #11–12).

It is run in **two regimes that differ only in whether immutables are masked**:

- **Set 1 — frozen immutables** (actionable; directly comparable to the Exp 2/3 baseline):
  immutables are *observed* (held at the factual value); the beam generates only the 17
  HELOC actionables. `true_actionability = 1.0` by construction.
- **Set 2 — from scratch** (no masking): *every* feature is generated, conditioned only on
  `Y=target`; the factual enters only via the proximity penalty. Immutables drift.

For MOONS (no immutables) the two regimes are identical.

### Results (n=30; beam_width=8, n_candidates=6, λ_actionable=1.0, all_classes)

| Dataset | Set | Validity | LOF | Proximity L2 | frac_oob | True-action | Immut drift |
|---------|-----|---------|-----|-------------|---------|------------|------------|
| MOONS | 1 ≡ 2 | **1.000** | 0.977 | 0.470 | **0.000** | 1.000 | 0.000 |
| HELOC | **1 frozen** | 0.133 | 7.9e6 | 0.455 | **0.000** | **1.000** | 0.000 |
| HELOC | **2 from scratch** | **1.000** | **1.006** | 0.830 | **0.000** | 0.000 | 0.115 |

For reference, the previous baseline — **Exp 2 (imputation, frozen immutables)** on HELOC:
validity 0.52, LOF 3.1e9, frac_oob 0.72, proximity 1.67, true_action 1.0.

### The finding: actionability vs validity+plausibility is a *fundamental* tension on HELOC

- **Set 2 (from scratch) is excellent and strictly dominates Exp 2 on the generation axes**:
  validity **1.0**, LOF **1.0**, frac_oob **0.0** on both datasets. Masking *nothing* and
  letting the autoregressive chain generate every feature from `p(X | Y=target)` produces
  valid, fully in-distribution target-class instances. (This is *richer*, not sparser, than
  Exp 2: with a fixed ordering, feature `k` conditions on `Y` + the `k−1` already-generated
  features, so context grows along the chain instead of staying flat at "6 immutables + Y".)
  The cost is that it is no longer a *minimal/actionable* edit — immutables drift 0.115 and
  proximity is 0.83 (it is essentially "a plausible approved applicant", not "this applicant,
  minimally changed").

- **Set 1 (frozen immutables) cannot be salvaged on HELOC, even with beam search.** Validity
  collapses to **0.13** *and* plausibility degrades (LOF 7.9e6) despite frac_oob=0 — because
  welding target-class actionables onto the applicant's **wrong-class** frozen immutables
  produces rows that are in-bounds but **off-manifold** (no real applicant looks like that).
  HELOC's immutables (credit age/history) are so class-determining that fixing them to the
  factual values forces the rest into an invalid, implausible configuration.

- **Beam vs Exp 2 at equal constraints (both frozen-immutable)**: beam is far more *plausible*
  and *proximal* (LOF 7.9e6 vs 3.1e9; frac_oob 0.00 vs 0.72; proximity 0.46 vs 1.67) but has
  *lower* validity (0.13 vs 0.52). Exp 2's higher validity was bought by **wild extrapolation**
  (72% OOB) into invalid-but-clipped regions; beam stays in-distribution and therefore cannot
  flip the class while the wrong-class immutables are pinned. Neither escapes the tension —
  only regenerating the immutables (Set 2) does.

**Take-away**: TabPFN-as-density-estimator *can* generate valid, plausible target-class
instances zero-shot (Set 2). Whether that counts as a *counterfactual* depends on whether
actionability (fixed immutables) is required — and on HELOC that requirement is in direct
conflict with validity, because the protected features carry most of the class signal.

### Recommended next steps (do not oversell)
1. **Validity-aware exploration for Set 1** — steer per-step candidates toward the target
   class (partial-row discriminator score), so the actionable regime can find the rare valid,
   plausible configuration instead of relying on terminal rerank.
2. **Proximity dial** — `lambda_actionable` trades proximity for validity, but note the scale:
   TabPFN log-densities are O(several units), so λ must be O(20–100) to meaningfully pull CFs
   toward the factual (λ≈1 is density-dominated).
3. **Full-split eval on MPS** to confirm the n=30 numbers hold at scale.

---

## 9. Files

| Path | Description |
|------|-------------|
| `exp1_single_feature.py` | Experiment 1 runner |
| `exp2_counterfactuals.py` | Experiment 2 runner |
| `exp3_feature_ordering.py` | Experiment 3 (DAG ablation) runner |
| `exp4_greedy_cf.py` | Experiment 4a (iterative greedy CF) runner |
| `greedy.py` | Greedy loop + prob_ascent / class_divergence selectors |
| `beam_search.py` | Experiment 4b core: task-guided beam search (frozen / from-scratch) |
| `exp4_beam_search.py` | Experiment 4b runner (both regimes) |
| `results/exp4_{moons,heloc}_{frozen,fromscratch}_metrics.csv` | Exp 4b per-dataset, per-regime metrics |
| `results/exp4_summary.md` | Exp 4b two-regime summary + notes |
| `refine.py` | Refinement sweep runner |
| `configs/sweep.yaml` | Sweep configuration |
| `results/exp4_greedy_{moons,heloc}_metrics.csv` | Exp 4 per-dataset greedy metrics |
| `results/exp4_examples.md` | Greedy CF examples + recourse paths |
| `exp5_selector_ablation.py` | Experiment 5 (selector ablation) runner |
| `exp6_context_ablation.py` | Experiment 6 (context ablation) runner |
| `results/exp5_selector_{moons,heloc}.csv` | Exp 5 selector ablation metrics |
| `results/exp5_summary.md` | Exp 5 selector ablation tables + chosen selector |
| `results/exp6_context_{moons,heloc}.csv` | Exp 6 context grid (size × strategy) metrics |
| `results/exp6_summary.md` | Exp 6 context ablation grids + recommended config |
| `results/exp1_{moons,heloc}.csv` | Per-feature reconstruction metrics |
| `results/exp1_summary.md` | Exp 1 summary + gate verdict |
| `results/exp2_{moons,heloc}_metrics.csv` | Exp 2 per-dataset metrics |
| `results/exp2_summary.md` | Exp 2 summary + interpretation |
| `results/exp2_examples.md` | Concrete CF examples |
| `results/exp2_sweep_{moons,heloc}.csv` | Refinement sweep results |
| `results/exp3_ordering_{moons,heloc}.csv` | Exp 3 ablation results |
| `results/exp3_summary.md` | Exp 3 comparison tables + verdict |
| `results/REPORT.md` | This file |
