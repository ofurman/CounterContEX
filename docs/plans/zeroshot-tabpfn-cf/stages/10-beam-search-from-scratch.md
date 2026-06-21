# Stage 10: From-Scratch Counterfactuals via Task-Guided Beam Search

**Goal**: Test a fundamentally different generation regime from Exp 2/3. Instead of
*imputing* the masked actionable features (freezing immutables as observed context),
**generate every feature of the counterfactual from scratch** — the only observed
signal is `Y=target`. The factual instance is never observed; it enters the search
only through a per-feature **proximity penalty**. Generation is driven by a
**task-guided beam search** that, at each autoregressive step, branches over several
candidate values, scores them by `log p(feature) − λ·|feature − factual|` (with a
hard `[0,1]` rejection), keeps the top-`beam_width` partial CFs, and finally reranks
the completed beams by validity.

**Dependencies**: Stages 1–8 DONE. Reuses the cel dataset loader, the LR discriminator,
and `metrics_harness.compute_metrics`. **Does not** use `TabPFNUnsupervisedModel` —
the autoregressive chain is reimplemented directly on the local `TabPFNRegressor`
conditional-density API so beam search controls the branch point.

> **Why this stage exists.** Exp 2/3 established that HELOC's failure is *sparse
> conditioning*: imputing 17/23 features from only 6 immutables + Y forces TabPFN to
> extrapolate (frac_oob 65–72%, LOF ≈ 3.1 **billion**). Two ideas motivated Exp 4:
> 1. **Beam search** — generate multiple candidate explanations in parallel and keep
>    the best-scoring partial CFs, rather than greedily sampling one value per cell.
> 2. **From scratch** — mask *everything* and generate autoregressively. This looks
>    sparser, but it is actually *richer*: with a fixed ordering, feature `k`
>    conditions on `Y + the k−1 already-generated features`, so the average
>    conditioning context grows along the chain instead of staying flat at "6 + Y".
>    It is the regime `generate_synthetic_data` is built for.

**Guardrail reminder**: do NOT modify `src/tabpfn/**`. Runs must work offline
(checkpoints staged once). Report the validity↔plausibility tradeoff honestly — if
from-scratch generation buys plausibility at the cost of validity, say so plainly.

---

## Mechanism

For a fixed generation ordering `f_1 … f_D` over **all** features (Y appended as the
last context column at index `y_idx = d`):

```
beams ← [ one partial CF per query, all features NaN, Y = target ]
for k in 1..D:
    observed = [y_idx] + ordering[:k-1]              # SAME for every beam & query
    fit reg once:  context[:, observed]  →  context[:, f_k]
    out = reg.predict(beams[:, observed], output_type="full")   # criterion, logits
    candidates ← { icdf(logits, q) : q in probs } ∪ { mode(logits) }   # K spread values
    reject candidates ∉ [0,1]                         # hard OOB control (clip-fallback if all OOB)
    step_score(c) = -criterion.forward(logits, c)  −  λ_k · |c − factual_{f_k}|
    expand each beam by its K candidates; keep top-`beam_width` per query by cumulative score
rerank completed beams per query: prefer disc.predict(cf)==target, tie-break by cumulative score
```

- **λ is per-feature**: `lambda_immutable` (large, soft-freeze) on immutable columns,
  `lambda_actionable` on the rest. Immutables are *still generated*, just strongly
  pulled to the factual value.
- **Efficiency**: with a fixed ordering, at step `k` every beam (and every query) shares
  the same observed-column set, so the regressor is **fit once per step** and all
  `(query × beam)` rows are predicted in one batch. Total cost ≈ `D` fits + `D` batched
  predicts (~13 s for HELOC's 23 features × 15 query pts on CPU).
- **Context must be `all_classes`** (mandatory): a constant Y in context trips TabPFN's
  constant-feature validator (same constraint Exp 3 hit, Decision #10).

---

## Steps

1. **Core module** `experiments/zeroshot_cf/beam_search.py`:
   - `build_generation_ordering(n_features, immutable_idx, actionable_order=None)` —
     full-feature order, immutables first (so their near-factual values condition the rest).
   - `BeamConfig` dataclass (`beam_width`, `n_candidates`, `lambda_actionable`,
     `lambda_immutable`, `max_context`, `candidate_probs`, `random_state`).
   - `_candidates_and_logpdf(criterion, logits, probs)` — `icdf` quantiles + `mode`,
     scored by `-forward` (log-density). Deterministic (no RNG) ⇒ reproducible beams.
   - `generate_cf_beam(reg, X_context, y_context, X_factual, target_class, ordering,
     immutable_idx, config, disc_model)` → `(X_cf, aux)` with per-row diagnostics
     (`chosen_valid`, `immutable_drift`, `n_oob_fallback`, cumulative score/log-density).

2. **Runner** `experiments/zeroshot_cf/exp4_beam_search.py`:
   - `generate_counterfactuals_beam(...)` — load data, train/load LR oracle, order
     actionables by `|LR coef|` desc, batch by target class, call `generate_cf_beam`.
   - `evaluate_and_report_beam(...)` — **does not** assert immutables unchanged (they
     drift by design); reports `immutable_drift_{mean,max}` and `true_actionability`
     as informational. Computes frac_oob (pre-clip) + the standard metric suite.
   - CLI: `--dataset --beam-width --n-candidates --lambda-actionable --lambda-immutable
     --max-context --max-test`. Writes `results/exp4_{moons,heloc}_metrics.csv` +
     `results/exp4_summary.md`.

3. **Tests** `experiments/zeroshot_cf/tests/test_beam_search.py` (8 tests, **offline** —
   a real `FullSupportBarDistribution` wired to a fake regressor; no checkpoint/network):
   ordering correctness; candidate shape/finiteness; end-to-end shape/no-NaN/in-bounds;
   validity rerank; immutable soft-freeze monotonicity (higher λ ⇒ less drift); OOB
   clip-fallback. Prior tests unaffected (they use the unsupervised model, which Exp 4
   does not).

4. **Run + sweep** (offline): canonical run at defaults for both datasets; a
   `lambda_actionable` frontier sweep on HELOC to map the validity↔proximity dial.

5. **Report & index**: add an "Experiment 4" section to `results/REPORT.md` (frontier
   table + honest verdict), a README pointer, and a Stage 10 row + Decisions here.

---

## Verification

- [x] `uv run pytest experiments/zeroshot_cf/tests/test_beam_search.py -q` → 8 passed (offline).
- [x] Live MOONS: validity=1.0, LOF≈0.98, frac_oob=0.0, proximity≈0.47.
- [x] Live HELOC (n=30, defaults): validity≈0.33, LOF≈1.03, **frac_oob=0.0**,
      proximity≈0.84, immutable_drift mean≈0.04.
- [x] `git diff --name-only main..HEAD -- src/tabpfn` is empty (core untouched).

---

## Outcomes (actuals)

- **Plausibility is solved unconditionally.** Across the entire `lambda_actionable`
  frontier, LOF ≈ 1.0 and **frac_oob = 0.00** — versus Exp 2's LOF ≈ 3.1e9 / OOB 72%.
  The `[0,1]` candidate rejection + growing autoregressive context eliminate the
  extrapolation that broke Exp 2.
- **`lambda_actionable` is a clean validity↔proximity dial**, but note the *scale*:
  TabPFN log-densities are O(several units), so λ≈1 is density-dominated (proximity has
  no teeth); λ must be O(20–100) to pull CFs toward the factual.

  | λ_actionable | validity | LOF | proximity L2 | frac_oob |
  |---|---|---|---|---|
  | 0.1 – 1.0 | 0.35 | 1.02 | 0.92 | 0.00 |
  | 20  | 0.05 | 1.03 | 0.20 | 0.00 |
  | 100 | 0.05 | 1.04 | 0.16 | 0.00 |

- **The tradeoff:** the whole frontier sits at **validity ≤ 0.35 < Exp 2's 0.52**.
  From-scratch generation trades validity for plausibility + proximity. The ~0.35
  ceiling is discriminator disagreement (TabPFN `p(X|Y=target)` produces plausible
  points the LR oracle labels "target" only ~35% of the time); the terminal rerank
  cannot pick a valid beam if none of the explored beams are valid.
- **Immutable soft-freeze** plateaus at the candidate-spacing floor (drift ≈ 0.04–0.05
  at λ≥100): the closest fixed quantile candidate to the factual is ~0.5/`n_candidates`
  away. Injecting the factual value itself as a guaranteed candidate for immutable
  columns would drive drift → 0 (documented as a next step, not implemented).

### Recommended next steps (do not oversell Exp 4)
1. **Validity-aware exploration**: add a per-step signal that steers candidates toward
   the target class (partial-row discriminator score, or a small validity bonus), so
   beams *explore* valid regions instead of relying on terminal rerank.
2. **Factual-as-candidate for immutables** → true_actionability ≈ 1.0 at zero extra cost.
3. **Full-split eval on MPS** to confirm the n=20–30 frontier holds at scale.

---

## Commit

`feat(zeroshot-cf): add from-scratch beam-search CF generation (Exp4)`
