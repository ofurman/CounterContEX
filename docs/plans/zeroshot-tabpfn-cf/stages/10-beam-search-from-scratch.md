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

- [x] `uv run pytest experiments/zeroshot_cf/tests/test_beam_search.py -q` → 9 passed (offline,
      incl. frozen-immutable mode test).
- [x] Live MOONS (Set 1 ≡ Set 2, no immutables): validity=1.0, LOF≈0.98, frac_oob=0.0, proximity≈0.47.
- [x] Live HELOC Set 2 (from scratch, n=30): validity=**1.0**, LOF≈1.01, frac_oob=0.0, drift≈0.12.
- [x] Live HELOC Set 1 (frozen, n=30): validity=0.13, LOF≈7.9e6, frac_oob=0.0, true_action=1.0.
- [x] `git diff --name-only main..HEAD -- src/tabpfn` is empty (core untouched).

---

## Outcomes (actuals) — two regimes

> **Correction.** An interim run kept immutables *soft-frozen* (large `lambda_immutable`) and
> reported "from-scratch caps at validity ≈0.35 / trades validity for plausibility". That was
> an **artifact of soft-freezing immutables to wrong-class values**, not a property of the
> method. Re-run with the two clean regimes below, the picture is different and stronger.

| Dataset | Set | validity | LOF | proximity | frac_oob | true_action | immut drift |
|---------|-----|---------|-----|-----------|---------|------------|------------|
| MOONS | 1 ≡ 2 | **1.00** | 0.98 | 0.47 | 0.00 | 1.00 | 0.00 |
| HELOC | 1 frozen | 0.13 | 7.9e6 | 0.46 | 0.00 | **1.00** | 0.00 |
| HELOC | 2 from scratch | **1.00** | **1.01** | 0.83 | 0.00 | 0.00 | 0.115 |

- **Set 2 (from scratch) strictly dominates Exp 2 on the generation axes**: validity **1.0**,
  LOF **1.0**, frac_oob **0** on both datasets. Masking nothing and generating every feature
  from `p(X|Y=target)` yields valid, in-distribution target-class instances — the conditioning
  is *richer*, not sparser (feature `k` sees `Y` + the `k−1` already-generated features). Cost:
  not a minimal/actionable edit (immutables drift 0.12, proximity 0.83).
- **Set 1 (frozen immutables) cannot be salvaged on HELOC, even with beam search**: validity
  collapses to **0.13** and plausibility degrades (LOF 7.9e6, off-manifold) despite frac_oob=0,
  because welding target-class actionables onto the wrong-class frozen immutables yields
  in-bounds-but-unreal rows. HELOC's immutables carry most of the class signal.
- **Beam-frozen vs Exp 2 (equal constraints)**: beam is far more plausible/proximal
  (LOF 7.9e6 vs 3.1e9, OOB 0 vs 0.72, prox 0.46 vs 1.67) but lower validity (0.13 vs 0.52) —
  Exp 2's validity was bought via 72% OOB extrapolation; beam stays in-distribution and can't
  flip the class while immutables are pinned.
- **The finding (Decision #13): actionability ⟂ validity+plausibility on HELOC.** You can have
  the protected immutables (Set 1, but invalid/implausible) or a valid, plausible target-class
  instance (Set 2, but immutables regenerated) — not both, because the protected features
  determine the class.

### Recommended next steps (do not oversell Exp 4)
1. **Validity-aware exploration for Set 1**: steer per-step candidates toward the target class
   (partial-row discriminator score) so the actionable regime can find rare valid+plausible
   configurations instead of relying on terminal rerank.
2. **Proximity dial**: `lambda_actionable` trades proximity for validity; it must be O(20–100)
   to overcome TabPFN's log-density scale (λ≈1 is density-dominated).
3. **Full-split eval on MPS** to confirm the n=30 numbers hold at scale.

---

## Commit

`feat(zeroshot-cf): add from-scratch beam-search CF generation (Exp4)` (initial), then
`feat(zeroshot-cf): add frozen-immutable regime + correct Exp4 two-regime analysis`.
