# Stage 5: Annealed Langevin + mix ablation + REPORT

**Goal**: Turn the deterministic flow into an annealed Langevin sampler that draws a *distribution* of diverse counterfactuals per factual, ablate the generative:discriminative drift mix `α:β` (proving the two greedy selectors are one mechanism), and write the consolidated REPORT with the recommended production config.
**Dependencies**: Stage 3 (DONE). Independent of Stage 4 (but fold Stage-4's result into the REPORT if available).

---

## Background (read `resources/math.md` §3, §5)

`β→∞`, noise→0 recovers the near-MAP commit; finite β with a schedule `β_k↑` samples `π_β`, yielding diverse recourses (turning L0-minimizer non-uniqueness into a feature). The mix ablation tests index Decision #2: discriminative drift (`α=0,β>0`) ≈ `prob_ascent`; generative drift (`α>0,β=0`) ≈ `class_divergence`; the combination should dominate either alone.

---

## Steps

1. Add annealing + sampling to the flow.
   - File: `experiments/zeroshot_cf/flow.py`
   - Add params `beta_schedule: str = "const"` (`"const"` | `"linear"` | `"geometric"`), `beta_max`, `n_samples: int = 1`. With `n_samples>1` and `noise>0`, run the chain `n_samples` times (distinct seeds) to draw multiple CFs per factual; anneal β over `n_steps`.
   - Return `info["samples"]` = list of `(x_cf, changed, flipped)` when `n_samples>1`; the single best (lowest L0 among valid) remains `x_cf` for metric compatibility.
   - `beta_schedule="const", n_samples=1, noise=0` is byte-identical to Stage 2/3 (regression-safe default).

2. Mix ablation driver.
   - File: `experiments/zeroshot_cf/exp7_flow_cf.py`
   - Support a small grid over `(alpha, beta)` ∈ {(1,0) generative-only, (0,1) discriminative-only, (1,1) dual, (0.5,2) validity-weighted}. Run on MOONS + HELOC (bounded n).
   - Write `results/exp7_mix_ablation.csv`.

3. Recourse-distribution metrics.
   - In the runner: with `n_samples>1`, report **diversity** (mean pairwise L2 among valid CFs per factual), **coverage** (fraction of factuals with ≥1 valid CF), and joint-NLL of the sampled set. Write to `results/exp7_sampling.csv`.

4. Consolidated REPORT.
   - File: `experiments/zeroshot_cf/results/REPORT.md` (append a new §) + `results/exp7_summary.md` (finalize).
   - Cover: Stage-1 score accuracy; Stage-3 flow-vs-greedy headline (MOONS plateau verdict, HELOC hold); Stage-4 path-adaptive context verdict (if run); Stage-5 mix-ablation (unification claim) + sampling diversity. State the **recommended production config** per dataset and whether the unified flow supersedes greedy.

5. Persist findings to memory.
   - Update `memory/iterative-greedy-cf-results.md` or add `memory/manifold-flow-cf-results.md` with the headline (did the flow break the MOONS plateau? did it hold HELOC? best config) + a one-line pointer in `MEMORY.md`. Convert relative dates to absolute.

---

## Verification

- [ ] `pytest experiments/zeroshot_cf/tests/ -q` fully green; `beta_schedule="const",n_samples=1,noise=0` reproduces Stage-3 output (regression guard).
- [ ] `results/exp7_mix_ablation.csv` shows the four `(α,β)` cells on both datasets; dual drift compared against either-alone.
- [ ] `results/exp7_sampling.csv` reports diversity + coverage for `n_samples>1`.
- [ ] REPORT § written with the recommended config and an explicit verdict on whether the unified flow supersedes greedy (positive or negative).
- [ ] Memory updated + `MEMORY.md` pointer added.
- [ ] Offline guarantee holds.

---

## Commit

`feat(manifold-flow): annealed Langevin sampling + mix ablation + consolidated REPORT (Stage 5)`
