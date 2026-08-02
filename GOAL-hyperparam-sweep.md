# Goal — Exp 7: beam-search hyperparameter sweep on PLGrid

Written 2026-08-02. Branch `lukasz/zeroshot-cf-beam-search`.

## Objective

The exp4 2×2 grid was produced with a **single, never-varied hyperparameter
configuration** (the `BeamConfig` defaults). Run a structured sweep over the beam-search
hyperparameters on Helios GH200, score every run under both metric conventions, and
publish an HTML report in the style of `experiments/zeroshot_cf/results/exp4_report.html`
(→ `results/exp7_sweep_report.html`).

Two questions the sweep must answer:

1. **Is heloc/frozen validity = 0.376 a hyperparameter artifact?** Finding 2 in
   `PROJECT_STATE.md` proved every missed flip is reachable in closed form and that
   generated CFs travel a median 0.3% of the available margin. Wider beams, more/wider
   candidates, or a weaker proximity penalty may close the gap — or prove the failure is
   structural.
2. **How sensitive are the headline numbers to the defaults?** A paper table built on one
   untested configuration is fragile; we need the sensitivity ranking of the axes.

## Baseline (current defaults — checked exactly once)

| hyperparameter | default | where |
|---|---|---|
| `beam_width` | 8 | `BeamConfig`, CLI `--beam-width` |
| `n_candidates` | 6 | `BeamConfig`, CLI `--n-candidates` |
| `lambda_actionable` | 1.0 | `BeamConfig`, CLI `--lambda-actionable` |
| `lambda_immutable` | = `lambda_actionable` in from-scratch; unused in frozen | `exp4_beam_search.py:503` |
| `max_context` | 256 | `BeamConfig`, CLI `--max-context` |
| `candidate_probs` | None → interior grid of K−1 quantiles + mode, tails excluded | `BeamConfig.probs()`, **not on CLI** |
| `N_ESTIMATORS` | 4 | module constant `exp4_beam_search.py:47`, **not on CLI** |
| `chunk_size` | 4096 | pinned — NOT a sweep axis (results are chunk-dependent) |
| `random_state` | 42 + target_class | fixed |

## Sweep design

One-factor-at-a-time around the defaults, plus a small set of targeted combinations.
No full factorial — the axes below would be 4·4·5·4·2 ≈ 640 cells.

### Axes (OFAT, ~15 configs)

| axis | values (default bold) | rationale |
|---|---|---|
| `beam_width` | 4, **8**, 16, 32 | more parallel hypotheses per row |
| `n_candidates` | 4, **6**, 10, 16 | more branching per step |
| `lambda_actionable` | 0.0, 0.1, **1.0**, 3.0, 10.0 | λ=0 removes the proximity penalty entirely — the direct test of whether proximity is suppressing validity (the 496 backwards-moving rows point here) |
| `max_context` | 128, **256**, 512, 1024 | more conditioning evidence per step |
| `candidate_probs` | **interior grid**, tail-inclusive `{0.05, 0.25, 0.5, 0.75, 0.95}` + mode | default candidates hug the mode → tiny steps; median CF travels 0.3% of available margin. Tail quantiles allow larger moves |

### Targeted combos (add after OFAT results, ~3–5 configs)

Combine the best level of the 2–3 most sensitive axes (e.g. `beam_width=16 ×
lambda_actionable=0.1 × tail probs`). Chosen from the OFAT table, not pre-committed.

### Cells

| cell | configs | why |
|---|---|---|
| `heloc/frozen` | all | the failing cell; question 1 |
| `law/frozen` | all | cheap (444 pts); sensitivity on a dataset where validity is already 1.0 — watch LOF/proximity/ε-sparsity instead |
| `heloc/fromscratch` + `law/fromscratch` | defaults only (cluster baseline) | completes a same-backend 2×2 at defaults |

**The cluster-defaults baseline doubles as open-work item 2** (`PROJECT_STATE.md` "Next"
#2): rerunning `law/frozen` and `heloc/*` on the GH200 removes the MPS-vs-CUDA confound.
Every sweep number is compared against the **cluster** baseline, never against the local
MPS numbers.

### Held fixed across every run

`--max-test -1` (full split), `--chunk-size 4096`, `N_ESTIMATORS=4` (unless added as a
late axis), `random_state` scheme, the discriminator pickles in
`experiments/zeroshot_cf/models/` (they define `y_target` — do not retrain), the
generation ordering (immutables first, then |coef|-desc actionables).

## Engineering prerequisites (must land before submitting)

The current pipeline cannot host a sweep — every run overwrites the same files:

1. **Config-tagged outputs.** `exp4_beam_search.py` hardcodes
   `arrays/exp4_{dataset}_{tag}_cfs.npz` and rewrites `exp4_summary.md`. Add a
   `--run-id` (short config slug, e.g. `bw16`, `lam0`, `probs-tail`) that lands in the
   npz filename and metrics CSV, defaulting to the current names so existing behaviour
   is unchanged. Store the full config dict inside the npz.
2. **CLI exposure** for `candidate_probs` (e.g. `--candidate-probs 0.05,0.25,...` or a
   named preset) and, if swept later, `--n-estimators`.
3. **Sweep driver** `plgrid/30_beam_sweep.sbatch`: sequential over configs within one
   job (same rationale as `20_beam_run.sbatch` — shared checkout, no array races), resume
   guard keyed on the config-tagged npz, `mirror_results` after every config,
   `FORCE=1` override. One job per cell (heloc/frozen and law/frozen can be two jobs in
   parallel — different npz namespaces).
4. **Aggregation.** Extend `exp4_metrics_table.py` / `reference_metrics.py` (or a new
   `exp7_sweep_table.py`) to glob the config-tagged npz files and emit one long-format
   CSV: `dataset, set, run_id, <config columns>, <registry metrics>, <reference metrics>`.
   Scoring runs locally — no GPU needed.
5. **Report generator or hand-built HTML** `results/exp7_sweep_report.html` (spec below).

## Execution plan

1. Land prerequisites 1–2, run `test_beam_search.py` (9/9) plus a new test that the
   default `--run-id` reproduces the existing filenames.
2. **Local micro-smoke only** (`--max-test 8`) to validate plumbing — generation
   otherwise never runs locally (standing rule: generation on PLGrid, scoring local).
3. Push the branch (cluster code is rsynced from the working tree — push first so runs
   are attributable to a commit), `bash plgrid/sync-to-plgrid.sh`.
4. Submit the cluster-defaults baseline job (4 cells, `FORCE=1` — stale smoke arrays
   exist on the cluster).
5. Submit the OFAT sweep jobs. Invoke the **`plgrid-run` skill before any cluster work.**
6. Pull arrays back, score locally, pick targeted combos, submit those, rescore.
7. Build the report; update `PROJECT_STATE.md` findings.

### Budget estimate

`law/fromscratch` full split took 86 s on the GH200 at defaults. Scaling by rows
(2092/444) and generated features (17/8) puts default heloc/frozen at ~15 min; the
largest configs (`beam_width=32` or `n_candidates=16`) at ~4× that. ~15 OFAT configs ×
2 cells + baselines + combos ≈ **15–25 GPU-hours worst case** — noise against the ~9060 h
remaining on `plgcountercontex-gpu-gh200`. Storage: each npz is ~100 KB; negligible.

## Deliverable — `results/exp7_sweep_report.html`

Same standalone-HTML style and skeleton as `exp4_report.html`:

- **Headline** — the answer to question 1, stated as a finding, not a table dump.
- **The sweep at a glance** — axes, values, cells, what was held fixed, backend.
- **Results** — per-axis sensitivity tables for heloc/frozen (validity first) and
  law/frozen (LOF median-log, proximity L1, ε-sparsity); both metric conventions, with
  the reference (dicoflex, valid-only, median-log LOF) convention primary.
- **Findings** — sensitivity ranking; whether any config beats validity 0.376 and at what
  proximity/plausibility cost; the Pareto view (validity vs L1) if a trade-off appears.
- **Numbers that must not be reported** — carry forward the exp4 list (registry
  `validity` ≡ discriminator accuracy under relabelling; all-rows-mean LOF;
  `sparsity_categorical`; `pairwise_diversity_mixed`), plus anything new.
- **Provenance** — commit hash per run, job IDs, backend, the all-zeros-rows caveat
  (still undecided — open item 3; report HELOC metrics with the caveat attached, or
  with/without the 115 rows if it changes a conclusion).
- **Reproduce** — exact sbatch/CLI incantations.

## Success criteria

- Every run scored under both conventions from config-tagged npz files; the long-format
  CSV committed; arrays mirrored to durable storage and copied locally.
- A defensible sensitivity statement per axis (even "flat within noise" is an answer).
- Question 1 answered: either a config that materially lifts heloc/frozen validity, or
  evidence the 0.376 ceiling is insensitive to all five axes (→ points the finger at the
  selector/scoring logic, feeding open-work item 1).
- Report committed; `PROJECT_STATE.md` updated (grid table, findings, open issues).

## Constraints and gotchas (inherited — do not relearn these)

- Generation on PLGrid only; scoring local. (`run-generation-on-cluster` memory.)
- Validity is `y_cf_pred == y_target`. Never the `cel` registry's `!= y_test`.
- `--chunk-size` stays 4096 in every run of every table.
- `FORCE=1` to defeat the resume guard when regenerating; stale smoke arrays exist on
  the cluster.
- sbatch log filename is hardcoded to the pattern in `--output`, not `--job-name`.
- The 115 all-zeros HELOC rows corrupt all-rows LOF means; use valid-only median-log.
- `envs/beam` python symlink dangles ~late August (SCRATCH 30-day retention) — verify the
  env still runs before the first submit; rebuild via `00_setup_env.sbatch` if not.
- Skills: `plgrid-run` before cluster work, `ml-engineer` before touching metric code,
  `commit-logical` style for commits.
