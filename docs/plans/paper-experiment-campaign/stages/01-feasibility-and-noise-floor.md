# Stage 1: Feasibility and noise floor

**Goal**: Establish that the two new datasets resolve end-to-end, measure the seed-to-seed
noise floor that every later numeric comparison needs, and find the root cause of the Lending
Club cost anomaly — before any code change or expensive run depends on assumptions about them.
**Dependencies**: None

This stage produces no production code. It produces measurements that later stages depend on,
and it is the only stage whose output is allowed to change the plan's own numbers.

---

## Steps

1. **Create and publish the campaign branch.** The branch named in the plan header does not
   exist yet, and the plan directory and `docs/papers/` are untracked on `main`. Create
   `paper-experiment-campaign` off `main` @ `e1195ce`, commit
   `docs/plans/paper-experiment-campaign/`, `docs/papers/`, and the `docs/plans/LESSONS.md`
   update onto it, and push it to `origin` — the DGX runbook provisions with
   `git clone --branch paper-experiment-campaign`, so the remote branch must exist before any
   DGX work. All later stage commits land on this branch.

2. **Verify Adult and German Credit resolve to a usable benchmark case.**
   - Where: `CelDatasetProvider.prepare_adapter()` in `experiments/zeroshot_cf/datasets/cel.py`
   - The provider reads `config/datasets/{name}.yaml` from the pinned CEL checkout and derives
     the action schema from `FileDataset.actionable_features`, which is populated only from
     per-feature `actionable:` flags in that YAML. A config without them yields an **empty**
     actionable set and generation cannot proceed.
   - Confirmed present during planning: `adult_census.yaml` (76 lines, per-feature `actionable:`
     flags, protected attributes marked `actionable: false`) and `german_credit.yaml` (86 lines).
     Confirm German Credit actually declares the flags — planning verified only that Adult does.
   - For each dataset build a benchmark case and record: row counts after split, feature count,
     one-hot group count, actionable unit count, immutable feature count, class balance, and
     the logistic-regression test accuracy.
   - If German Credit lacks `actionable:` flags, do **not** invent an action schema. Record the
     gap, open a backlog item, and continue with Adult alone; Stage 6 then sizes matrices for
     five datasets. Substituting a guessed schema would make its results incomparable to
     everything else.

3. **Measure per-factual cost for the two new datasets** at n=25, k=3, TabICL backend, the
   `full_reference.yaml` CounterContEx parameters. This replaces the estimates in
   [compute-budget.md](../resources/compute-budget.md), which are explicitly labelled as
   estimates.

4. **Profile the Lending Club anomaly.** 27.52 s/factual against HELOC's 2.61 s on fewer columns
   than Give Me Some Credit. Start from stored artifacts, not a new run: `manifest.json` in
   `results/local/full_reference/` records `attempt_steps` and `diverse_histories` per point.
   - Compare the distribution of `attempt_steps` and beam-level candidate counts across the four
     datasets. Report the one-hot group count and total legal categorical alternatives per
     dataset alongside them.
   - If the artifacts do not settle it, run n=25 on Lending Club and HELOC with per-phase timing
     and compare.
   - A fix is **not** required by this stage. Identifying the cause is. If a cheap fix appears,
     record it in `decisions.md` and open a backlog item; do not implement it here, because a
     search-behavior change is a scientific identity change and must not land unversioned.

5. **Measure the seed-to-seed noise floor.** Run CounterContEx on HELOC at n=100, k=1, the
   `full_reference.yaml` parameters, across all five campaign seeds `[17, 42, 101, 202, 303]`.
   Record the mean, standard deviation and full range of `proximity_grouped_gower`,
   `action_unit_sparsity_mean`, `validity_returned_class` and `coverage`.
   - This is what makes later tolerance bands legitimate. Without it, every numeric comparison
     in the campaign is a point comparison, which fails a true-null change roughly half the time.
   - Publish the spread in `journal.md` and write it into
     [compute-budget.md](../resources/compute-budget.md) so later stages read it from one place.

6. **Confirm run determinism.** Re-run one of the Stage 1 cells with identical scientific
   settings and a different output root. The `run_id` must be identical and every deterministic
   `summary.csv` column must match exactly.
   - This is the only structural guarantee that later "the numbers changed" observations mean
     something. If it fails, everything downstream is unreliable and the stage blocks.

7. **Record the frozen baseline pointer.** Do not copy the 24-cell numbers into this plan; they
   live in [`docs/papers/positioning-draft.md`](../../../papers/positioning-draft.md) §1.4.
   Record in `journal.md` only the artifact paths and the fact that
   `results/local/full_reference/` and `results/local/architecture_full_reference/` are
   read-only historical evidence for the duration of this plan.

---

## Verification

- [ ] GATE Adult resolves to a benchmark case with a non-empty actionable unit set and at least
      one immutable feature — read from the constructed `BenchmarkCase`, not from the YAML. A
      config whose `actionable:` flags fail to reach `FeatureSchema` turns it red.
- [ ] GATE Re-running one cell reproduces its `run_id` byte-for-byte and every deterministic
      `summary.csv` column exactly — read from the two published `manifest.json` files. A
      nondeterministic seed path or an unpinned identity input turns it red.
- [ ] REPORT Three measurements, all recorded in `journal.md` and written into
      `resources/compute-budget.md`: (a) seed-to-seed mean, std and range of
      `proximity_grouped_gower` over 5 seeds on HELOC, read from the five published
      `summary.csv` files; (b) the Lending Club root-cause comparison and its conclusion, with a
      backlog item if a fix is identified; (c) measured per-factual cost for Adult and German
      Credit, replacing the estimates.

---

## Commit

`docs(plan): record campaign feasibility, noise floor, and cost profile`
