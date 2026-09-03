# Decisions

Append-only. **<=15 lines per entry** -- detail goes in `resources/`.

```
### D-N: [Title]
**Date**: YYYY-MM-DD - **Stage**: N (or "planning")
**Options**: A) [...] B) [...]
**Chosen**: [Option]
**Rationale**: [Why -- 1-3 lines]
```

An amendment to a success criterion uses this shape instead:

```
### D-N: Amend [criterion name]
**Date**: YYYY-MM-DD - **Stage**: N - **Type**: AMENDMENT
**Original**: "[quote verbatim]"
**Replacement**: "[new criterion]"
**Evidence**: [what proved the criterion was the defect -- link the journal entry]
**Class**: measurement (amendable). [Why this is not a correctness criterion.]
```

---

### D-1: Pre-execution review revisions
**Date**: 2026-09-01 - **Stage**: planning
**Options**: A) execute the plan as written B) revise per the pre-execution review's confirmed findings
**Chosen**: B
**Rationale**: The review confirmed three defects. (1) No stage owned the matrix target-model
axis — `orchestration/matrix.py` expands only datasets × methods × seeds, so E1 was
inexpressible; Stage 2 now owns the axis, with a GATE and an index entry. (2) Wachter in this
repository is black-box (`predict_proba` only), so the "expected clean failure on XGBoost"
premise was false; Stage 2 step 4 and Stage 7 step 4 now measure compatibility instead of
presuming failure. (3) The campaign branch did not exist and no step created it; Stage 1 now
starts by creating, committing, and pushing it.

### D-2: Defer Lending Club pool-fill early stopping
**Date**: 2026-09-01 - **Stage**: 1
**Options**: A) stop diverse search once k valid candidates exist B) keep the frozen search and defer a controlled stopping-policy comparison
**Chosen**: B
**Rationale**: The GB10 profile identifies the 16-candidate pool-fill rule as the long-tail
cause, but stopping at k can change DPP set quality and is therefore a scientific behavior and
identity change. Stage 1 records B-1 instead of modifying an unversioned search policy.

### D-3: Extend matrix v1 compatibly for target-model axes
**Date**: 2026-09-01 - **Stage**: 2
**Options**: A) bump the matrix schema B) add mutually exclusive `target_models` while retaining singular `target_model`
**Chosen**: B
**Rationale**: The plural field only adds Cartesian shorthand; each resolved `RunSpec` and its
scientific identity are unchanged. Existing v1 YAML/TOML matrices still resolve byte-for-byte,
while specifying both forms is rejected as ambiguous.

### D-4: All retained baselines support all target families
**Date**: 2026-09-01 - **Stage**: 2
**Options**: A) record expected clean failures B) retain all 15 method-family combinations
**Chosen**: B
**Rationale**: The measured HELOC n=1 matrix produced one available target-class-valid
candidate for NICE, Wachter, Growing Spheres, DiCE, and FACE against each of LR, MLP, and
XGBoost. No combination may be excluded from later completeness gates as an expected failure.

### D-5: Keep TabICL joint density method-internal
**Date**: 2026-09-01 - **Stage**: 3
**Options**: A) promote TabICL joint log-density to common evaluation B) retain it as method diagnostics
**Chosen**: B
**Rationale**: A common score from CounterContEx's own proposal model would structurally favour
that method and is unavailable for the empirical backend. Evaluation v2 instead adds neutral
detectability and grouped-Gower neighbour support; joint density may accompany method-specific
ablations but cannot become a cross-method headline metric.

### D-6: Expose native sets only for DiCE among retained baselines
**Date**: 2026-09-01 - **Stage**: 4
**Options**: A) synthesize sets for every baseline B) expose only methods' native set semantics
**Chosen**: B
**Rationale**: DiCE natively requests multiple genetic candidates. NICE selects one nearest
unlike neighbour, Wachter one optimized solution, Growing Spheres one nearest enemy, and FACE
one graph-path endpoint. Sampling or perturbing those single outputs would fabricate diversity,
so their single-counterfactual guards remain intact.

### D-7: Analyze historical metric schemas without rewriting them
**Date**: 2026-09-01 - **Stage**: 5
**Options**: A) rerun Stage 1 under evaluation v2 B) add a validated read-only v1 analysis path
**Chosen**: B
**Rationale**: The noise-floor evidence is a published historical v1 result and rerunning would
create different scientific identities rather than analyze that evidence. The compatibility
reader validates COMPLETE payloads, typed tables, content identities, and matrix membership;
it never upgrades, rewrites, or mixes v1 and v2 values in one seed group.

### D-8: Amend campaign matrix count
**Date**: 2026-09-02 - **Stage**: 6 - **Type**: AMENDMENT
**Original**: "Write the ten campaign matrices named in experiment-catalog.md."
**Replacement**: Write the nine executable matrices named there; E8 remains matrix-free.
**Evidence**: The catalog has YAML names for E1-E7, E9, and E10, while E8 explicitly says
"none — scoring pass over E1 arrays"; Stage 11 likewise forbids an E8 generation run.
**Class**: measurement (amendable). The declared count was arithmetically inconsistent with the
catalog; preserving nine avoids fabricating a generation protocol for a read-only evaluation.

### D-9: Declare categorical vocabularies from the action schema in DiCE
**Date**: 2026-09-02 - **Stage**: 7
**Options**: A) let DiCE infer categories from the selected reference rows B) declare every
schema category without adding synthetic reference observations
**Chosen**: B
**Rationale**: E1 exposed legal factual categories absent from a finite reference slice. The
adapter now supplies the schema vocabulary as permitted ranges, pandas categorical metadata,
and label-encoder classes, keeping query and KD-tree dimensions aligned without fabricating
training rows or leaking factuals. This changes existing behavior, so the implementation is
`dice-v5`; v2--v4 artifacts remain historical evidence under separate roots.

### D-10: Disable unsupported proposal features in the backend ablation
**Date**: 2026-09-02 - **Stage**: 8
**Options**: A) compare each backend's maximum feature set B) hold search fixed at the shared capability set
**Chosen**: B
**Rationale**: The empirical backend does not implement confidence conditioning or joint
scoring. E3 disables both in the TabICL arm too, so the comparison changes the proposal backend
rather than simultaneously changing search policy.

### D-11: Amend E3 resolved-identity difference gate
**Date**: 2026-09-02 - **Stage**: 8 - **Type**: AMENDMENT
**Original**: "The TabICL and empirical arms of E3 differ in exactly one resolved scientific field, `backend_implementation`."
**Replacement**: Paired arms differ only in the backend identity bundle: declared backend,
resolved backend implementation, and backend-owned checkpoint content IDs.
**Evidence**: A leaf diff of completed matched manifests found exactly those four paths; requiring
only the resolved implementation would contradict the scientific spec and content-addressed identity.
**Class**: measurement (amendable). The original field count omitted required backend provenance;
the replacement preserves the intended one-axis ablation and makes all backend content auditable.

### D-12: Fix the set-metric orientation in the analysis layer, not the evaluator
**Date**: 2026-09-03 - **Stage**: 8 (erratum)
**Options**: A) rename or re-sign the set metrics so the name carries its direction
B) leave every stored metric untouched, freeze its orientation in a test, document it, and add
the proximity normalization as an analysis-only derived value
**Chosen**: B
**Rationale**: Renaming or re-signing `set_action_jaccard_*` would break the published E2
artifacts and require an evaluation-version bump, and Stage 3 permits exactly one, already spent.
The defect was in the reading, not the measurement.

Orientation of record for every set metric, as implemented and now documented in
`experiments/zeroshot_cf/README.md`:

- `set_coverage_at_k`: fraction of factuals whose **every** requested slot returned a candidate
  (`returned_count >= k`), availability only. Higher is better.
- `set_returned_count_mean`: mean returned candidates per factual. Higher is more available.
- `set_action_jaccard_mean` / `set_action_jaccard_min`: mean and minimum pairwise Jaccard
  **distance** between changed-feature sets, one-hot groups counting as one unit. 0 means every
  candidate changes the same features, 1 means disjoint sets. Higher is more action-diverse.
- `set_pairwise_gower_mean` / `set_pairwise_gower_min`: mean and minimum grouped-Gower distance
  between pairs of returned candidates. Higher is more value-diverse.
- Denominator for every pairwise set metric: pairs of available candidates, so only factuals with
  at least two returned candidates contribute. Set metrics do not condition on class or threshold
  validity.

`set_pairwise_gower_ratio = set_pairwise_gower_mean / proximity_grouped_gower` is derived by
`analysis.core.derive_cell_metrics` from summary fields that already exist, is null when
proximity is missing, non-finite, or non-positive, and is never written into `summary.csv` or any
manifest. It is not part of any scientific identity, so no evaluation version changed and every
historical artifact remains readable and byte-identical. It exists because raw pairwise Gower is
confounded with proximity: a candidate set farther from its factual spreads more for free.

### D-13: Omit the 1.5 arm from E2b and reuse the E2 cells for that point
**Date**: 2026-09-03 - **Stage**: 8 (erratum)
**Options**: A) sweep `diversity.max_gower_ratio` over {1.5, 2.5, 4.0} inside E2b
B) sweep {2.5, 4.0} and treat the E2 CounterContEx cells as the 1.5 point
**Chosen**: B
**Rationale**: E2b holds every scientific field byte-identical to the E2 CounterContEx arm and
varies only `diversity.max_gower_ratio`. A 1.5 arm at seeds 17, 42, and 101 would therefore
resolve to exactly the same `cell_id` as the corresponding E2 cells, colliding in the
cross-matrix uniqueness contract and re-running work whose artifacts already exist under
`campaign/e2_diverse`. Reusing the E2 cells keeps the sweep at one axis, keeps 36 new cells, and
keeps the 1.5 point on published evidence rather than a duplicate identity. The cost is that the
1.5 point carries five seeds while the new arms carry three, so budget comparisons must be read
on the shared seeds {17, 42, 101}.
