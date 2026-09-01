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
