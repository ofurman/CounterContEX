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
