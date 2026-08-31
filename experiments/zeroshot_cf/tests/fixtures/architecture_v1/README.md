# Architecture v1 compatibility fixtures

These fixtures freeze the public surface that the architecture migration must keep while
ownership moves into typed packages. `compatibility.json` records the six benchmark methods,
legacy CSV/NPZ names, shared columns, public commands, and the supported generator import.
`semantic_cases.json` records small reasoned examples for the new generation and evaluation
contracts. `boundary_edges.json` is an allowlist for the known forbidden targets in its monitored
legacy modules: an edge may be removed, but a new matching violation in that scope fails the
architecture test. Stages 2 and 3 expand the boundary check to the new package layers.

The legacy `coverage` and `validity` columns are compatibility-only. Existing runners commonly
return a factual or best-effort row on failure, so legacy coverage can be 1 even when no
counterfactual is available. Legacy validity is target-class validity and does not express the
search probability threshold.

The canonical evaluator intentionally reports truthful, separate names:

- `coverage`: factuals with at least one available candidate / factuals.
- `validity_returned_class`: target-class candidates / available candidates.
- `validity_returned_threshold`: target-class candidates meeting the threshold / available
  candidates.
- `valid_success_rate_*_per_requested_slot`: successes / requested slots.

JSON `null` represents the required NaN payload of an unavailable slot. Best-effort rows live in
namespaced artifacts and do not make a slot available. All examples are synthetic; they are not
copied from ignored local result directories.
