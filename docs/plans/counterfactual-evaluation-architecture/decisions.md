# Decisions

Append-only. Detail lives in `resources/architecture.md`.

---

### D-1: Use a narrow contract architecture
**Date**: 2026-08-31 · **Stage**: planning
**Options**: A) common runner around current functions; B) method/result contracts plus reusable layers; C) dynamic plugin framework.
**Chosen**: B.
**Rationale**: A preserves unstable method signatures; C adds machinery without a current external-plugin need. A typed method/result seam removes the actual duplication.

### D-2: Keep explicit local registration
**Date**: 2026-08-31 · **Stage**: planning
**Options**: A) explicit factory map; B) Python entry-point discovery; C) dependency-injection framework.
**Chosen**: A, inside the existing `experiments.zeroshot_cf` package.
**Rationale**: It is deterministic, offline, easy to test, and fits the current `package = false` workspace. A future package move is independent.

### D-3: Separate availability from validity
**Date**: 2026-08-31 · **Stage**: planning
**Options**: A) preserve factual fallback as covered; B) make availability explicit and evaluate class/threshold validity separately.
**Chosen**: B, while retaining best-effort outputs only as namespaced method artifacts.
**Rationale**: Coverage, class validity, and threshold success answer different questions and must not be inferred from method-specific diagnostics.

### D-4: Use capability-specific foundation backends
**Date**: 2026-08-31 · **Stage**: planning
**Options**: A) universal tabular-foundation-model API; B) DiCoFlex proposal protocol with explicit optional capabilities.
**Chosen**: B.
**Rationale**: TabICL, TabPFN, and TabFM need not share conditional sampling or joint-density semantics. Unsupported search/backend combinations should fail at preparation.

### D-5: Preserve compatibility during cutover
**Date**: 2026-08-31 · **Stage**: planning
**Options**: A) delete numbered entry points immediately; B) retain thin shims and a v1 artifact exporter through the migration.
**Chosen**: B.
**Rationale**: Athena and local workflows already depend on those commands. Thin forwarding modules do not retain architectural coupling.
