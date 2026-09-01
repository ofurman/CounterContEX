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

### D-1: Use a hard active cutover
**Date**: 2026-09-01 - **Stage**: planning
**Options**: A) hard cutover; B) canonical name plus deprecated aliases; C) display-only rename.
**Chosen**: A, with a private bridge only while staged commits migrate internal consumers.
**Rationale**: The user requested the proper name everywhere. Permanent aliases would leave the old name active and create ambiguous scientific identity.

### D-2: Separate current naming from historical evidence
**Date**: 2026-09-01 - **Stage**: planning
**Options**: A) rewrite every textual occurrence; B) rename active tracked surfaces and preserve immutable records.
**Chosen**: B.
**Rationale**: Completed plans are append-only and old manifests are content-addressed. Rewriting either would falsify history rather than rename the current method.

### D-3: Preserve v1 artifact bytes
**Date**: 2026-09-01 - **Stage**: planning
**Options**: A) rename all legacy artifacts; B) rekey current dispatch while preserving frozen file/field/value contracts.
**Chosen**: B.
**Rationale**: The frozen artifact surface contains TabICL-era names, not the old current brand. It can remain byte-compatible while the active dispatcher and CLI change.

### D-4: Distinguish project and method capitalization
**Date**: 2026-09-01 - **Stage**: planning
**Options**: A) copy project spelling to the method; B) use the user's exact method spelling.
**Chosen**: B: `CounterContEx` for the method, `countercontex` for machine identifiers.
**Rationale**: Existing project title `CounterContEX` and schema namespace are separate surfaces and remain unchanged.

### D-5: Move implementation before the atomic identity cutover
**Date**: 2026-09-01 - **Stage**: planning review
**Options**: A) temporary alias bridge from D-1; B) retain old identity in Stage 1 and cut all identity producers over in Stage 2; C) merge both stages.
**Chosen**: B; this supersedes only D-1's temporary-bridge mechanism, not its hard-cutover result.
**Rationale**: Registry lookup occurs after `MethodSpec` establishes identity, so an alias there cannot normalize cell IDs. Keeping the old key/version briefly preserves green commits without duplicate identities or forwarding packages.

### D-6: Define v1 compatibility semantically
**Date**: 2026-09-01 - **Stage**: planning review
**Options**: A) literal artifact bytes from D-3; B) exact schema/IDs and deterministic values, excluding variable timing equality.
**Chosen**: B; this clarifies D-3's compatibility intent.
**Rationale**: Runtime timing fields make raw CSV bytes nondeterministic. Complete deterministic CSV and NPZ comparisons detect rename regressions without an impossible byte-identity gate.
