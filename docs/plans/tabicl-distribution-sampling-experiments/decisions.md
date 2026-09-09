# Decisions

Append-only. **<=15 lines per entry** -- detail goes in `resources/`.

### D-1: Define continuous greedy/top-k/top-p on a fixed quantile partition
**Date**: 2026-09-10 - **Stage**: planning
**Options**: A) apply token decoders directly to a continuous distribution B) freeze a finite quantile-space approximation C) use only categorical top-k/top-p
**Chosen**: B for numerical q; categorical policies use exact discrete masses.
**Rationale**: Continuous top-k/top-p is otherwise undefined. Split u into 256 equal-mass bins, rank each by `log_prob(Q(midpoint))`, retain five bins for top-k and the density-ordered `.9`-mass union for top-p. Greedy remains the existing `quantile_mode`, not the 256-bin approximation. The declared truncation includes both tails and is not an intrinsic TabICL property.

### D-2: Count raw proposals and oracle work separately
**Date**: 2026-09-10 - **Stage**: planning
**Options**: A) refill until every arm has the same number of unique rows B) cap raw draws and classifier rows, retaining natural no-ops/duplicates
**Chosen**: B.
**Rationale**: Refilling gives duplicate-prone arms hidden compute. Every arm records raw draws, projected/no-op/duplicate rows, unique classifier rows, model calls, and runtime; matched claims are made only within the same declared raw and oracle caps.

### D-3: Keep the main study at k=1
**Date**: 2026-09-10 - **Stage**: planning
**Options**: A) repeat E3 at k=3 B) establish the mechanism at k=1, then run a small predeclared k=3 replication
**Chosen**: B.
**Rationale**: Current k=3 search enumerates categories and adds beam/DPP confounds. The replication is limited to grid-9, IID-9, `beta=0`, and `beta=1` under an explicit shared categorical budget.

### D-4: Keep pushforward evidence outside common evaluation
**Date**: 2026-09-10 - **Stage**: planning
**Options**: A) add ragged proposal metrics to evaluation v3 B) publish a method-specific, hash-checked diagnostic
**Chosen**: B.
**Rationale**: E12 measures proposal mechanics, not method-blind final-candidate quality. Canonical final outputs still use the common evaluator; E12 uses `proposal-diagnostic-v1` and does not invalidate evaluation-v2 grouping.

### D-5: Implement pi-beta by common-pool SIR
**Date**: 2026-09-10 - **Stage**: planning
**Options**: A) multiply sampled rows by an estimated q density B) draw a common pool from q and weight only by classifier probability
**Chosen**: B, with `M=64`, `B=9`, `beta in {0,.5,1,2}` and standard weighted resampling with replacement.
**Rationale**: The empirical pool already represents q; multiplying by q again would approximate `q^2 p^beta`. With-replacement SIR targets the empirical pi-beta approximation; repeated/no-op draws consume budget, projection precedes scoring, and all beta arms share raw draws and resampling uniforms.

### D-6: Separate pilot and confirmation factuals by split
**Date**: 2026-09-10 - **Stage**: planning
**Options**: A) reuse the first test factuals for engineering pilots B) use validation factuals for pilots and test factuals only for confirmation
**Chosen**: B.
**Rationale**: A protocol-level factual-partition field preserves the existing test default and scientific identity while preventing pilot inspection from contaminating confirmation. Within each split, reuse the benchmark's deterministic target-stratified factual selection rather than source order. No pilot outcome may change the predeclared primary contrasts.

### D-7: Scope common pools to identical adaptive states
**Date**: 2026-09-10 - **Stage**: pre-review correction
**Options**: A) require common pool IDs across complete beta trajectories B) require them only for identical state/action keys and separate fixed-state mechanism evidence
**Chosen**: B.
**Rationale**: Beta-dependent choices change later states and therefore change conditional q. Original-factual companions provide exact pool/query matching; adaptive runs retain matched per-expansion M/B and report realized cumulative oracle work.

### D-8: Qualify factual identity by partition
**Date**: 2026-09-10 - **Stage**: pre-review correction
**Options**: A) compare bare local validation/test indices B) use `(partition, local_source_index)` C) recover global upstream row IDs
**Chosen**: B now; C only if the provider later exposes authenticated global IDs.
**Rationale**: Validation and test are physically disjoint arrays but reuse local integer coordinates. The qualified key is sufficient for case identity, pairing, and leakage checks without changing dataset provenance.

### D-9: Add target stratification without changing historical stratification
**Date**: 2026-09-10 - **Stage**: pre-review correction
**Options**: A) reinterpret `stratified` B) add `target_stratified` C) keep truth-label balance and rename the experiment text
**Chosen**: B.
**Rationale**: Existing `stratified` selects on true labels before predictions exist and is a compatibility contract. The new policy predicts the complete partition, derives opposite targets, and deterministically selects within target strata. This supersedes D-6's inaccurate phrase that called the existing benchmark policy target-stratified.

### D-10: Seal canonical campaigns outside immutable run directories
**Date**: 2026-09-10 - **Stage**: pre-review correction
**Options**: A) mutate completed runs to add hashes B) version the entire artifact schema C) add an external campaign inventory and strict reader
**Chosen**: C.
**Rationale**: Current canonical `COMPLETE` markers carry no payload digests. An atomically published external inventory adds mutation detection without rewriting historical or newly completed run directories.

---
