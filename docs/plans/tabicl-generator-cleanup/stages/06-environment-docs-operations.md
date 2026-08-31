# Stage 6: Create the Focused uv Environment and Documentation

**Goal**: Make the retained suite reproducible from a fresh checkout and document only its generator, benchmark, baselines, data bootstrap, and Athena workflows.
**Dependencies**: Stage 5

---

## Steps

1. Add suite-owned dependency metadata and a tracked lock under `experiments/zeroshot_cf`.
   - Declare direct runtime/test dependencies only, including exact `tabicl==2.1.1`, NumPy,
     PyTorch, pandas, sklearn, SciPy, PyYAML, joblib, Hugging Face Hub, typing extensions,
     DiCE/raiutils, and the retained pinned CEL source.
   - Remove root editable TabPFN and `tabpfn-extensions`; configure non-zero focused pytest
     discovery. Do not copy the root project's invalid or unrelated resolver/tool settings.
2. Make vendor/data bootstrap reproducible.
   - Document the pinned CEL revision, required four config/data files, validation command, cache
     location, and dataset provenance/license caveat.
   - Setup must be idempotent and must not replace a mismatched user checkout without an explicit
     error and recovery instruction.
3. Rewrite `experiments/zeroshot_cf/README.md` around the retained suite.
   - Cover setup, public generator API, single-CF/refinement and multi-CF/diversity modes, Exp9
     protocol, four baselines, outputs, checkpoint staging, offline smoke, tests, and Athena.
   - State that targets follow classifier predictions and that actionability currently means
     immutable preservation plus atomic categories, without directional/causal constraints.
4. Align checkpoint, output, and Athena contracts.
   - Preserve checksum verification and offline runtime; `--help` must not access network/weights.
   - Ignore generated `results/local/` and `results/athena/` consistently while preserving tracked
     source assets. Verify the four Athena cases and environment variables.
5. Review license/notice coverage for TabICL, CEL, DiCE, and retained Prior Labs files. Record any
   unresolved legal action in the backlog; do not delete or rewrite root notices speculatively.

---

## Verification

- [ ] GATE `uv sync --project experiments/zeroshot_cf --python 3.12 --locked` succeeds from the suite manifest/lock without installing the root project or `tabpfn-extensions` — dependency metadata and lock are the inputs; missing/legacy dependencies turn it red.
- [ ] GATE `uv run --project experiments/zeroshot_cf pytest -q` discovers a non-zero focused suite and passes — local pytest configuration, retained tests, and production modules are the inputs; zero discovery or regression turns it red.
- [ ] GATE Exp8/9/11/12/13/14 and checkpoint/smoke `--help` commands documented in the README execute in the locked environment with `HF_HUB_OFFLINE=1` — stale docs, eager network, or checkpoint loading turns the relevant command red.
- [ ] GATE `bash -n` passes for both Athena shell scripts and the case TSV contains exactly four non-comment cases — tracked Athena inputs are measured; shell drift or dataset omission turns it red.
- [ ] REPORT Record direct dependency list, resolved CEL/TabICL versions, and notice review result in `journal.md`; unresolved legal questions open a backlog item and continue.

---

## Commit

`build(tabicl-cf): isolate environment and operations`
