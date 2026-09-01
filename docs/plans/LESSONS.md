# Lessons

Durable, cross-plan facts. One line each, linking to the plan that learned it.

This is the file the next plan's author reads before exploring, and the one an unattended
run can reach when no memory server is running. Add a line when a fact cost someone a run
to learn -- especially one that has now been re-derived from scratch in more than one plan.

Keep it short. A fact that only matters inside one plan belongs in that plan's `journal.md`.

| Fact | Learned in | Date |
|------|-----------|------|
| The TabICL counterfactual path can pull legacy TabPFN and CEL at import time through small shared helpers/types; extract action types and candidate projection before deleting legacy experiments. | [tabicl-generator-cleanup](tabicl-generator-cleanup/index.md) | 2026-08-30 |
| Exp9 dataset loading requires both CEL code and configs/CSVs under a gitignored vendor checkout; pin the vendor revision and freeze split/feature/action-space fingerprints before changing data ownership. | [tabicl-generator-cleanup](tabicl-generator-cleanup/index.md) | 2026-08-30 |
| The tracked upstream TabPFN implementation is under `src/tabpfn`; the top-level `tabpfn/` contains only ignored bytecode/cache artifacts, so a complete root cleanup must remove both paths. | [tabicl-generator-cleanup](tabicl-generator-cleanup/index.md) | 2026-08-30 |
| The retained common metric kernels are reusable, but Exp9/11–14 still duplicate lifecycle, result shaping, persistence, and aggregation; the stable seam is a canonical generation result consumed by a method-blind evaluator. | [counterfactual-evaluation-architecture](counterfactual-evaluation-architecture/index.md) | 2026-08-31 |
| The 24-cell full reference matrix took 9.42 measured hours, including 7.64 hours for DiCoFlex/Lending Club; use deterministic contract and one-factual gates during refactors and reserve the full matrix for a final REPORT. | [counterfactual-evaluation-architecture](counterfactual-evaluation-architecture/index.md) | 2026-08-31 |
| A method registry name and implementation version both contribute to content-addressed experiment identity; renaming a method must create new run IDs and must not rewrite or resume manifests created under the old identity. | [countercontex-method-rename](countercontex-method-rename/index.md) | 2026-09-01 |
