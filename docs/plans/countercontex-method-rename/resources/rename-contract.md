# CounterContEx Rename Contract

## Canonical mapping

| Surface | Current | Canonical |
|---------|---------|-----------|
| Human method name | `DiCoFlex` | `CounterContEx` |
| Python type prefix | `DiCoFlex*` | `CounterContEx*` |
| Package and registry key | `dicoflex` | `countercontex` |
| Implementation identity | `dicoflex-v3` | `countercontex-v3` |
| Exp9 module and operational slug | `exp9_dicoflex_*` | `exp9_countercontex_*` |

`CounterContEX` is the repository/project title, not the method spelling. The matrix schema
namespace `countercontex.*` already has the correct machine form and is not versioned solely for
this rename.

## Audited active surface

Planning found only two tracked old-name variants: `DiCoFlex` and `dicoflex`. The repository has
392 tracked occurrences across 56 files; 90 are in completed plan history, leaving 302 active
code, test, config, script, and documentation references. Under tracked `README.md` and
`experiments/`, Git tracks 17 paths containing the old machine name.

The main active path families are:

- `experiments/zeroshot_cf/methods/dicoflex/` (10 modules)
- `experiments/zeroshot_cf/exp9_dicoflex_benchmark.py`
- `experiments/zeroshot_cf/athena/exp9_dicoflex_*` and `submit_exp9_dicoflex.sh`
- `experiments/zeroshot_cf/configs/matrices/dicoflex_ablation_example.yaml`
- method-specific test modules

Direct and indirect consumers also live in the method registry, Exp8 compatibility runner,
orchestration legacy/v1 code, evaluation helpers, matrix configs, compatibility fixture, root and
suite READMEs, and Athena README.

## Identity and artifact rules

- `MethodSpec.name` is identity-bearing and changes `cell_id`.
- The registry implementation version is identity-bearing and changes `run_id`.
- New CounterContEx runs must not resume old-name manifests.
- Do not rewrite an old manifest or rename a content-addressed run directory in place.
- The v1 export's `exp9_tabicl_*` stems, ordered CSV fields, NPZ keys, and `tabicl_v2_*` method
  values are artifact compatibility data, not current method branding. Preserve exact schema and
  IDs, all deterministic CSV values, and every NPZ array value/dtype/shape. Compare variable timing
  fields by presence/type only rather than literal value.
- The v1 dispatch dictionary key and current CLI module name are active interfaces and change to
  the canonical machine name.

## Intentional historical exclusions

Completed directories `docs/plans/tabicl-generator-cleanup/` and
`docs/plans/counterfactual-evaluation-architecture/` contain append-only journals, decisions,
state, stage filenames, measurements, and architecture snapshots. Preserve them verbatim.

Ignored local results include old-name manifests under `architecture_one_factual`, the completed
cell under `architecture_full_reference`, and older Exp9 output roots. They are historical,
content-addressed data. Inventory them as a REPORT; do not edit, delete, rename, resume, or rerun
them.

The final zero-match gate covers tracked root `README.md`, `experiments/`, and the active durable
`docs/plans/LESSONS.md`. It must not acquire broad exclusions inside that active scope. This plan
and predecessor plan directories may name the old surface because they document the migration and
historical evidence.

Ignored `__pycache__` directories beneath the old method package are disposable workspace state,
not historical experiment artifacts. Remove them after the tracked package move and verify with
`importlib.util.find_spec()` that the old package no longer resolves as a namespace package.
