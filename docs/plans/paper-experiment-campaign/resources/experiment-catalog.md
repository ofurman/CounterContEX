# Experiment catalog

Maps the E-numbers from [`docs/papers/positioning-draft.md`](../../../papers/positioning-draft.md)
§5 to the stage that owns them, the matrix that defines them, and the output root that holds
their artifacts. Output roots are fixed here so a resumed or re-examined stage writes to the
same place.

| E | Name | Stage | Matrix | Output root under `results/` |
|---|---|---|---|---|
| E1 | Main comparison, k=1 | 7 | `campaign_e1_main.yaml` | `campaign/e1_main` |
| E2 | Diverse sets, k=3 vs DiCE | 8 | `campaign_e2_diverse.yaml` | `campaign/e2_diverse` |
| E3 | Backend ablation | 8 | `campaign_e3_backend.yaml` | `campaign/e3_backend` |
| E4 | Confidence / τ Pareto | 9 | `campaign_e4_confidence.yaml` | `campaign/e4_confidence` |
| E5 | Search + diversity ablations | 10 | `campaign_e5_search.yaml` | `campaign/e5_search` |
| E6 | Context ablation | 10 | `campaign_e6_context.yaml` | `campaign/e6_context` |
| E7 | Cost-quality Pareto | 10 | `campaign_e7_cost.yaml` | `campaign/e7_cost` |
| E8 | Robustness to retraining | 11 | none — scoring pass over E1 arrays | `campaign/e8_robustness` |
| E9 | Foundation-model swap | 11 | `campaign_e9_fmswap.yaml` | `campaign/e9_fmswap` |
| E10 | Frozen headline, n=1000 | 12 | `campaign_e10_headline.yaml` | `campaign/e10_headline` |

## Gaps closed

| Gap | Description | Closed by |
|---|---|---|
| B1 | One target classifier | Stages 2, 7 |
| B2 | One seed, no significance | Stages 5, 7 |
| B3 | Four datasets | Stages 1, 6, 7 |
| B4 | k-mismatch in the headline table | Stages 4, 7, 8 |
| B5 | Zero ablations executed | Stages 8, 10 |
| B6 | Threshold validity 0.000 at τ=0.7 | Stages 3, 9 |
| B7 | Runtime unexplained | Stages 1, 10 |
| B8 | Plausibility evidence inadequate | Stages 3, 12 |
| B9 | No qualitative example | Stage 12 |
| B10 | No robustness to model multiplicity | Stage 11 |
| B11 | TabICL-specific, not FM-general | Stage 11 |

B12 (directional constraints), B13 (multiclass), B14 (formal statement) and B15 (human
evaluation) are **out of scope for this plan** — B12 and B13 are method extensions rather than
experiments, B14 is writing, B15 needs a study protocol. Record them in `backlog.md` at the end
of the run so the next plan picks them up.

## Shared protocol for every campaign matrix

Unless a stage brief says otherwise:

```yaml
seeds: [17, 42, 101, 202, 303]
protocol:
  max_test: 250
  test_selection: stratified
evaluation:
  probability_threshold: 0.7
legacy_export: false
```

Datasets: `heloc`, `bank_marketing`, `give_me_some_credit`, `lending_club`, `adult_census`,
`german_credit`. Target models: `retained_logistic_regression`, `retained_mlp`,
`retained_xgboost`.

**`legacy_export: false` everywhere.** The v1 flat export has a known defect on resumed
CounterContEx runs when a stored diagnostic is JSON `null` (see the root README). The canonical
artifact directory is the campaign's record; nothing in this plan reads the flat CSVs.
