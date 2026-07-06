# Experiment 6: Context Ablation — size × strategy

Two-factor grid (size `{256, 512, 1024, 2048}` × strategy `{random_target, random_both, knn_target, knn_both}`) at the Stage-2 winning selector, on each dataset.
Held identical across all cells **within a dataset**: selector, `temperature=1e-9` (MAP commit), `n_permutations`, `tau`, `n_test` (= `--max-test`).

> **Strategy = class scope × selection.** `*_target` draws context from the per-point target-class pool; `*_both` from all training rows. `random_*` uniformly subsamples (one fit per class batch, reused); `knn_*` keeps the `size` nearest neighbours to each factual point (re-fit per test point, Decision #5).
> **`effective_size`** is the rows actually used, capped at `size` and at the available pool (`effective_size <= size`, `<= pool_size`).
> If the selector is `class_divergence`, the 8 `*_target` cells are skipped (it needs a both-classes pool); only 8 `*_both` cells appear.

## MOONS

Selector: `prob_ascent` · cells: 16 · n_test: 100

**validity**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 0.7 | 0.73 | 0.63 | 0.37 |
| 512 | 0.71 | 0.82 | 0.71 | 0.56 |
| 1024 | 0.71 | 0.82 | 0.71 | 0.82 |
| 2048 | 0.71 | 0.82 | 0.71 | 0.82 |

**l0_count_mean**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 1.271 | 1.342 | 1.381 | 1.297 |
| 512 | 1.254 | 1.329 | 1.254 | 1.232 |
| 1024 | 1.254 | 1.341 | 1.254 | 1.341 |
| 2048 | 1.254 | 1.341 | 1.254 | 1.341 |

**proximity_l2_jaccard**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 0.5975 | 0.6352 | 0.5232 | 0.2634 |
| 512 | 0.6138 | 0.6589 | 0.6138 | 0.3841 |
| 1024 | 0.6138 | 0.6605 | 0.6138 | 0.6605 |
| 2048 | 0.6138 | 0.6605 | 0.6138 | 0.6605 |

**frac_oob**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 0 | 0 | 0 | 0 |
| 512 | 0 | 0 | 0 | 0 |
| 1024 | 0 | 0 | 0 | 0 |
| 2048 | 0 | 0 | 0 | 0 |

**lof_scores_cf**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 1.009 | 1.039 | 1.021 | 1.068 |
| 512 | 1.006 | 1.027 | 1.006 | 1.169 |
| 1024 | 1.006 | 1 | 1.006 | 1 |
| 2048 | 1.006 | 1 | 1.006 | 1 |

**Auto-derived best cells:**

- **Best validity cell**: size=512, strategy=random_both (validity=0.82, frac_oob=0, l0_count_mean=1.329)
- **Best frac_oob cell**: size=512, strategy=random_both (frac_oob=0, validity=0.82, lof_scores_cf=1.027)

## HELOC

Selector: `prob_ascent` · cells: 16 · n_test: 15

**validity**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 0.7333 | 0.6667 | 0.7333 | 0.6667 |
| 512 | 0.6667 | 0.6667 | 0.8 | 0.6667 |
| 1024 | 0.7333 | 0.4 | 0.6667 | 0.6667 |
| 2048 | 0.4667 | 0.4667 | 0.6667 | 0.7333 |

**l0_count_mean**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 1.636 | 1.6 | 1.636 | 1.5 |
| 512 | 1.5 | 1.5 | 1.833 | 1.5 |
| 1024 | 1.636 | 1.5 | 1.5 | 1.5 |
| 2048 | 1.714 | 1.714 | 1.5 | 1.636 |

**proximity_l2_jaccard**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 0.6371 | 0.6098 | 0.6515 | 0.6225 |
| 512 | 0.6107 | 0.6102 | 0.6915 | 0.6204 |
| 1024 | 0.6649 | 0.752 | 0.6388 | 0.6098 |
| 2048 | 0.6801 | 0.6798 | 0.6355 | 0.6654 |

**frac_oob**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 0.1333 | 0.1333 | 0.1333 | 0 |
| 512 | 0.1333 | 0.1333 | 0.2 | 0.2 |
| 1024 | 0.2667 | 0.4667 | 0.2667 | 0.2 |
| 2048 | 0.5333 | 0.4667 | 0.2 | 0.1333 |

**lof_scores_cf**

| size \ strategy | random_target | random_both | knn_target | knn_both |
|---|---|---|---|---|
| 256 | 3.164e+07 | 1.024e+07 | 1.609e+06 | 1.982 |
| 512 | 1.016e+07 | 1.591e+07 | 1.127e+06 | 7.202e+05 |
| 1024 | 1.070e+10 | 9.940e+06 | 4.109e+06 | 1.461e+06 |
| 2048 | 6.342e+06 | 1.145e+07 | 3.131e+06 | 2.480e+06 |

**Auto-derived best cells:**

- **Best validity cell**: size=512, strategy=knn_target (validity=0.8, frac_oob=0.2, l0_count_mean=1.833)
- **Best frac_oob cell**: size=256, strategy=knn_both (frac_oob=0, validity=0.6667, lof_scores_cf=1.982)

## Verdict (recommended context config)

Run on the remote DGX GPU (NVIDIA GB10). MOONS n_test=100; **HELOC n_test=15**, bounded
for runtime (the size-2048 kNN cells dominate cost — the full grid took ~5.3 h even at
n=15). HELOC validity values are therefore noisy at ±0.12 granularity; the **`frac_oob` /
`LOF` (plausibility) trends are the robust signal**, and they are monotone and consistent.

**1. Does larger context help HELOC? — No; it *hurts*.** For the `random_*` strategies,
plausibility degrades monotonically with size: `frac_oob` rises 256→2048 from 0.13→0.53
(`random_target`) and 0.13→0.47 (`random_both`), and validity falls (e.g. `random_both`
0.67→0.40→0.47; `random_target` 0.73→0.47). A larger *random* context pulls in rows far
from the factual point, diluting the local conditioning that the dense single-column
greedy step relies on. This **refutes hypothesis (i)** (larger context → better) for HELOC.

**2. Does kNN beat random? — Yes, decisively for plausibility, especially at large size.**
At every size the kNN cells have far lower `LOF` than random (kNN ≈ 1e6 vs random 1e7–1e10),
and at size 2048 kNN holds `frac_oob` at 0.13–0.20 while random blows up to 0.47–0.53.
kNN's relevance-based selection is what protects against the large-context degradation in
(1). **Hypothesis (ii) holds.**

**3. Both-classes vs target-only?** Mixed for validity (noisy at n=15), but the single
best plausibility cell is **`knn_both` at size 256: `frac_oob = 0.000`, `LOF = 1.98`** —
i.e. *every* counterfactual in-distribution, vs `LOF` in the millions–billions for every
other HELOC cell. Drawing the nearest neighbours from *both* classes at a *small* size
gives the tightest on-manifold context.

### Recommended (selector, size, strategy)

- **HELOC → `(prob_ascent, size=256, knn_both)`.** The plausibility standout: `frac_oob`
  0.000, `LOF` 1.98, `l0_count_mean` 1.50, validity 0.67. Versus the predecessor baseline
  `(256, random_target)` (validity 0.73 but `frac_oob` 0.13, `LOF` 3.2e7) it trades ~0.07
  validity for a *qualitative* plausibility jump (LOF 3.2e7 → 2.0). If validity is the sole
  priority, `(512, knn_target)` gives 0.80 but at `frac_oob` 0.20 — still a kNN cell.
- **MOONS → `(prob_ascent, size=512, random_both)`.** Validity 0.82 (the dataset's near-MAP
  ceiling — Decision #9), `frac_oob` 0, `LOF` 1.03. MOONS plausibility is trivially fine
  everywhere (`frac_oob ≡ 0`); size **saturates by ~512** (effective_size caps at the
  ~403/class or 800-row pool, Decision #7) and kNN gives no benefit on easy 2-D data. At
  saturation `knn_both` ties `random_both` (both draw the whole pool).

### Bottom line

Greedy + a **small, relevance-selected (kNN) context** is what salvages HELOC. Combined
with Stage 2 (greedy + `prob_ascent` lifts HELOC validity 0.538 → 0.90 and L0 17 → 1.67 vs
one-pass), Stage 4 shows the remaining plausibility gap closes with `knn_both@256`
(`frac_oob` 0.65 → 0.00). **Bigger context is not the lever — relevant context is.**
