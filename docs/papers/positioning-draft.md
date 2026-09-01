# CounterContEx: Positioning Draft and Paper Completion Plan

Status: draft for discussion. Target venues: NeurIPS (Datasets & Benchmarks or main track)
or ICLR main track. Written 2026-09-01 from the state of this repository at `e1195ce`.

Every number quoted below is measured and traceable to an artifact in this repository.
Sources are named inline. Nothing in this document is an estimate unless it says so.

---

## 1. What the repository actually contains

### 1.1 The method in one sentence

CounterContEx runs a classifier-guided discrete search over feature values that a
**tabular foundation model proposes from a per-factual, target-conditioned in-context
set** — the foundation model never decides validity, and nothing is trained per dataset.

### 1.2 The mechanism, stated precisely

For a factual row `x` and target class `y*`:

1. **Local context.** Select up to 512 Gower-nearest training rows. Label them with the
   *explained classifier's own predictions*, not ground truth. Encode each one-hot group
   as one categorical identifier so a 12-level variable does not out-weigh a numerical one.
2. **Conditional proposal.** Mask feature `j`, append the requested class, and query
   `p_theta(X_j | X_-j = z_-j, Y = y*)`. Take the mode of the piecewise-quantile density,
   or a 9-point quantile grid. Optionally condition on a confidence anchor `C = c`
   derived from `p_f(y* | x_i)` over target-class context rows.
3. **Legal one-action trials.** Project to training bounds, snap to observed support when
   a feature has <= 20 unique values, swap one-hot groups atomically, hold immutables fixed.
4. **Classifier decides.** Score every complete trial in one batch with `f`. Validity is
   `f(x') = y*` and `p_f(y* | x') >= tau`.
5. **Search.** `k = 1`: greedy target-probability ascent, return the closest valid row at
   first crossing. `k > 1`: bounded beam with a niche rule over changed-action sets, a
   quality filter relative to the closest valid member, then exact fixed-size DPP MAP.

The load-bearing structural claim is the **decoupling**: the foundation model is a
*proposal distribution*, the explained classifier is the *sole oracle*. This is why the
method needs no gradients, no differentiable surrogate, and no per-dataset training.

### 1.3 The benchmark harness

| Property | Value |
|---|---|
| Datasets | HELOC, Bank Marketing, Give Me Some Credit, Lending Club |
| Split | deterministic 64/16/20, seed 42 |
| Target classifier | logistic regression, `C=1.0`, `max_iter=1000` |
| Factuals | 1,000 per dataset, deterministic stratified |
| Baselines | NICE, Wachter, Growing Spheres, DiCE, FACE |
| Metrics | coverage, class validity, threshold validity, grouped Gower / Manhattan / Euclidean proximity, sparsity, action-unit sparsity, immutable actionability, LOF, Isolation Forest, out-of-bounds, set diversity (action Jaccard, pairwise Gower), phase timings |
| Reproducibility | content-addressed `run_id` from resolved scientific identity; `COMPLETE` marker published after payloads; aggregation rejects missing/extra/partial/mismatched cells |
| Tests | 188 `def test_` across 33 test modules |

This harness is a genuine asset and is currently under-sold. It is the strongest
"we did the science properly" evidence in the repository.

### 1.4 Measured headline results (24-cell run, 9.42 h)

Source: `experiments/zeroshot_cf/results/local/full_reference/*/**_metrics.csv`.
CounterContEx appears under its pre-rename identity `tabicl_v2_diverse_dpp`.
**Caveat that must be fixed before publication:** CounterContEx ran at `k=3`, all
baselines at `k=1`, and the proximity/sparsity columns aggregate over all returned
candidates. This is not a like-for-like comparison yet.

| Dataset | Method | Coverage | Validity | Grouped Gower | Action units | LOF (CF) | LOF (test) | Gen. time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bank_marketing | **CounterContEx** | 1.000 | 1.000 | **0.0197** | **1.047** | 2.007 | 1.643 | 1,981 |
| bank_marketing | nice | 1.000 | 1.000 | 0.0309 | 1.229 | 1.839 | 1.643 | 1.43 |
| bank_marketing | wachter | 1.000 | 0.998 | 0.0636 | 1.571 | 4.066 | 1.643 | 9.68 |
| bank_marketing | growing_spheres | 1.000 | 1.000 | 0.0745 | 1.534 | 3.563 | 1.643 | 3.15 |
| bank_marketing | face | 1.000 | 1.000 | 0.0816 | 3.492 | 1.724 | 1.643 | 14.46 |
| bank_marketing | dice | 1.000 | 1.000 | 0.1255 | 2.527 | 2.549 | 1.643 | 124.0 |
| heloc | **CounterContEx** | 1.000 | 0.997 | 0.0133 | **1.368** | 1.186 | 1.106 | 2,606 |
| heloc | wachter | 1.000 | 0.989 | **0.0125** | 1.211 | 1.187 | 1.106 | 12.7 |
| heloc | nice | 1.000 | **0.631** | 0.0164 | 4.173 | 1.104 | 1.106 | 1.40 |
| heloc | growing_spheres | 1.000 | 1.000 | 0.0165 | 2.461 | 1.198 | 1.106 | 7.82 |
| heloc | dice | 1.000 | 0.929 | 0.0189 | 2.013 | 1.177 | 1.106 | 68.3 |
| heloc | face | 1.000 | 1.000 | 0.0439 | 7.040 | 1.095 | 1.106 | 25.7 |
| gmsc | **CounterContEx** | 1.000 | 1.000 | 0.0759 | 1.084 | **12.098** | 1.305 | 1,161 |
| gmsc | nice | 1.000 | 1.000 | **0.0545** | 1.423 | 1.713 | 1.305 | 1.32 |
| gmsc | face | 1.000 | 1.000 | 0.0689 | 1.984 | 1.410 | 1.305 | 103.4 |
| gmsc | wachter | 1.000 | 0.999 | 0.0902 | 1.042 | 1.227 | 1.305 | 10.2 |
| gmsc | growing_spheres | 1.000 | 1.000 | 0.0937 | 1.039 | 16.621 | 1.305 | 2.20 |
| gmsc | dice | 1.000 | 0.999 | 0.1058 | 1.086 | 2.594 | 1.305 | 52.0 |
| lending_club | **CounterContEx** | 1.000 | 1.000 | 0.0872 | 1.289 | 1.838 | 1.158 | **27,519** |
| lending_club | wachter | 1.000 | 0.999 | **0.0766** | 1.189 | 1.831 | 1.158 | 11.1 |
| lending_club | growing_spheres | 1.000 | 1.000 | 0.0911 | 1.255 | 1.821 | 1.158 | 2.52 |
| lending_club | nice | 1.000 | 1.000 | 0.0958 | 1.526 | 1.437 | 1.158 | 1.37 |
| lending_club | dice | 1.000 | 0.999 | 0.1064 | 1.534 | 1.533 | 1.158 | 73.2 |
| lending_club | face | 1.000 | 1.000 | 0.1598 | 5.113 | 1.237 | 1.158 | 96.2 |

**What the numbers currently support:**

- Best or near-best grouped-Gower proximity on 2/4 datasets and best action-unit
  sparsity on 2/4, *while returning three candidates instead of one*.
- Very stable validity (0.997–1.000) where NICE collapses to 0.631 on HELOC and DiCE to
  0.929. Robust validity across datasets is a real, defensible strength.
- Full coverage everywhere.

**What the numbers currently do not support, and reviewers will say so:**

- **Plausibility.** LOF for CounterContEx CFs exceeds the LOF of real test data on 3/4
  datasets, and on GMSC by 9.3x (12.10 vs 1.30). A method whose selling point is a
  data-distribution-aware proposal model cannot ship a table that says its outputs are
  more outlying than the data. Either the metric is the wrong instrument (defensible —
  LOF on a 44-column one-hot space is noisy) or the mechanism is not doing what the
  narrative says. This must be resolved, not glossed.
- **Cost.** 27,519 s vs 1.37 s for NICE on Lending Club is a 20,000x gap. Lending Club
  is not the widest dataset (32 columns vs GMSC's 44) yet costs 10x more per factual,
  which points at search depth or categorical breadth, not raw dimensionality. Unprofiled.
- **Threshold validity.** On the canonical HELOC run
  (`architecture_full_reference/cf9d0c3a.../summary.csv`),
  `validity_returned_threshold = 0.000` at tau = 0.7 across all 2,991 returned candidates.
  Every counterfactual sits in the band `0.5 <= p_f(y*) < 0.7`. The excellent proximity is
  *partly an artifact of stopping at the first crossing of 0.5*. This is the single most
  exploitable finding in the repository — see section 5.

---

## 2. Where CounterContEx sits in the literature

### 2.1 The four neighbourhoods

**(a) Classical tabular counterfactual / recourse methods.** Wachter et al. (gradient
descent on a differentiable loss), DiCE (diversity via a DPP-style determinant term),
NICE (nearest unlike neighbour + greedy feature substitution), FACE (feasible paths over
a density-weighted graph), Growing Spheres (random sampling in growing annuli). All are
in this repository as baselines. None of them models `p(X_j | X_-j, Y = y*)`.

**(b) Plausible / on-manifold counterfactuals via a trained generative model.**
C-CHVAE, REViSE, CRUDS, CLUE (VAE latents); TABCF (transformer VAE with a Gumbel-Softmax
detokenizer, CIKM 2024); Tabular Diffusion Counterfactual Explanations (2025);
DensityFlow / density-guided robust CE under model multiplicity (2026). **Every one of
these trains a dataset-specific generative model as a prerequisite.** This is the
neighbourhood CounterContEx displaces, and the displacement is the paper.

**(c) Tabular foundation models.** TabPFN and TabPFN v2 (Nature 2025), TabICL
(ICML 2025) and TabICLv2, TabDPT, and the generative offshoots: TabPFGen (turns TabPFN
into an energy-based generator), UnmaskingTrees / TabPFN-as-imputer (masked-feature
conditional generation). These supply the *capability* CounterContEx consumes.

**(d) Explainability built on tabular foundation models — the newest and closest.**
ExplainerPFN (Jan 2026): zero-shot Shapley-value estimation without model access.
KernelICL: in-context kernel regression for sample-based interpretability.
"Real-Time Explanations for Tabular Foundation Models" (2026). **This neighbourhood is
entirely attribution-based. No counterfactual member exists yet.** That gap is the claim.

**(e) Benchmarks and the reproducibility critique.** CARLA (NeurIPS 2021 D&B),
RecourseBench (2026), CounterEval (AAAI 2026), and **CEL: Comprehensive Counterfactual
Explanations Library and Benchmark (arXiv 2607.22045)** — of which you are first author,
and which this repository vendors at a pinned revision. CEL must be cited prominently and
differentiated explicitly: CEL is the measuring instrument, CounterContEx is the method.
Reviewers who notice the shared authorship and *don't* see it declared will treat it badly.

### 2.2 Differentiation table (this is a figure/table for the paper)

| Approach | Per-dataset training | Needs gradients | Models `p(x)` | Native `k>1` | Model-agnostic |
|---|:--:|:--:|:--:|:--:|:--:|
| Wachter | no | **yes** | no | no | no |
| Growing Spheres | no | no | no | no | yes |
| NICE | no | no | implicit (NUN) | no | yes |
| FACE | no (graph build) | no | kernel density | no | yes |
| DiCE (genetic) | no | no | no | **yes** | yes |
| C-CHVAE / REViSE / CRUDS | **yes** (VAE) | usually | yes | limited | partly |
| TABCF | **yes** (transformer VAE) | yes | yes | limited | no |
| Tabular diffusion CE | **yes** (diffusion) | yes | yes | yes | partly |
| DensityFlow | **yes** (flow) | yes | yes | yes | partly |
| **CounterContEx** | **no** | **no** | **yes, in-context** | **yes (beam + DPP)** | **yes** |

The bottom row is the only one with no "yes" in the first two columns and a "yes" in the
last three. **That row is the paper.**

### 2.3 Honest novelty risks — address these before a reviewer does

1. **Masked-feature conditional query is a known TFM capability**, used for imputation and
   for generation (TabPFGen, UnmaskingTrees). Do not claim the mechanism. Claim the
   *application*: a per-factual, classifier-labelled, target- and confidence-conditioned
   context used as a proposal distribution inside a validity-constrained discrete search.
2. **TabPFGen already makes a TFM generative.** Differentiate: TabPFGen does
   class-conditional *unconditional-per-row* sampling via an energy function. CounterContEx
   does per-factual *conditional* proposals over one masked coordinate at a time, inside a
   search whose oracle is a different model. Consider adding TabPFGen-sample-then-filter as
   a baseline — it is the most obvious "why not just do this?" question, and beating it
   cheaply is a strong result.
3. **NICE is conceptually adjacent** (greedy feature substitution toward a nearest unlike
   neighbour). The differentiator is that NICE substitutes *an observed neighbour's actual
   value*, while CounterContEx substitutes *a value sampled from a learned conditional
   posterior*. Make this contrast an explicit, illustrated figure. It is the cleanest way
   to explain the idea in 30 seconds.
4. **The empirical backend is your own strongest attack.** If checkpoint-free target-class
   quantiles match TabICL, the foundation model is decorative. You must run this ablation
   and report it whichever way it lands. Reviewers will assume you hid it if it is absent.

---

## 3. The pitch

### 3.1 Title options

1. **Counterfactuals in Context: Training-Free Counterfactual Explanations with Tabular
   Foundation Models** — clearest, names both halves.
2. **The Foundation Model Proposes, the Classifier Disposes: Decoupled Counterfactual
   Search for Tabular Data** — memorable, states the architecture.
3. **CounterContEx: In-Context Conditional Proposals for Plausible, Diverse, Training-Free
   Counterfactual Explanations** — safest, most conventional.

Recommendation: (1) for the title, (2) as the framing sentence in the introduction.

### 3.2 Draft abstract

> Counterfactual explanations that stay close to the data distribution currently require
> training a dataset-specific generative model — a VAE, a normalizing flow, or a diffusion
> model — before a single explanation can be produced. We show this step is unnecessary.
> We introduce CounterContEx, which uses a pretrained tabular foundation model as a
> *training-free conditional proposal distribution* inside a classifier-guided discrete
> search. For each factual instance we assemble a local in-context set labelled by the
> explained classifier's own predictions, and query the foundation model for
> `p(X_j | X_-j, Y = y*)` one masked feature at a time; the explained classifier alone
> decides validity. The design requires no gradients, no differentiable surrogate, no
> per-dataset fitting, and treats the explained model as a black box. A bounded beam search
> with determinantal point process selection returns diverse counterfactual sets without
> padding. Across N tabular benchmarks and M classifier families, CounterContEx attains
> [validity/proximity/sparsity claims] while [plausibility claim], and its confidence
> conditioning exposes a control knob that no prior method offers: the ability to request
> counterfactuals at a specified target-class confidence, rather than accepting whatever
> the decision boundary yields. We release the full benchmark harness with content-addressed
> run identities and strict metric denominators.

Bracketed slots are deliberate. Fill them only after section 4 is executed.

### 3.3 Contribution list

1. **A training-free plausibility mechanism.** Foundation-model in-context conditional
   proposals replace the per-dataset generative model that on-manifold CF methods require.
2. **Proposal/oracle decoupling.** The generative model and the explained model are
   separate, so the method is gradient-free, black-box, and model-agnostic by construction.
3. **Confidence-conditioned counterfactuals.** A new controllable axis, motivated by the
   measured finding that sparse CF methods systematically hug the decision boundary.
4. **A reproducible evaluation protocol** with separated availability/validity denominators,
   content-addressed run identity, and no-padding candidate-set semantics — released.

Contribution 3 is currently a *capability with no experiment behind it*. Section 5 fixes that.

---

## 4. What must be added before submission

Ordered by severity. Items 1–6 are, in my judgement, individually sufficient grounds for
rejection at NeurIPS or ICLR.

### Blocking

**B1. One target classifier is not enough.**
Every result uses logistic regression. A reviewer's first sentence will be "the method is
only validated on a linear model." Add at least an MLP and a gradient-boosted tree
(XGBoost or LightGBM). This is not merely defensive — **it is where you win**: Wachter
requires gradients and cannot explain the tree at all, so a three-classifier table converts
a weakness into the model-agnosticism contribution's evidence.
Cost: 3x the current matrix. Priority: highest.

**B2. One seed, no variance, no significance.**
`seeds: [42]`. Report mean +/- std over >= 5 seeds, and a paired statistical test across
the dataset x method grid (Wilcoxon signed-rank with Holm correction, or a Demsar critical
difference diagram). Without this, no proximity gap of 0.0133 vs 0.0125 means anything.

**B3. Four datasets is thin, and only two exercise the categorical path.**
CEL already ships 18. Target 8–10, chosen to stress the mechanism: Adult, German Credit,
COMPAS, Diabetes, Default of Credit Card Clients. Adult in particular has rich one-hot
groups and immutable attributes, which is where the atomic-group and immutability
contracts earn their keep.

**B4. The k-mismatch invalidates the headline table.**
CounterContEx returned 3 candidates; every baseline returned 1; aggregate proximity was
compared across them. Fix with two separate tables:
(a) **k = 1 head-to-head** using `primary_*` metrics only — the harness already computes
these, so this is a re-run, not new code;
(b) **k = 3 diversity comparison** against DiCE at k = 3, which natively supports it.
Currently no baseline in the results returns a set, so *no diversity claim is supported at
all*. `set_action_jaccard_mean = 0.634` on HELOC is unanchored without a comparator.

**B5. Zero ablations have been executed.**
`countercontex_ablation_example.yaml` exists and defines the right axes; there is no
evidence any cell was ever run, and `docs/countercontex-method.md` §11 lists four
hypotheses explicitly labelled as untested. Minimum viable ablation set:

| Ablation | Isolates | Reviewer question it answers |
|---|---|---|
| TabICL vs `empirical` backend | value of the foundation model | "Is the FM doing anything?" — **the** question |
| mode vs 9-point quantile grid | proposal breadth vs cost | "Why so slow?" |
| confidence conditioning on/off | contribution 3 | "What does the confidence knob buy?" |
| DPP vs random vs greedy-farthest pool selection | diversity machinery | "Is DPP worth the complexity?" |
| `sparse` vs `data_plausible` | joint-density refinement | "Does refinement improve plausibility?" |
| context size 64 / 128 / 256 / 512 | in-context set design | "Why 512?" |
| context labels: classifier predictions vs true labels | a genuinely interesting design choice | "Why predicted labels?" |

The last row is under-appreciated. Labelling the context with the classifier's *predictions*
rather than ground truth means the proposal model is conditioned on the model's view of the
world, not reality's. That is a defensible and interesting choice — but only if measured.

**B6. Threshold validity is 0.000 at tau = 0.7 and this is not discussed anywhere.**
Measured on the canonical HELOC run over 2,991 candidates. Two ways forward, and the second
is much better:
- *Defensive:* set tau = 0.5 for evaluation and stop reporting 0.7.
- *Offensive (recommended):* make it a headline finding. Show that **all** sparse CF methods
  hug the boundary, quantify it as a distribution of `p_f(y*|x')` per method, and then show
  confidence conditioning moving CounterContEx along a proximity-vs-confidence Pareto curve
  that no baseline can traverse. This converts your worst number into contribution 3's
  entire evidence base, and it is the most differentiated result available to you.

### Strongly expected

**B7. Runtime needs a section, not a footnote.**
20,000x NICE is not survivable as an unexplained table column. Required: GPU model and
count, wall-clock decomposition into prepare/generate/evaluate (the harness already records
these), per-factual cost, and a cost-quality Pareto plot with at least one deliberately
cheap configuration (mode-only proposals, k=1, context 128). Separately: **profile Lending
Club.** 27.5 s/factual against HELOC's 2.6 s, on a dataset with fewer columns than GMSC,
is a 10x anomaly that is currently unexplained and looks like an inefficiency, not a
property.

**B8. Plausibility evidence is inadequate and currently contradicts the narrative.**
LOF and Isolation Forest on one-hot-expanded features are weak instruments, and they
currently say CounterContEx CFs are *more outlying than real data*. Add measures that are
harder to dismiss:
- distance to k-th nearest training neighbour (in grouped-Gower space, matching the search);
- **detectability AUC**: train a discriminator to separate real rows from CFs — near-0.5 AUC
  is a strong, intuitive plausibility claim and is cheap to compute;
- per-feature marginal divergence (KS or total variation) between CF and target-class data;
- the TabICL joint log-density the method already computes, reported as an evaluation
  statistic across *all* methods (it is currently a method-internal ranking signal only —
  moving it into the evaluator makes it a method-blind metric, but note it then favours
  your own model class and must be reported alongside a neutral measure).

**B9. No qualitative example exists.**
One HELOC and one Adult case study: factual row, the three returned counterfactuals, which
features moved, and the same for two baselines. Reviewers use these to decide whether the
method is sensible. This is a half-day of work with the largest credibility-per-hour ratio
in this document.

**B10. Robustness to model multiplicity.**
Retrain the classifier under a different seed and measure what fraction of counterfactuals
remain valid. This is a live sub-field (DensityFlow, robust CE with probabilistic
guarantees) and the experiment is cheap: you already store all CF arrays, so it is a
scoring pass, not a generation pass. A boundary-hugging method is likely to do *badly*
here — which makes it another argument for the confidence-conditioning story in B6.

**B11. Foundation-model swap.**
Run one dataset with TabPFN v2 or TabICLv2 in place of TabICL. If results hold, the
contribution is "tabular foundation models enable this," not "TabICL enables this" — a
substantially stronger and more durable claim. The `ProposalSession` boundary in
`methods/countercontex/backends/` was built for exactly this and is currently unexercised
by a second foundation backend.

### Differentiating, if time allows

**B12. Directional and monotonic constraints.** `docs/countercontex-method.md` §7 states
plainly that these are unsupported. The recourse community will ask. A discrete
proposal-based search can enforce them almost for free by filtering the proposal set —
this is a genuinely cheap extension with high reviewer value.

**B13. Multiclass.** The target policy is binary-only by construction. Even one multiclass
dataset with a "nearest other class" policy widens applicability considerably.

**B14. A formal statement of what the search optimizes.** One short subsection framing the
greedy path as constrained first-crossing under a proposal-support prior, with the
optimality gap stated honestly. This is the cheapest available defence against "it is just
a pile of heuristics," which is the most likely form of a borderline reject.

**B15. Human or utility evaluation.** Upadhyay et al. (2025) report that counterfactuals
may not be the best recourse presentation at all. A paragraph engaging with this, or a
small user study, signals awareness of where the field is heading.

---

## 5. Recommended experimental programme

The compute is the constraint, so spend it deliberately. Measured CounterContEx cost per
factual, from the 24-cell run: HELOC 2.61 s, Bank Marketing 1.98 s, GMSC 1.16 s,
Lending Club 27.52 s (mean ~8.3 s, median ~2.3 s).

| # | Experiment | Axes | Factuals | Est. CounterContEx GPU-h | Priority |
|---|---|---|---:|---:|---|
| E1 | Main comparison | 8 datasets x 3 classifiers x 5 seeds x 6 methods, k=1 primary rank | 250 | ~70 | P0 |
| E2 | Diverse-set comparison | 8 datasets x 1 classifier x 5 seeds, CounterContEx vs DiCE, k=3 | 250 | ~25 | P0 |
| E3 | Backend ablation | TabICL vs empirical, 8 datasets x 3 seeds | 250 | ~15 | P0 |
| E4 | Confidence / tau Pareto | confidence quantiles x tau in {0.5,0.6,0.7,0.8,0.9}, 4 datasets | 250 | ~20 | P0 |
| E5 | Search + diversity ablations | proposal breadth, DPP variants, cf_mode, revisits | 100 | ~10 | P1 |
| E6 | Context ablation | size {64,128,256,512} x labels {predicted, true} | 100 | ~10 | P1 |
| E7 | Cost-quality Pareto | 4 cheap-to-expensive configurations, 4 datasets | 250 | ~12 | P1 |
| E8 | Robustness to retraining | scoring pass over stored E1 arrays | reuse | ~0 | P1 |
| E9 | Foundation-model swap | TabPFN v2 / TabICLv2 on 2 datasets | 250 | ~8 | P2 |
| E10 | Headline 1,000-factual table | best configuration only, 4 original datasets | 1,000 | ~9 | P2 |

Total CounterContEx GPU time: roughly 180 hours, plus baselines (negligible by comparison
except DiCE and FACE). This is one week on four GPUs, or a few days on the Athena
allocation the repository is already configured for.

**Two protocol warnings, both already recorded as project contracts:**

- Do not tune the confidence quantiles or tau on the same held-out results later reported
  as the unbiased comparison. Split a validation dataset set from a reporting set, or
  disclose the selection explicitly. `CLAUDE.md` already states this rule; the paper must
  visibly obey it.
- E10 must not be the run you also select configurations from. Freeze the configuration on
  E1–E7, then execute E10 once.

---

## 6. Paper skeleton

| Section | Content | Blocking dependency |
|---|---|---|
| 1. Introduction | The per-dataset generative model is the hidden cost of on-manifold CFs; foundation models remove it. Figure 1: NICE substitutes an observed neighbour's value, CounterContEx samples from a conditional posterior. | B9 |
| 2. Related work | The four neighbourhoods of §2.1. Declare CEL authorship. Position against TabPFGen explicitly. | — |
| 3. Method | Problem setting; local context construction; conditional proposals; confidence conditioning; the decoupled search; beam + DPP. Reuse `docs/countercontex-method.md` §§1–6, which is already close to publication prose. | — |
| 4. Experimental setup | Datasets, classifier families, protocol, metric denominators, run identity. The denominator table is a differentiator — most CF papers are vague here and reviewers know it. | B1, B3 |
| 5. Main results | k=1 head-to-head with variance and significance; k=3 diversity vs DiCE. | B1–B4 |
| 6. Does the foundation model matter? | E3 backend ablation, given its own section because it is the question every reviewer has. | B5 |
| 7. Boundary-hugging and confidence conditioning | The tau = 0.0 finding as a field-wide observation, then the Pareto curve only CounterContEx can traverse. | B6 |
| 8. Plausibility | Detectability AUC, k-NN distance, marginal divergence, LOF/IF with an honest caveat. | B8 |
| 9. Cost | Runtime decomposition, Lending Club profile, cost-quality Pareto. | B7 |
| 10. Ablations | Search, diversity, context, foundation-model swap. | B5, B6, B11 |
| 11. Limitations | Binary-only targets, no directional/causal constraints, heuristic search, local context can miss rare valid regions, TabICL density is model-relative. §12 of the method doc is already honest and well-written — keep that tone. | — |

**Figures and tables to produce:**

- F1: mechanism diagram — NICE substitution vs conditional proposal (hand-drawn concept).
- F2: differentiation table from §2.2.
- F3: critical-difference diagram over datasets x methods (E1).
- F4: proximity-vs-confidence Pareto, all methods as points, CounterContEx as a curve (E4).
- F5: cost-quality Pareto (E7).
- F6: distribution of `p_f(y*|x')` per method — the boundary-hugging figure (E1).
- F7: qualitative case study (B9).
- T1: main k=1 results with variance. T2: k=3 diversity. T3: backend ablation.

---

## 7. Venue judgement

**NeurIPS Datasets & Benchmarks** is the lower-risk fit *today*: the harness, the metric
denominators, the content-addressed identity, and the 188 tests are already at that
track's standard, and the method becomes the flagship entry. But the method is more
interesting than a benchmark contribution, and framing it as one undersells it.

**ICLR / NeurIPS main track** is the right ambition, and reaching it is a question of
executing B1–B6. The core idea — that a pretrained tabular foundation model removes the
per-dataset generative model that on-manifold counterfactual methods have always required —
is a main-track idea. It is currently supported by one classifier, one seed, four datasets,
no ablations, and a k-mismatched table.

The gap between what the idea deserves and what the evidence currently shows is roughly
the experimental programme in §5.
