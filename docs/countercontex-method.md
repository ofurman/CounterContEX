# CounterContEx: Method Overview

This document describes CounterContEx for machine learning researchers. It
tracks the `countercontex-v3` implementation in this repository.

CounterContEx generates counterfactual explanations for tabular classifiers. It
combines a target-aware proposal model with a classifier-guided search. The
default proposal model is TabICL, a tabular in-context learning model.

TabICL does not define counterfactual validity. It proposes feature values that
fit a local, target-conditioned data context. The classifier under explanation
scores each complete candidate and decides whether it reaches the target.

## 1. Problem setting

Let $f$ be a trained binary classifier with class probabilities
$p_f(y\mid x)$. For a factual row $x$, the benchmark defines the target as
the class opposite to $f(x)$:

$$
\{y^*\} = \mathcal Y \setminus \{f(x)\}, \qquad |\mathcal Y|=2.
$$

A returned row $x'$ is valid for generation when both conditions hold:

$$
f(x') = y^*, \qquad p_f(y^*\mid x') \ge \tau.
$$

The search also follows an action schema. A numerical feature is one action
unit. A one-hot categorical group is one atomic action unit. Immutable features
must keep their factual values.

CounterContEx seeks valid rows that remain close to the factual row. It also
supports sets whose members use different actions or values. The implementation
uses bounded heuristic search and does not claim a global optimum.

The benchmark explains the classifier, not the observed label. This choice
includes model errors in the behavior under study. It also makes the current
target policy specific to binary classification.

## 2. Method at a glance

CounterContEx separates proposal quality from decision-boundary search:

```text
training features ──► target classifier ──► predicted context labels
       │                       │
       │                       └──────────► validity and target probability
       ▼
factual-specific Gower neighbors
       ▼
[local features, predicted label, optional confidence]
       ▼
TabICL conditional feature proposals
       ▼
legal complete candidate rows
       ▼
greedy search or bounded beam search
       ▼
one counterfactual or a diverse counterfactual set
```

This design has three main parts:

1. A proposal backend estimates target-conditioned numerical values and
   categorical probabilities.
2. A search procedure turns those proposals into complete legal rows.
3. A method-blind evaluator measures returned rows with common metrics.

The boundary between the proposal backend and search is deliberate. It permits
foundation-model ablations without changes to the action space, search, or
evaluation.

## 3. Local conditional proposal model

### 3.1 Factual-specific context

CounterContEx prepares a new TabICL context for each factual row. It selects up
to 512 Gower-nearest training rows. The context includes both predicted classes
and uses classifier predictions as labels.

Numerical features use their normalized absolute difference in Gower distance.
Each categorical variable contributes either zero or one. Thus, a categorical
variable does not gain weight because it has many one-hot columns.

Search distance averages over actionable units only. The common evaluator later
recomputes grouped Gower over the complete feature schema. Immutable units add
zero distance but remain in the evaluation denominator.

CounterContEx encodes each one-hot group as one categorical identifier before
TabICL sees it. The target classifier and final output retain the original
one-hot representation.

### 3.2 Numerical proposals

For a current search state $z$ and an actionable numerical feature $j$, the
adapter masks $z_j$. It appends the requested class $y^*$ to the row. TabICL
then estimates a conditional distribution of the form

$$
\widehat p_\theta(X_j \mid X_{-j}=z_{-j}, Y=y^*).
$$

The default point proposal is the mode of TabICL's piecewise-quantile density.
The implementation selects the midpoint of the densest interior interval. It
breaks density ties toward the median and excludes extrapolated tail intervals.

A quantile grid can replace the single mode. For levels
$\mathcal Q=\{q_1,\ldots,q_m\}$, the backend proposes

$$
\widehat F^{-1}_{j\mid -j,y^*}(q), \qquad q\in\mathcal Q.
$$

The adapter batches all eligible feature queries at each greedy step. Beam
search also batches all row and feature pairs at a search depth. This structure
reduces repeated foundation-model calls.

### 3.3 Confidence conditioning

Confidence conditioning is optional. CounterContEx first computes
$p_f(y^*\mid x_i)$ for rows in the selected local context. It derives
confidence anchors from target-class context rows at configured quantiles.

The proposal distribution then has the form

$$
\widehat p_\theta(X_j \mid X_{-j}, Y=y^*, C=c),
$$

where $c$ is a requested confidence anchor. It is not the current candidate's
measured confidence. The grid lets the search inspect feature values associated
with several target-region confidence levels.

### 3.4 Categorical proposals

For an actionable categorical group $g$, TabICL estimates

$$
\widehat p_\theta(X_g=a \mid X_{-g}, Y=y^*, C=c)
$$

over all legal categories $a$. The backend ranks categories within each
group. The target classifier then compares complete atomic swaps against all
numerical candidates.

The normal single-counterfactual path uses the highest-ranked alternative from
each group. If no proposed row improves target probability, the search exposes
all remaining legal categories once. This fallback prevents an incomplete
local ranking from reducing coverage.

### 3.5 Domain projection

CounterContEx clips numerical proposals to training-set bounds. It also snaps a
feature to its nearest observed value when the training support has at most 20
unique values. These rules reduce invalid numerical states and preserve small
discrete supports.

## 4. Single-counterfactual search

The single-counterfactual path starts from the factual row. At each step, it
creates one-action trials from all eligible numerical features and categorical
groups. It scores every complete trial in one classifier batch.

If no trial is valid, the search commits the row with the largest target
probability. The new probability must strictly exceed the current probability.
Otherwise, the search stops and preserves the best attempted row only as a
diagnostic.

If one or more trials are valid, the search selects the smallest grouped-Gower
distance from the factual. It breaks ties with local proposal support and then
target probability. The first valid row becomes the sparse counterfactual.

The default step budget equals the number of actionable units. A configured
budget can increase this limit. With revisits enabled, a later pass can propose
a new value for an action unit used in an earlier pass.

The following pseudocode summarizes sparse mode:

```text
current = factual
while validity budget remains:
    proposals = propose one legal change for each eligible action unit
    proposals = project numerical values and apply atomic categorical swaps
    probabilities, labels = classifier(proposals)

    if any proposal is valid:
        current = closest valid proposal under grouped Gower
        return current

    next = proposal with maximum target probability
    if target_probability(next) <= target_probability(current):
        return unavailable
    current = next

return unavailable
```

Sparse mode uses no joint-density call. Its sparsity comes from a short
first-crossing path, not from exact minimization of the changed-feature count.

## 5. Optional joint-density refinement

The `data_plausible` mode adds one refinement attempt after sparse validity. It
is available only for a single requested counterfactual and requires a backend
with joint scoring.

The method generates valid one-action changes and reversions around the sparse
counterfactual. A candidate can use only a configured number of extra action
units. CounterContEx keeps one representative per action unit before it fills a
bounded shortlist.

The TabICL backend scores the sparse incumbent and shortlist in one batch. It
uses complete-row log densities under the same local context and target class.
The method accepts a candidate only when its score exceeds the incumbent by the
configured minimum gain.

This score supports relative ranking within one factual. It is not a calibrated
probability of realism. It is also distinct from the benchmark's LOF and
Isolation Forest diagnostics.

## 6. Multiple counterfactuals

Requests for $k>1$ use a separate bounded beam search in sparse mode. Each
beam level expands every eligible action from every retained state.

The search adds a candidate to the valid pool only when it reaches the target
class and generation threshold. It retains an invalid state only when it
strictly improves its parent's target probability.

Beam pruning preserves states from different changed-action sets. This niche
rule prevents one high-probability action pattern from filling the complete
beam.

CounterContEx filters the final pool relative to its closest valid member. A
candidate must satisfy both bounds:

$$
d_G(x,x_i') \le r\,d_G(x,x'_{\mathrm{anchor}})+\delta,
$$

$$
s(x,x_i') \le s(x,x'_{\mathrm{anchor}})+b.
$$

Here, $d_G$ is grouped Gower distance and $s$ counts changed action units.
The values $r$, $\delta$, and $b$ are configured quality tolerances.

The method then selects an exact fixed-size determinantal point process subset
from the bounded pool. For candidate $i$, the default quality term is

$$
q_i = \exp\left(-4d_G(x,x_i')-\frac{s(x,x_i')}{M}\right),
$$

where $M$ is the number of action units. The similarity kernel uses an RBF
distance over action indicators and changed values:

$$
L_{ij}=q_i\exp\left(-\frac{\lVert z_i-z_j\rVert^2}{2\sigma^2}\right)q_j.
$$

The default embedding gives 75 percent of its weight to changed-action
indicators and 25 percent to values. Exact MAP selection maximizes the log
determinant for the requested subset size.

CounterContEx never pads a set with duplicate, invalid, or factual rows. If it
finds fewer than $k$ valid rows, the remaining slots stay unavailable.

## 7. Actionability contract

The current actionability contract has two enforced rules:

- Immutable features cannot change.
- A one-hot categorical variable changes as one atomic group.

The method does not yet encode directional, monotonic, causal, feasibility, or
user-cost constraints. Its outputs are counterfactual explanations under the
declared action schema. They are not automatically valid real-world recourse
actions.

## 8. Proposal-backend ablation

The deterministic `empirical` backend provides a checkpoint-free control. It
uses reference rows that the classifier assigns to the target class.

For numerical features, it proposes target-class medians or configured
quantiles. For categorical groups, it uses unit-smoothed target-class
frequencies. It supports the same sparse search but does not support confidence
conditioning or joint scoring.

This backend isolates the value of TabICL proposals from the value of the search
procedure. Unsupported backend and search combinations fail during method
preparation.

## 9. Reference experimental protocol

The tracked full reference matrix evaluates four datasets:

- HELOC
- Bank Marketing
- Give Me Some Credit
- Lending Club

The dataset pipeline uses a deterministic 64/16/20 train, validation, and test
split with seed 42. It fits preprocessing on training data. Numerical features
use MinMax scaling, and categorical variables use grouped one-hot encodings.

The target classifier is logistic regression with `C=1.0`, `max_iter=1000`, and
seed 42. The protocol selects up to 1,000 factuals through deterministic
stratified sampling.

The CounterContEx cell requests three counterfactuals. It uses nine numerical
quantiles from 0.1 through 0.9 and five confidence quantiles. Its beam width is
8, and its candidate pool size is 16.

The cell uses one TabICL estimator, temperature `1e-9`, and at most 100 validity
steps. It permits two extra actions inside the diverse-pool quality bound.

The generation threshold is $\tau=0.5$. The evaluator separately reports
threshold validity at 0.7. Generation and evaluation thresholds must remain
distinct when interpreting results.

## 10. Evaluation and denominators

Availability and validity answer different questions. Coverage measures
factuals with at least one returned candidate. Returned validity uses only
available candidates as its denominator.

Per-requested-slot success includes unavailable slots in its denominator.
Per-factual success measures factuals with at least one successful candidate.
Primary metrics use one configured rank, while set metrics use all returned
candidates.

The common evaluator uses two candidate populations:

- Grouped-Gower and continuous proximity use returned target-class candidates.
- Sparsity, action-unit changes, immutable preservation, bounds, LOF, and
  Isolation Forest use all available returned candidates.

LOF stores `-score_samples`, so larger values mean more outlying candidates.
Isolation Forest stores `decision_function`, so larger values mean more inlying
candidates. Neither value is a probability of realism.

Diversity metrics include action-set Jaccard distance and pairwise grouped
Gower distance. Set coverage at $k$ requires all requested ranks to be
available.

## 11. Testable research hypotheses

The implementation supports controlled tests of these hypotheses:

1. Target-conditioned TabICL proposals improve coverage or proximity over
   target-class empirical quantiles.
2. Confidence anchors improve high-threshold validity without unacceptable
   losses in coverage, proximity, or runtime.
3. Beam search and DPP selection improve set diversity while preserving
   validity and bounded factual distance.
4. Joint-density refinement improves within-factual TabICL density without
   unacceptable increases in action count or grouped Gower distance.

These statements are hypotheses, not conclusions. Tests must hold the dataset,
factuals, classifier, target policy, action schema, seed, and requested set size
constant.

## 12. Limitations

CounterContEx has several current limitations:

- The benchmark target policy supports binary classifiers only.
- Search is heuristic and can stop at a local target-probability plateau.
- TabICL density reflects the selected context and model, not a causal data
  distribution.
- The one-shot plausibility mode does not support multiple counterfactuals.
- The method has no learned cost model for user-specific action difficulty.
- Runtime grows with factual count, proposal-grid size, search depth, and beam
  width.
- A local context can omit rare but valid regions of the target class.

These limits define useful future work. They also constrain claims about
optimality, realism, and recourse.

## 13. Implementation map

The main implementation files are:

- [`methods/countercontex/config.py`](../experiments/zeroshot_cf/methods/countercontex/config.py)
  defines search, diversity, and foundation settings.
- [`methods/countercontex/method.py`](../experiments/zeroshot_cf/methods/countercontex/method.py)
  implements the benchmark-facing method lifecycle.
- [`methods/countercontex/search.py`](../experiments/zeroshot_cf/methods/countercontex/search.py)
  adapts proposal sessions to the search core.
- [`methods/countercontex/backends/tabicl.py`](../experiments/zeroshot_cf/methods/countercontex/backends/tabicl.py)
  owns TabICL representation, local context, and proposal state.
- [`grouped_categorical.py`](../experiments/zeroshot_cf/grouped_categorical.py)
  implements mixed-data greedy search and joint refinement.
- [`diverse_search.py`](../experiments/zeroshot_cf/diverse_search.py) implements
  bounded beam search and DPP selection.
- [`evaluation/`](../experiments/zeroshot_cf/evaluation/) contains the
  method-blind metrics.
- [`full_reference.yaml`](../experiments/zeroshot_cf/configs/matrices/full_reference.yaml)
  records the reference scientific configuration.
- [`countercontex_ablation_example.yaml`](../experiments/zeroshot_cf/configs/matrices/countercontex_ablation_example.yaml)
  records independent search, backend, diversity, dataset, and seed axes.
