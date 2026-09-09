# Frozen Experiment Protocol: E11-E13

This resource fixes the scientific questions and comparison rules before implementation or
checkpoint-backed execution. Implementation details may change to satisfy these contracts, but
outcome-dependent changes require a recorded decision and a new scientific identity.

## Prior evidence and limits

- E3: six datasets, LR, seeds 17/42/101, up to 250 factuals, `k=3`, quantiles `.1-.9`, context
  512, one estimator, sparse beam/DPP search, generation tau `.5`, evaluation threshold `.7`.
- TabICL minus empirical: coverage `-.0033`, returned threshold validity `+.0057`, grouped-Gower
  `+.0079`, neighbour support `-.0004`, total runtime `8.09x`.
- The threshold effect was Lending-Club-led; proximity worsened on five of six datasets.
- The authenticated diagnostic found 2,519/2,520 different raw numerical quantiles. In matching
  traced credit cases, no numerical proposal was valid and common categorical edits filled the
  same pool. E3 seed outputs were identical and are not independent repetitions.
- E11-E13 evaluate the four retained datasets: HELOC, Bank Marketing, Give Me Some Credit, and
  Lending Club. HELOC has no actionable categorical group and is a numerical-only control.

## Shared definitions

For factual/state `x`, target `y_t`, and action unit `j`:

```text
q_j(z) = q_TabICL(z | x_-j, y_t)
x_j(z) = project(x with action unit j replaced by z)
s_j(z) = p_clf(y_t | x_j(z))
delta_j(z) = s_j(z) - p_clf(y_t | x)
pi_beta,j(z) proportional to q_j(z) * s_j(z)^beta
```

`pi_beta` is a classifier-guided energy tilt, not a Bayesian posterior or a coherent joint over
all features. TabICL q is already conditioned on the classifier-derived target. No Gower,
proximity, user cost, causal rule, or plausibility proxy enters q decoding or pi-beta weighting.
The unchanged search may use Gower only after it has a valid candidate; proposal-only reports
isolate the sampler from that downstream rule.

An action unit is one numerical feature or one atomic one-hot categorical group. Dummy columns
are never independent action units. Immutable features remain fixed.

## Decoder semantics

### Numerical q

- `grid-9`: ICDF at `.1,.2,...,.9`.
- `grid-49`: ICDF at `r/50` for `r=1,...,49`.
- `iid-B`: `B` keyed uniforms from `U(0,1)`, mapped through ICDF.
- Partition `u in (0,1)` into 256 equal-mass bins `I_r=[r/256,(r+1)/256]`. Score bin r by
  `log_prob(Q((r+.5)/256))`; ties prefer smaller r. The first and last midpoint-represented bins
  are the declared tail approximation. Sampling within a retained bin uses
  `u=(r + clip(v, 2^-53, 1-2^-53))/256` for a keyed `v~U(0,1)`.
- `greedy`: the existing TabICL `quantile_mode`, based on its native quantile representation. It
  is a one-proposal reference, is not derived from the 256-bin truncation, and is not described as
  budget-matched to B=9 or B=49.
- `top-k-B`: retain the five highest-scoring bins, preserve their equal original q masses,
  renormalize, and draw `B` keyed samples.
- `top-p-B`: retain the first 231 density-ordered bins (the smallest union with mass at least
  `.9`), preserve their equal original masses, renormalize, and draw `B` keyed samples.

Serialize bin scores, ordering, membership, midpoint values, within-bin uniforms, and selected
values. This fixed approximation defines the experimental policy; do not describe its results as
a general property of continuous top-k/top-p sampling or exact highest-density regions.

### Categorical q

Use the exact `CategoryProposals` support and probabilities. `greedy` is argmax; `top-k` retains
the five most probable legal categories; `top-p` is the smallest probability-ordered prefix with
at least `.9` mass. Ties are resolved by canonical legal-support order. The current category is
not removed before decoding: a no-op consumes proposal mass and budget and is reported.

### Projection and duplicates

Project before classifier scoring. Clipping, support snapping, no-ops, and duplicates consume raw
proposal budget. Do not refill to obtain B unique candidates. Aggregate duplicate probability mass
for pi-beta normalization, score each unique legal row once, and retain a reverse map from every
raw draw to that score and disposition.

## Budget and RNG contract

Track four budgets independently:

1. raw q proposals per state/action unit (`B`),
2. q pool size for guidance (`M`),
3. unique projected rows evaluated by the classifier,
4. TabICL calls/rows and elapsed phases.

The strict experimental search caps each action unit at B raw proposals and never invokes the
historical categorical fallback. A category group with fewer legal values simply yields fewer
unique trials. The full support size and effective budget are reported.

Key every uniform from canonical bytes containing run seed, factual source index, state/depth,
parent ID, action-unit ID, and draw/resample index. Do not use Python `hash()`. Results must be
invariant to pair ordering and batch chunking. Arms share q-pool uniforms and, where applicable,
resampling uniforms (common random numbers). Sampling seeds are Monte Carlo repetitions of the
same factuals, not independent factual observations.

For E11 and E13, pair outcomes by `dataset x factual_partition x factual_source_index x target`. Average Monte Carlo
seeds within factual before inference; never treat candidates, draws, quantiles, or seeds as
independent cases. Report dataset-specific effects and a hierarchical bootstrap over datasets then
factuals as REPORTs. With only four datasets, do not interpret failure to reject as equivalence.

## E11: q representation and decoding

**Hypothesis**: grid-9 misses useful q mass; IID or truncated sampling can increase threshold-valid
success per requested slot at a fixed raw/oracle cap. Projection or search may erase the effect.

**Shared settings**: four retained datasets, retained LR target model, `k=1`, context 512,
predicted context labels, one estimator, no confidence conditioning, no joint score, sparse search,
generation tau `.5`, evaluation threshold `.7`. Grid arms use ICDF directly; sampling arms use
full-distribution temperature `1.0`.

**Numerical-decoder block**: keep the categorical policy fixed to IID-9 draws from its exact q
with a strict categorical cap of 9 in every arm, including numerical B=49. This makes the 9-to-49
budget curve a numerical-budget change only. Compare:

- one-proposal mode reference;
- B=9: grid-9, IID-9, top-k-9, top-p-9;
- B=49: grid-49, IID-49, top-k-49, top-p-49.

**Categorical-decoder block**: keep numerical proposals fixed to IID-9 and compare categorical
IID-9, top-k-9, and top-p-9 under B=9, plus a one-proposal categorical greedy reference. Do not
change both numerical and categorical decoders in one primary contrast.

Primary predeclared contrasts are grid-9 versus IID-9 and grid-49 versus IID-49. The B=9 versus
B=49 difference is a budget curve, not a decoder-only effect. Truncation and mode effects are
secondary REPORTs.

Primary final metric: `valid_success_rate_threshold_per_requested_slot` at `.7`. First inspect
coverage and class-valid/threshold-valid counts. Guardrails: grouped-Gower, action units,
actionability, neighbour support, out-of-bounds, raw/effective budgets, classifier rows, TabICL
calls/rows, and lifecycle runtime.

## E12: feature-distribution pushforward

**Hypothesis**: on mixed-action datasets, an atomic categorical action unit changes target
probability more than a numerical action unit on average. This is a hypothesis about the fixed
classifier's response under q, not about causal effect or user cost.

Use fixed factual states before adaptive search. Record later-depth common states only as a
separate diagnostic; never compare backend-specific trajectories as though they were common
inputs. For categories, enumerate and weight the legal support exactly. For numerical features,
use a fixed 256-point stratified ICDF grid; on the pilot only, add five independent 64-draw banks
to estimate Monte Carlo sensitivity.

Run LR, MLP, and XGBoost separately. The main factual-level estimand is:

```text
D_i = mean over categorical action units of E_q[abs(delta) | projected value changes]
    - mean over numerical action units of E_q[abs(delta) | projected value changes]
```

Each action unit has weight one, regardless of category cardinality. HELOC is excluded from the
paired categorical-minus-numerical aggregate but retained as a numerical reference. Also report:
unconditional `E_q abs(delta)` with no-op mass, positive delta, probability of positive delta,
class/tau-.5/threshold-.7 crossing mass, median/q90/best-of-B effects, target-log-odds effects,
projection displacement, no-op and duplicate mass, dataset, classifier, target direction, and
feature/action-unit summaries. No distance or cost normalization is allowed.

If an action unit has zero q mass on legal projected values different from the factual, its
conditional effect is `NOT ESTIMABLE`, not zero. Report its count and no-op mass. A factual-level
`D_i` is `NOT ESTIMABLE` when either action-type mean has no estimable unit; exclude it from the
paired conditional-effect aggregate under this fixed rule while retaining it in unconditional
effects and availability denominators.

Statistical unit: `dataset x classifier x factual_partition x factual_source_index x target`. Average action-unit and
sampling-seed observations within factual before inference. Never treat proposals, quantiles, or
seeds as independent cases. Report dataset-specific estimates and a hierarchical bootstrap over
datasets then factuals; confidence intervals and p-values are REPORTs. Failure to support the
hypothesis is a valid result.

## E13: classifier-guided pi-beta sampling

**Hypothesis**: beta=1 increases threshold-valid success per requested slot over beta=0 under the
same `M=64` and `B=9` caps at every expansion. The fixed-state companion compares beta arms on
exactly the same q pool and classifier rows; adaptive end-to-end runs report, but do not assume
equality of, realized cumulative classifier rows and calls.

For each common state/action unit, draw `M=64` raw proposals with replacement from q for both
numerical and categorical units. Project and aggregate duplicate mass, then batch-score unique
rows. The categorical sampler uses exact q probabilities to draw its common pool but does not
enumerate the full support in E13. For beta in `{0,.5,1,2}`, compute in log space:

```text
log w_m = beta * log(max(s_m, epsilon))
```

Do not multiply by an estimated q density: the empirical pool already represents q. Select `B=9`
pool indices by standard weighted resampling with replacement using common uniforms. Repeated
indices consume budget and are not refilled. Cache classifier scores so search does not query those
rows twice. Beta=0 is the pure-q control; beta=1 versus beta=0
is the primary contrast. Beta `.5` and `2` are dose-response REPORTs. Confidence conditioning and
joint scoring remain off.

Common pool IDs and resampling uniforms are required only when the canonical `(factual, state,
depth, parent, action unit)` key is identical. Once beta arms select different states, their later
conditional q pools are necessarily distinct and remain reproducible through their own keyed
uniforms. A fixed-state companion at the original factual isolates reweighting under exactly
matched q-pool and oracle work; the adaptive run measures complete-policy utility under matched
per-expansion caps.

Add a secondary `classifier-top-B` operational comparator: from the same already scored M pool,
pass the B highest-scoring unique projected rows onward, with canonical pool-order tie-breaking.
It is a REPORT, not a sample from pi-beta and not part of the beta dose response. It reveals whether
stochastic reweighting is dominated by direct use of classifier evaluations already paid for.

Run E13 under the E11 shared settings. Report the E11 final metrics plus estimated normalizer,
weight ESS, selected unique fraction, raw and unique oracle rows, and calls/rows until first
validity and in total. The same classifier guides proposals and defines validity; this establishes
algorithmic utility against that oracle, not independent distribution quality or calibration.

## Pilot and confirmation

The authorized pilot uses 20 deterministically target-stratified **validation** factuals per
retained dataset. It is an engineering
and mechanism check, not performance evidence and must not be used to change the predeclared E11
or E13 contrasts. E12 pilot runs all three classifiers; E11/E13 pilot LR. Seeds are 17, 42, and
101 where stochasticity applies. The pilot records a projected GPU-hour estimate for confirmation.

After separate authorization, confirmation uses 100 deterministically target-stratified **test**
factuals per dataset. E11 and
E13 use seeds 17, 42, 101, 202, and 303. E12 uses the deterministic 256-point integration for all
three classifiers and no seed multiplication. A limited k=3 replication uses grid-9, IID-9,
beta=0, and beta=1, 50 factuals, seeds 17/42/101, and one explicit common categorical budget. It
is reported separately from the k=1 primary experiments.

The protocol's factual-partition field is part of case/scientific identity and defaults to `test`
for every historical matrix. Add a new `target_stratified` policy: score the complete factual
partition with the fixed classifier, derive opposite-class targets, then deterministically sample
within target strata with selection seed 42. Preserve the existing `stratified` policy unchanged;
it is truth-label-stratified and remains the historical default. Selection is target-model-specific
and E12 classifier families are analyzed separately rather than treated as paired factual samples.

Factual identity is `(partition, local_source_index)`, not the bare local integer. Validation and
test therefore remain disjoint by partition even when both contain the same local index value.
Classifier fitting remains unchanged, and validation factuals are not added to the training
reference.

## Trace and result provenance

Every trace metadata record includes dataset/case/model identity, backend/method versions,
checkpoint hashes, context fingerprint, factual partition and local source index, target, action schema, policy,
B/M/k/p/beta/temperature, RNG key scheme, and generation/evaluation thresholds.

Every proposal record includes state/parent/depth, action unit/type/cardinality, raw value,
uniform/quantile or category, q mass/log-density where defined, truncation rank/membership,
projected value/displacement, no-op/duplicate linkage, target probabilities before/after, delta,
class/tau/threshold flags, beta weights/ESS, selection status, classifier batch, and transitions
through proposed/projected/unique/scored/improving/valid/beam/pool/selected stages.

All trace directories are new and immutable after a `COMPLETE.json` hash inventory. Standard
end-to-end arms retain canonical run directories unchanged and use strict aggregation followed by
an external `campaign-inventory-v1` file that hashes every required payload and resolved run ID.
The inventory is published atomically outside run directories only after exact membership passes;
all later analysis verifies it before reading values. Missing/failed slots remain missing; no
metric, row, candidate, or status may be fabricated.

## Interpretation constraints

- E11 tests the declared TabICL interface/decoder, not the total quality of every representation
  of TabICL's predictive distribution.
- E12 measures classifier sensitivity under q, not action cost, causality, calibration, or real-
  world effect.
- E13 uses the same oracle for guidance and success; it does not prove independent robustness.
- Larger B/M consumes more compute; report quality against all budget axes rather than presenting
  only the best point.
- Projection can turn numerical distributions into discrete/no-op-heavy proposal measures.
- A new policy is a scientific identity change. Never aggregate it into historical E3/E5 roots.
- No checkpoint-backed smoke, pilot, or confirmation runs without explicit user authorization.
