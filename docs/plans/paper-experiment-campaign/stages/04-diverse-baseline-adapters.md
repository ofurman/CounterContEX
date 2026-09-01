# Stage 4: Diverse baseline adapters

**Goal**: Give the k=3 comparison a comparator, by letting DiCE return counterfactual sets.
**Dependencies**: Stage 1

Without this stage there is no diversity claim at all. `set_action_jaccard_mean = 0.634` on
HELOC is an unanchored number: no baseline in the existing results returns a set, so nothing
says whether 0.634 is good.

---

## Steps

1. **Audit which baselines can return sets.**
   - Where: `supports_multiple_counterfactuals` in each method module, and the guard in
     `experiments/zeroshot_cf/methods/base.py` that raises when `request.n_counterfactuals != 1`.
   - DiCE natively supports `total_CFs > 1`; `generate_dice_counterfactuals()` in
     `experiments/zeroshot_cf/methods/dice.py` currently hard-codes `total_CFs=1`.
   - Record in `decisions.md` which other baselines could support sets and which genuinely
     cannot. Do not force a single-CF method into a set interface by sampling or perturbing its
     one output — that would fabricate diversity the method does not have.

2. **Wire DiCE to the requested set size.**
   - Pass `request.n_counterfactuals` through to `total_CFs`; flip the capability declaration;
     remove the guard for this method only.
   - **Preserve the no-padding contract.** If DiCE returns fewer than `k` candidates, the
     remaining slots stay unavailable. Never repeat a row, never insert the factual, never
     substitute an invalid candidate. Unavailable slots are evidence and must reach the
     per-requested-slot denominators intact.
   - Preserve determinism: DiCE's genetic method is stochastic, so its seed must come from the
     spec seed and not from ambient global RNG state.

3. **Confirm the set metrics behave on a real comparator.** Run one one-factual cell with DiCE
   at k=3 and check that `set_coverage_at_k`, `set_action_jaccard_mean` and
   `set_pairwise_gower_mean` are populated and finite, and that a run returning two of three
   candidates records coverage accordingly rather than silently padding.

4. **Register the DiCE implementation version change.** Where: `methods/registry.py`. Changing
   what a method returns is a scientific identity change; `dice-v1` must become `dice-v2` so
   existing specs cannot resolve to the same `run_id` under new behavior.

---

## Verification

- [ ] GATE DiCE at k=3 on a deterministic fixture returns three **distinct** rows, none equal to
      the factual — read from the returned candidate array. A padding implementation that repeats
      a row turns it red, and a run returning only two candidates must leave the third slot
      unavailable rather than padded.
- [ ] GATE Two DiCE runs with the same spec seed produce identical candidate arrays — read from
      the two `arrays.npz` files. Reliance on ambient global RNG state turns it red. The
      resolved manifest must also carry a `dice` implementation version differing from the
      pre-stage value — read from `methods/registry.py` and the manifest.
- [ ] REPORT DiCE k=3 set metrics on one dataset — record in `journal.md`.

---

## Commit

`feat(methods): support counterfactual sets in the DiCE adapter`
