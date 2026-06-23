# Stage 6: MOONS Per-Step Trajectory Visualization

**Goal**: Produce the meeting's headline plots — on the 2-D MOONS dataset, show **where each test point lands after step 1, 2, 3, …**, which feature was selected at each step, and the "blocked regions" where a single-feature move cannot find probability mass on the cut slice. This is the visual evidence for *why* revisiting (Stage 5) helps.
**Dependencies**: Stage 5 DONE (so trajectories reflect the revisit-enabled loop). Uses the per-step `info["history"]` already recorded by the greedy loop (`greedy.py:145–151`).

---

## Motivation (from the meeting)

> "przygotuję tabelkę i wykresy na moonsach, żebyśmy zobaczyli gdzie te próbki rzeczywiście lądują po pierwszym, drugim, trzecim stepie"
> "na moonsach możemy sobie to wyplotować, jak się zmieniają te cechy … są regiony, gdzie jak zmienimy jedną cechę, to nie ma masy probabilistycznej w tej uciętej pionowej albo poziomej dla tej drugiej … tam on to przesuwa w jakieś inne miejsce, które blokuje przesunięcie tej drugiej cechy."

There is **no existing plotting module** — write one from scratch (the matplotlib setup pattern in `exp1_single_feature.py:225–265` is a reusable starting point).

---

## What to plot

A new script `experiments/zeroshot_cf/exp8_moons_trajectories.py` (MOONS only) that, for a small set of test points (e.g. 20–40, emphasis on near-boundary ones):

1. **Background**: MOONS train scatter colored by class + the **logistic-regression discriminator's decision boundary** (the external oracle, from `discriminator.py`) drawn as a contour at `p=0.5`. Since each greedy step changes exactly one of the 2 features, every move is **axis-aligned** (horizontal = feature 0, vertical = feature 1) — this is what makes the "blocked slice" visible.
2. **Trajectories**: reconstruct the intermediate states by replaying `info["history"]` from the factual point — start at `x_factual`, apply each `(feature_idx, committed_value)` in order. Draw the factual (circle), each intermediate (small dot), and the final CF (triangle), connected by arrows. Color the arrow by which feature moved.
3. **Step annotation**: number each segment (1, 2, 3, …) so the per-step landing is legible; with revisits, the same feature axis may move more than once.
4. **Blocked-region overlay** (the key insight): for a representative stalled/failed point, plot the TabPFN **conditional density** of each candidate feature on its cut slice (1-D bar distribution along the horizontal/vertical line through the current point, via `sampler.predictive_distribution`), to show that the mode sits on the wrong side of the boundary — the geometric reason a single-feature move can't flip it. One or two illustrative panels suffice.
5. **Status coloring**: distinguish flipped (validity=1) vs stalled/failed trajectories (e.g. green vs red), so the plot doubles as a validity diagnostic.

Save to `results/figures/moons_trajectories.png` (+ a couple of zoom-in panels, e.g. `moons_blocked_slice.png`). Create `results/figures/` if absent.

---

## Steps

1. **Add a trajectory hook.** The greedy loop already returns `info["history"]` with `(feature_idx, committed_value, p_target_after, score)` per step (Decision #8) — confirm this is enough to replay states from the factual point. If the committed value alone is insufficient to reconstruct (it is sufficient: each step sets `x_cf[j]=val`), no loop change is needed; otherwise add the post-step `x_cf` snapshot to the history tuple. **Prefer no loop change** — replay from the factual row.
2. **Write `exp8_moons_trajectories.py`.** Load MOONS via `data.py:load_dataset("moons")`, fit the discriminator, run the greedy loop (Stage-4 MOONS config: `prob_ascent`, `random_both@512`, the Stage-5 revisit loop) over a bounded set of test points, collect histories, and render the figure(s) described above. Reuse the matplotlib pattern from `exp1_single_feature.py:225–265`.
3. **Pick illustrative points.** Sample/order test points by distance to the decision boundary so the figure shows both easy single-step flips and hard multi-step / stalled cases (the blocked regions). Document the selection in the script.
4. **Write a short `results/figures/README.md`** describing each figure and the takeaway (one paragraph: "revisiting unblocks points X; the blocked-slice panel shows why a single move stalls").

---

## Verification

- [ ] `results/figures/moons_trajectories.png` exists, shows train scatter + `p=0.5` boundary + factual→CF axis-aligned trajectories with per-step numbering and flip/stall coloring.
- [ ] At least one blocked-slice panel shows the candidate-feature conditional density relative to the boundary for a stalled point.
- [ ] The script runs offline (`get_models()` only; no `tabpfn_client`), MOONS only, bounded test count, and regenerates the figures deterministically.
- [ ] `results/figures/README.md` explains the figures.
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Notes

- This is a **visualization** stage — no new metrics, no model behaviour to unit-test. A light smoke test (the script produces a non-empty PNG on a tiny `--max-test`) is enough; do not over-test plotting.
- Keep figures publication-legible (labeled axes in MinMax-[0,1] space, legend for feature/step/status). These plots are a deliverable for the next meeting and a likely paper figure.

## Commit

`feat(greedy-cf): MOONS per-step trajectory + blocked-slice visualization (Exp8)`
