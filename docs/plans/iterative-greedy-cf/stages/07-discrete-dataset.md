# Stage 7: Add a Discrete/Categorical Dataset + Validity Sanity Check

**Goal**: Wire a genuinely **discrete (categorical)** dataset into the pipeline and confirm the meeting's prediction that greedy + TabPFN should achieve **near-100% validity** there. All current datasets (MOONS, HELOC) are continuous; this stage closes that gap and stress-tests the classifier-head path that HELOC's int-collapsed columns only partially exercise.
**Dependencies**: Stage 1 DONE (Exp4 runner + metrics). Independent of Stages 5/6/8. Benefits from Stage 5's revisit loop but does not require it.

---

## Motivation (from the meeting)

> "warto pewnie dorzucić jakieś dyskretne dataset … na takim [dyskretnym] to powinno to podejście działać, także przynajmniej validity będzie 100%."

TabPFN models a categorical feature with a softmax over its categories (the classifier head), so a class-conditional commit lands on an actual in-support category — there is no extrapolation off a continuous bar distribution, so validity *should* be easy. Demonstrating this isolates the continuous-binning effects (Stage 8) from the method's core soundness.

---

## Design

The dataset infrastructure already supports categorical features via the CEL config system (`data.py:load_dataset` at `:43–69`; mixed example `vendor/counterfactuals/config/datasets/adult_census.yaml:16–30` with `one_hot_encode` in `initial_transforms`). No code change is needed to *load* a categorical dataset — only config wiring + an actionability split.

**Dataset choice** (resolve during execution, record under Decisions): prefer an **already-available, predominantly-categorical** CEL dataset over inventing one. Candidates already configured: `adult_census` (8 categorical / 4 continuous), `german_credit`, `bank_marketing`. Pick the one with the **highest categorical fraction and a clean immutable/actionable split** (e.g. `german_credit` or `adult_census` with age/sex/race immutable). If none is cleanly all-discrete, construct a small synthetic all-categorical dataset (CSV + YAML) — the meeting explicitly wanted at least one dataset where validity *must* be 100%.

---

## Steps

1. **Select & wire the dataset.**
   - If using an existing CEL config: confirm its YAML at `vendor/counterfactuals/config/datasets/<name>.yaml` declares `categorical_features` and `one_hot_encode`; confirm `load_dataset("<name>")` returns a `DatasetBundle` with `categorical_features_indices` populated (`data.py:25–41`).
   - Add an actionability config `configs/<name>_actionability.yaml` (HELOC pattern, `configs/heloc_actionability.yaml:1–41`) so `get_actionable_immutable` (`data.py:72–108`) routes correctly: immutables = protected/non-actionable attributes.
   - If synthesizing: create `vendor/counterfactuals/data/<name>.csv` + the YAML + actionability config.

2. **Confirm classifier-head routing.** For a one-hot/low-cardinality categorical column, verify the sampler routes it through the **classifier** head (`sampler.py:535` `use_classifier_`; `type_detection.infer_categorical_features`). The committed value should be a valid category, not a continuous interpolation. Note one-hot columns are 0/1 — ensure the commit + metrics handle the encoding consistently (the CF stays a valid one-hot vector, or document the relaxation).

3. **Run Exp4** with `prob_ascent` on the new dataset (and, if Stage 5 is done, with a budget allowing revisits). Bound `--max-test` as needed.

4. **Record results** in `results/exp4_greedy_<name>_metrics.csv` (the runner already writes per-dataset CSV) and a short note in `results/REPORT.md`: did validity reach ≈1.0? What are `l0_count`, `proximity`, `frac_oob` (the latter should be ~0 — categories are in-support by construction)?

5. **Tests.** Add a minimal `tests/test_discrete_dataset.py` (shared `models` fixture): assert `load_dataset("<name>")` returns categorical indices, and a smoke test that Exp4 produces a CSV row with `validity` defined on a tiny `--max-test`. Do not gate on an exact validity number in the unit test (that lives in the report).

---

## Verification

- [ ] `load_dataset("<name>")` loads the discrete dataset with non-empty `categorical_features_indices`; immutables correctly split via the actionability config.
- [ ] Exp4 on the dataset writes `results/exp4_greedy_<name>_metrics.csv` with validity, `l0_count`, proximity, `frac_oob`, `true_actionability`.
- [ ] `results/REPORT.md` reports the validity result and states whether the ≈100% prediction held; `frac_oob ≈ 0` (categories in-support).
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` passes (incl. the new smoke test).
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` empty; no `tabpfn_client` import.

---

## Expected outcomes

- Validity **≈1.0** on a genuinely categorical dataset (the meeting's prediction); `frac_oob ≈ 0`. If validity is materially below 1.0, that is itself an interesting finding about the classifier-head commit and feeds Stage 8.
- A third dataset in the headline table (Stage 9) covering the discrete regime, complementing MOONS (continuous, simple) and HELOC (continuous, hard).

## Commit

`feat(greedy-cf): wire discrete/categorical dataset + validity sanity check`
