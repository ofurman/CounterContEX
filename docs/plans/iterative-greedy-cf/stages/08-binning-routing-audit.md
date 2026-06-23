# Stage 8: Continuous-Feature Binning & Classifier-Routing Audit

**Goal**: Resolve the meeting's open questions about how TabPFN handles **continuous** features — bin/softmax structure, ordering preservation, and how a scalar is recovered on commit — and **fix the proximity/validity leak** caused by low-cardinality integer columns being auto-routed to the *classifier* head (which discards their ordered support). This is the documented root cause of `class_divergence`'s HELOC degradation (Fixed Issue #1) and a suspected proximity drag on `prob_ascent`.
**Dependencies**: Stage 1 DONE (sampler + selectors). Independent of Stages 5/6/7.

---

## Motivation (from the meeting)

> "jeśli binujemy, to tracimy chyba informację o orderingu tych cech i to wtedy możemy mieć problem z proximity … upewnię się jak tam z tymi cechami ciągłymi to się dzieje."
> "Validity może być w ogóle to binowanie powinno być robione uwzględniając klasę, ale nie wiem … musiałbym sprawdzić."

## Findings already established (verify, then document)

From code exploration (cite these in the writeup with line refs):
- TabPFN v2 models a continuous feature with a **`FullSupportBarDistribution`** — a softmax over **ordered** bins (the v2 backbone uses 5000 buckets; borders are sorted, `searchsorted`-based). **Ordering IS preserved** within the continuous path: `src/tabpfn/architectures/base/bar_distribution.py` (bins/borders, `icdf` at ~`:261–288`, `mean` at ~`:594–603`).
- On commit, the greedy loop recovers a scalar via `sampler.sample_feature(... temperature=1e-9 ...)` → `impute_masked` → `model.impute(t=…)` → **`icdf` interpolation within the selected bin** (not a bare bin midpoint). So sub-bin precision is retained — `greedy.py:190–198`, `sampler.py:392–458`.
- **The real leak**: a low-cardinality **integer** column is auto-routed to TabPFN's **classifier** head by `infer_categorical_features` (`src/tabpfn/preprocessing/type_detection.py`), checked at `sampler.py:535` (`use_classifier_`). The classifier head's `classes_` are int-cast, so the **ordered MinMax-[0,1] support is lost** — exactly why `class_conditional_shift` had to fall back to total-variation distance (Fixed Issue #1 / Decision #11). For a column that is *semantically continuous* but happens to be low-cardinality integer, this is a modeling mismatch that can hurt both proximity (no ordering) and validity.

So the audit is mostly **verification + an override experiment**, not discovery.

---

## Steps

1. **Write the binning audit note** (`results/binning_audit.md`): with line-referenced evidence, document (a) bar-distribution + ordered bins, (b) `icdf` scalar recovery on commit (ordering preserved, not midpoint), (c) the classifier-routing of low-cardinality integer columns and the resulting support loss, (d) whether the bin layout is **class-aware** — it is not (borders are class-independent), which answers the meeting's "czy binowanie uwzględnia klasę" with "no; only the per-class *softmax weights* differ, the bins don't." Keep it tight and evidence-led.

2. **Identify HELOC's misrouted columns.** Programmatically list which HELOC columns `use_classifier_` routes to the classifier head vs the regressor head (log counts + names). This quantifies the blast radius.

3. **Add a routing override knob.** Expose a way to **force specified columns to the regressor (bar-distribution) head** despite low cardinality — most cleanly via the existing `categorical_features` / `categorical_features_indices` plumbing (pass an explicit *empty* or curated categorical set so the integer columns stay numerical). Add it as an Exp4/Exp9 flag (e.g. `--force-numeric-cols all|none|<idx,list>`), default `none` (current behaviour, no regression).

4. **Run the override experiment `exp9_routing_audit.py`** (HELOC; MOONS is all-continuous so unaffected — use it as a null control): compare `prob_ascent` at the Stage-4 config with vs without forcing the int columns numeric. Report Δ validity, Δ `proximity_l2_jaccard`, Δ `frac_oob`, Δ `l0_count`. Write `results/exp9_routing_summary.md` with the verdict: does forcing ordered/numeric treatment improve proximity and/or validity, or does it hurt (e.g. extrapolation off a too-coarse support)?

5. **Tests.** Add `tests/test_routing.py` (shared `models` fixture): assert the override flag flips a known low-cardinality HELOC column from classifier to regressor routing (inspect `predictive_distribution` return shape — `{"proba","classes"}` vs `{"logits","criterion"}`, `sampler.py:464–551`), and that `--force-numeric-cols none` is byte-identical to current behaviour.

---

## Verification

- [ ] `results/binning_audit.md` answers all four meeting questions (bin/softmax structure, ordering, scalar recovery, class-awareness) with file:line evidence.
- [ ] The misrouted-HELOC-columns list is logged (names + counts).
- [ ] `--force-numeric-cols` flag works: with `none` the run is unchanged; with the int-column list, those columns route to the regressor head (verified via `predictive_distribution` shape).
- [ ] `results/exp9_routing_summary.md` reports Δ validity / Δ proximity / Δ frac_oob with vs without the override on HELOC, with a clear verdict.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` passes (incl. `test_routing.py`).
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` empty (the override goes through the sampler/runner, NOT a `src/tabpfn` edit); no `tabpfn_client` import.

---

## Expected outcomes

- A clean, citable answer to "how does TabPFN treat continuous features, and does binning hurt proximity?" — ordering is preserved on the continuous path; the real risk is **classifier-routing of low-cardinality integers**, which *does* discard ordering.
- An empirical read on whether forcing numeric treatment of those columns improves HELOC proximity/validity. Either direction is a useful, reportable result; if it helps, it becomes a recommended preprocessing step; if it hurts (support too coarse), it justifies the current auto-routing.

## Commit

`feat(greedy-cf): binning audit + classifier-routing override experiment (Exp9)`
