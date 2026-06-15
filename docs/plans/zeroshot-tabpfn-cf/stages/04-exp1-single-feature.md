# Stage 4: Experiment 1 — Single-Feature Estimation (Sanity Check)

**Goal**: Verify TabPFN can sensibly reconstruct a single masked feature of a factual point by sampling from the target-class context distribution — the go/no-go gate for Experiment 2.
**Dependencies**: Stage 3 (sampler), Stage 2 (data)

---

## Steps

1. **Runner.**
   - File: `experiments/zeroshot_cf/exp1_single_feature.py`
   - For each dataset (MOONS, HELOC) and each feature index `j`:
     - Take test points; mask feature `j`; reconstruct via `ConditionalDensitySampler.sample_feature` using context = train rows (optionally class-conditioned on the point's true class — start with the **same-class** context, since the brief frames context as "the provided target-class context distribution").
     - Use near-MAP temperature (`t=1e-9`) for the point estimate, and also draw N=50 samples at `t=1.0` to inspect the conditional spread.

2. **Baselines & metrics.**
   - Baseline 1 (marginal mean): predict `j` as the train mean of feature `j` (ignores conditioning).
   - Baseline 2 (optional): a quick `sklearn` regressor (e.g. `RidgeCV`) trained to predict `j` from the other features — a cheap conditional reference.
   - Metric: per-feature reconstruction **MSE** (and MAE) in scaled space, factual vs. reconstructed. Also report calibration: does the true value fall within the sampled distribution's central interval (e.g. fraction inside the 10–90% sampled quantiles)?

3. **Aggregate & decide the gate.**
   - Table: per-feature TabPFN MSE vs. marginal-mean MSE vs. Ridge MSE.
   - MOONS: expect TabPFN MSE clearly below marginal baseline on both features.
   - HELOC: expect TabPFN to beat the marginal baseline on a majority of the 23 features.
   - Write a one-line **gate verdict** into the results: PASS (proceed to Stage 5 with confidence), WEAK (proceed but flag low expectations), or FAIL (proceed but record that Experiment 2 is unlikely to work out-of-the-box and that refinement focus shifts to context/temperature).

4. **Outputs.**
   - `experiments/zeroshot_cf/results/exp1_<dataset>.csv` (per-feature metrics), a summary table in `results/exp1_summary.md`, and for MOONS a scatter plot of true vs. reconstructed (`results/exp1_moons.png`).

---

## Verification

- [ ] `uv run python experiments/zeroshot_cf/exp1_single_feature.py --dataset moons` completes and writes `results/exp1_moons.csv` + plot.
- [ ] `--dataset heloc` completes and writes `results/exp1_heloc.csv`.
- [ ] Summary table shows TabPFN MSE vs. baselines per feature; gate verdict recorded in `results/exp1_summary.md`.
- [ ] Calibration fraction reported (true value inside sampled central interval).

---

## Commit

`feat(zeroshot-cf): Experiment 1 single-feature reconstruction sanity check`
