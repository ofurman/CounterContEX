# Stage 11: Second foundation backend and robustness

**Goal**: Show the contribution is about tabular foundation models rather than about TabICL
specifically, and measure how counterfactuals survive classifier retraining.
**Dependencies**: Stages 6, 7

Closes gaps B10 and B11. E9 is the stronger of the two: if results hold under a second
foundation model, the claim generalises from "TabICL enables this" to "tabular foundation models
enable this."

---

## Steps

1. **Implement a second proposal backend.** TabPFN v2 or TabICLv2, behind the existing
   `ProposalSession` contract in
   `experiments/zeroshot_cf/methods/countercontex/backends/base.py`. That boundary was built for
   exactly this and has never been exercised by a second foundation model, so expect it to need
   adjustment — record any contract change in `decisions.md`.
   - Declare capabilities honestly. If the new backend cannot do confidence conditioning or joint
     scoring, say so in `ProposalCapabilities` and let unsupported combinations fail during
     preparation, before the search loop.
   - Register a distinct `backend_implementation` version and its own checkpoint content IDs.

2. **E9 — foundation-model swap** (`campaign_e9_fmswap.yaml`). Two datasets × both backends ×
   three seeds, everything else held fixed to the Stage 10 frozen configuration.

3. **E8 — robustness to retraining.** No generation run: retrain each target model under three
   fresh seeds and rescore the counterfactuals already stored in the Stage 7 `arrays.npz` files.
   - Report the fraction of counterfactuals that remain valid, split by method and by original
     `p_f(y*|x')` from the Stage 3 arrays. The prediction from the boundary-hugging finding is
     that candidates near 0.5 fail most — measuring that link is what makes E8 worth reporting
     rather than a bare percentage.
   - The retrained models are **evaluation instruments, not target models**. They must not enter
     any run identity or overwrite a cached classifier used by the campaign.

4. Produce the E8 and E9 tables through the Stage 5 analysis layer.

---

## Verification

- [ ] GATE The second backend declares its capabilities and an unsupported search/backend
      combination raises during method preparation, before any classifier call — read from the
      preparation call in a contract test. A backend that silently degrades turns it red.
- [ ] GATE E8 rescoring leaves every Stage 7 artifact byte-identical and creates no run directory
      under a Stage 7 output root — read by comparing file digests before and after. Overwriting
      historical evidence turns it red.
- [ ] REPORT E9 metric differences between the two foundation backends against the Stage 1 noise
      floor, and E8 validity retention by method and by original target probability — record in
      `journal.md`.

---

## Commit

`feat(countercontex): add a second foundation proposal backend; record E8-E9 results`
