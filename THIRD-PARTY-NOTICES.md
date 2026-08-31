# Third-Party Notices

This repository no longer tracks the upstream local model implementation. The
retained CounterContEX surface is a thin orchestration layer around third-party
packages and one pinned vendor bootstrap. Transitive dependencies are locked in
`uv.lock` and `experiments/zeroshot_cf/uv.lock`; their licenses remain governed
by their distributors.

---

## Summary

| Component | Usage surface | License | Notes |
|---|---|---|---|
| `tabicl` | Core generator and checkpoint runtime | BSD-3-Clause | Upstream distribution also carries an Apache-2.0 notice for its forecast subtree |
| Counterfactual Explanations Library (CEL) | Pinned local bootstrap fetched by `vendor_setup.py` | MIT | The repository pins revision `3587f943826f6b087a0d198c8c4aa4373712c7ee`; the checkout is local-only and ignored by git |
| `dice-ml` | Exp13 DiCE baseline runtime | MIT | Imported only when the DiCE baseline runs |
| `raiutils` | Direct retained baseline dependency | MIT | Required by the retained baseline environment |

---

## Notes

- No third-party source tree is tracked in the repository root after the cleanup.
- `vendor_setup.py` materializes CEL into an ignored local checkout, so license
  review for redistributed raw datasets remains the operator's responsibility.
- Add a row here if future work vendors source code or introduces a direct
  runtime dependency whose notice must be surfaced at the repository level.
