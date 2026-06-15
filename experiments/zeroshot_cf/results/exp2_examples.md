# Experiment 2: CF Examples — HELOC

Temperature: 1.0, n_permutations: 5, max_context: 256

## Example 1 (idx=0, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 74 | 0.5802 | -73.42 * |
| MSinceOldestTradeOpen | 52 | 52 | +0  |
| MSinceMostRecentTradeOpen | 5 | 5 | +0  |
| AverageMInFile | 26 | 26 | +0  |
| NumSatisfactoryTrades | 16 | -9 | -25 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 100 | -9 | -109 * |
| MSinceMostRecentDelq | -7 | -7 | +0  |
| MaxDelq2PublicRecLast12M | 7 | 9 | +2 * |
| MaxDelqEver | 8 | -9 | -17 * |
| NumTotalTrades | 16 | 16 | +0  |
| NumTradesOpeninLast12M | 2 | -9 | -11 * |
| PercentInstallTrades | 19 | -9 | -28 * |
| MSinceMostRecentInqexcl7days | 0 | 0 | +0  |
| NumInqLast6M | 1 | -9 | -10 * |
| NumInqLast6Mexcl7days | 1 | -9 | -10 * |
| NetFractionRevolvingBurden | 45 | -7.412 | -52.41 * |
| NetFractionInstallBurden | 56 | -8.352 | -64.35 * |
| NumRevolvingTradesWBalance | 3 | -8.756 | -11.76 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 0 | -9 | -9 * |
| PercentTradesWBalance | 56 | -8.967 | -64.97 * |

---

## Example 2 (idx=3, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 75 | -1.885 | -76.89 * |
| MSinceOldestTradeOpen | 360 | 360 | +0  |
| MSinceMostRecentTradeOpen | 4 | 4 | +0  |
| AverageMInFile | 68 | 68 | +0  |
| NumSatisfactoryTrades | 42 | 22.15 | -19.85 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 100 | -9 | -109 * |
| MSinceMostRecentDelq | -7 | -7 | +0  |
| MaxDelq2PublicRecLast12M | 7 | -9 | -16 * |
| MaxDelqEver | 8 | 8 | +0  |
| NumTotalTrades | 22 | 22 | +0  |
| NumTradesOpeninLast12M | 3 | 4.093 | +1.093 * |
| PercentInstallTrades | 14 | 11.28 | -2.717 * |
| MSinceMostRecentInqexcl7days | 0 | 0 | +0  |
| NumInqLast6M | 1 | 3.025 | +2.025 * |
| NumInqLast6Mexcl7days | 1 | 3.015 | +2.015 * |
| NetFractionRevolvingBurden | 11 | 128.9 | +117.9 * |
| NetFractionInstallBurden | 79 | -8.078 | -87.08 * |
| NumRevolvingTradesWBalance | 10 | 3.15 | -6.85 * |
| NumInstallTradesWBalance | 2 | 1.008 | -0.9916 * |
| NumBank2NatlTradesWHighUtilization | 0 | 1.058 | +1.058 * |
| PercentTradesWBalance | 48 | 49.45 | +1.447 * |

---

## Example 3 (idx=4, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 73 | 86.69 | +13.69 * |
| MSinceOldestTradeOpen | 158 | 158 | +0  |
| MSinceMostRecentTradeOpen | 4 | 4 | +0  |
| AverageMInFile | 75 | 75 | +0  |
| NumSatisfactoryTrades | 13 | -2.738 | -15.74 * |
| NumTrades60Ever2DerogPubRec | 1 | -9 | -10 * |
| NumTrades90Ever2DerogPubRec | 1 | -9 | -10 * |
| PercentTradesNeverDelq | 93 | 96.64 | +3.64 * |
| MSinceMostRecentDelq | 80 | 80 | +0  |
| MaxDelq2PublicRecLast12M | 6 | 9 | +3 * |
| MaxDelqEver | 3 | -9 | -12 * |
| NumTotalTrades | 14 | 14 | +0  |
| NumTradesOpeninLast12M | 1 | -9 | -10 * |
| PercentInstallTrades | 57 | 48.04 | -8.961 * |
| MSinceMostRecentInqexcl7days | -7 | -7 | +0  |
| NumInqLast6M | 1 | -9 | -10 * |
| NumInqLast6Mexcl7days | 1 | -9 | -10 * |
| NetFractionRevolvingBurden | 97 | 5.768 | -91.23 * |
| NetFractionInstallBurden | 77 | 15.68 | -61.32 * |
| NumRevolvingTradesWBalance | 2 | 0.0001051 | -2 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 2 | -9 | -11 * |
| PercentTradesWBalance | 100 | 99.64 | -0.3626 * |

---

## Example 4 (idx=7, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 77 | -9 | -86 * |
| MSinceOldestTradeOpen | 229 | 229 | +0  |
| MSinceMostRecentTradeOpen | 2 | 2 | +0  |
| AverageMInFile | 72 | 72 | +0  |
| NumSatisfactoryTrades | 28 | 11.95 | -16.05 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 100 | 99.62 | -0.3808 * |
| MSinceMostRecentDelq | -7 | -7 | +0  |
| MaxDelq2PublicRecLast12M | 7 | -9 | -16 * |
| MaxDelqEver | 8 | -9 | -17 * |
| NumTotalTrades | 33 | 33 | +0  |
| NumTradesOpeninLast12M | 3 | -9 | -12 * |
| PercentInstallTrades | 48 | 100 | +52 * |
| MSinceMostRecentInqexcl7days | 0 | 0 | +0  |
| NumInqLast6M | 10 | -9 | -19 * |
| NumInqLast6Mexcl7days | 10 | -9 | -19 * |
| NetFractionRevolvingBurden | 30 | 0.4322 | -29.57 * |
| NetFractionInstallBurden | -8 | -8.761 | -0.7607 * |
| NumRevolvingTradesWBalance | 6 | -7.9 | -13.9 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 2 | -9 | -11 * |
| PercentTradesWBalance | 50 | -7.429 | -57.43 * |

---

## Example 5 (idx=10, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 67 | -9 | -76 * |
| MSinceOldestTradeOpen | 235 | 235 | +0  |
| MSinceMostRecentTradeOpen | 5 | 5 | +0  |
| AverageMInFile | 39 | 39 | +0  |
| NumSatisfactoryTrades | 11 | 29.12 | +18.12 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 91 | 100 | +9 * |
| MSinceMostRecentDelq | 6 | 6 | +0  |
| MaxDelq2PublicRecLast12M | 4 | 9 | +5 * |
| MaxDelqEver | 6 | -9 | -15 * |
| NumTotalTrades | 30 | 30 | +0  |
| NumTradesOpeninLast12M | 3 | -9 | -12 * |
| PercentInstallTrades | 27 | 100 | +73 * |
| MSinceMostRecentInqexcl7days | -7 | -7 | +0  |
| NumInqLast6M | 1 | -9 | -10 * |
| NumInqLast6Mexcl7days | 1 | -9 | -10 * |
| NetFractionRevolvingBurden | 47 | -7.768 | -54.77 * |
| NetFractionInstallBurden | 78 | 42.95 | -35.05 * |
| NumRevolvingTradesWBalance | 3 | -7.834 | -10.83 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 0 | -9 | -9 * |
| PercentTradesWBalance | 71 | -3.171 | -74.17 * |

---

