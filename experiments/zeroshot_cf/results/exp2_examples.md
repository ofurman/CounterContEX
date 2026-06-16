# Experiment 2: CF Examples — MOONS

Temperature: 1.0, n_permutations: 5, max_context: 256

## Example 1 (idx=0, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | 0.7692 | -0.2204 | -0.9896 * |
| 1 | -0.4074 | 1.12 | +1.528 * |

---

## Example 2 (idx=1, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | -0.807 | 0.6074 | +1.414 * |
| 1 | 0.2121 | -0.5323 | -0.7444 * |

---

## Example 3 (idx=2, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | 1.555 | -0.2264 | -1.781 * |
| 1 | -0.2855 | 0.9418 | +1.227 * |

---

## Example 4 (idx=3, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | -0.8687 | 0.6421 | +1.511 * |
| 1 | 0.02874 | -0.445 | -0.4738 * |

---

## Example 5 (idx=4, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| 0 | 1.143 | -0.2542 | -1.397 * |
| 1 | -0.6148 | 0.9541 | +1.569 * |

---

# Experiment 2: CF Examples — HELOC

Temperature: 1.0, n_permutations: 5, max_context: 256

## Example 1 (idx=4, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 73 | -9 | -82 * |
| MSinceOldestTradeOpen | 158 | 158 | +0  |
| MSinceMostRecentTradeOpen | 4 | 4 | +0  |
| AverageMInFile | 75 | 75 | +0  |
| NumSatisfactoryTrades | 13 | 6.315 | -6.685 * |
| NumTrades60Ever2DerogPubRec | 1 | -9 | -10 * |
| NumTrades90Ever2DerogPubRec | 1 | -9 | -10 * |
| PercentTradesNeverDelq | 93 | 94.4 | +1.396 * |
| MSinceMostRecentDelq | 80 | 80 | +0  |
| MaxDelq2PublicRecLast12M | 6 | 9 | +3 * |
| MaxDelqEver | 3 | 8 | +5 * |
| NumTotalTrades | 14 | 14 | +0  |
| NumTradesOpeninLast12M | 1 | -9 | -10 * |
| PercentInstallTrades | 57 | 100 | +43 * |
| MSinceMostRecentInqexcl7days | -7 | -7 | +0  |
| NumInqLast6M | 1 | -9 | -10 * |
| NumInqLast6Mexcl7days | 1 | -9 | -10 * |
| NetFractionRevolvingBurden | 97 | -7.898 | -104.9 * |
| NetFractionInstallBurden | 77 | -7.906 | -84.91 * |
| NumRevolvingTradesWBalance | 2 | -7.279 | -9.279 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 2 | -9 | -11 * |
| PercentTradesWBalance | 100 | -3.195 | -103.2 * |

---

## Example 2 (idx=6, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 82 | 27.01 | -54.99 * |
| MSinceOldestTradeOpen | 194 | 194 | +0  |
| MSinceMostRecentTradeOpen | 6 | 6 | +0  |
| AverageMInFile | 87 | 87 | +0  |
| NumSatisfactoryTrades | 33 | 3.722 | -29.28 * |
| NumTrades60Ever2DerogPubRec | 1 | -9 | -10 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 97 | -9 | -106 * |
| MSinceMostRecentDelq | 66 | 66 | +0  |
| MaxDelq2PublicRecLast12M | 6 | -9 | -15 * |
| MaxDelqEver | 5 | -9 | -14 * |
| NumTotalTrades | 34 | 34 | +0  |
| NumTradesOpeninLast12M | 2 | 3.003 | +1.003 * |
| PercentInstallTrades | 21 | 30.37 | +9.375 * |
| MSinceMostRecentInqexcl7days | 10 | 10 | +0  |
| NumInqLast6M | 0 | -0.08609 | -0.08609 * |
| NumInqLast6Mexcl7days | 0 | 0.004767 | +0.004767 * |
| NetFractionRevolvingBurden | 2 | 0.9588 | -1.041 * |
| NetFractionInstallBurden | 84 | 48.51 | -35.49 * |
| NumRevolvingTradesWBalance | 5 | 0.8889 | -4.111 * |
| NumInstallTradesWBalance | 2 | 2.97 | +0.9702 * |
| NumBank2NatlTradesWHighUtilization | 0 | -0.05134 | -0.05134 * |
| PercentTradesWBalance | 44 | 65.71 | +21.71 * |

---

## Example 3 (idx=7, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 77 | 82.16 | +5.157 * |
| MSinceOldestTradeOpen | 229 | 229 | +0  |
| MSinceMostRecentTradeOpen | 2 | 2 | +0  |
| AverageMInFile | 72 | 72 | +0  |
| NumSatisfactoryTrades | 28 | 18.1 | -9.905 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 100 | 93.71 | -6.295 * |
| MSinceMostRecentDelq | -7 | -7 | +0  |
| MaxDelq2PublicRecLast12M | 7 | -9 | -16 * |
| MaxDelqEver | 8 | -9 | -17 * |
| NumTotalTrades | 33 | 33 | +0  |
| NumTradesOpeninLast12M | 3 | -9 | -12 * |
| PercentInstallTrades | 48 | 83.2 | +35.2 * |
| MSinceMostRecentInqexcl7days | 0 | 0 | +0  |
| NumInqLast6M | 10 | -9 | -19 * |
| NumInqLast6Mexcl7days | 10 | -9 | -19 * |
| NetFractionRevolvingBurden | 30 | 0.4391 | -29.56 * |
| NetFractionInstallBurden | -8 | -8 | -0.0002099 * |
| NumRevolvingTradesWBalance | 6 | 0.002052 | -5.998 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 2 | -9 | -11 * |
| PercentTradesWBalance | 50 | -3.124 | -53.12 * |

---

## Example 4 (idx=8, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 59 | 24.88 | -34.12 * |
| MSinceOldestTradeOpen | 255 | 255 | +0  |
| MSinceMostRecentTradeOpen | 39 | 39 | +0  |
| AverageMInFile | 111 | 111 | +0  |
| NumSatisfactoryTrades | 11 | 9.497 | -1.503 * |
| NumTrades60Ever2DerogPubRec | 1 | -9 | -10 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 75 | 90.19 | +15.19 * |
| MSinceMostRecentDelq | 3 | 3 | +0  |
| MaxDelq2PublicRecLast12M | 4 | -9 | -13 * |
| MaxDelqEver | 5 | -9 | -14 * |
| NumTotalTrades | 12 | 12 | +0  |
| NumTradesOpeninLast12M | 0 | -9 | -9 * |
| PercentInstallTrades | 50 | 88.02 | +38.02 * |
| MSinceMostRecentInqexcl7days | 6 | 6 | +0  |
| NumInqLast6M | 0 | -9 | -9 * |
| NumInqLast6Mexcl7days | 0 | -9 | -9 * |
| NetFractionRevolvingBurden | 91 | 7.333 | -83.67 * |
| NetFractionInstallBurden | 83 | -8.292 | -91.29 * |
| NumRevolvingTradesWBalance | 5 | -0.04434 | -5.044 * |
| NumInstallTradesWBalance | 1 | -9 | -10 * |
| NumBank2NatlTradesWHighUtilization | 2 | -9 | -11 * |
| PercentTradesWBalance | 100 | 81.32 | -18.68 * |

---

## Example 5 (idx=10, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 67 | 28.99 | -38.01 * |
| MSinceOldestTradeOpen | 235 | 235 | +0  |
| MSinceMostRecentTradeOpen | 5 | 5 | +0  |
| AverageMInFile | 39 | 39 | +0  |
| NumSatisfactoryTrades | 11 | 32.47 | +21.47 * |
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
| NetFractionRevolvingBurden | 47 | 114.5 | +67.46 * |
| NetFractionInstallBurden | 78 | -8.668 | -86.67 * |
| NumRevolvingTradesWBalance | 3 | -8.118 | -11.12 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 0 | -9 | -9 * |
| PercentTradesWBalance | 71 | -7.049 | -78.05 * |

---

