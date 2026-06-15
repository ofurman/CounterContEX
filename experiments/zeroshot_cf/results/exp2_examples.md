# Experiment 2: CF Examples — HELOC

Temperature: 1.0, n_permutations: 5, max_context: 256

## Example 1 (idx=2, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 82 | -5.641 | -87.64 * |
| MSinceOldestTradeOpen | 270 | 270 | +0  |
| MSinceMostRecentTradeOpen | 30 | 30 | +0  |
| AverageMInFile | 89 | 89 | +0  |
| NumSatisfactoryTrades | 12 | 13.83 | +1.831 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 100 | 95.72 | -4.275 * |
| MSinceMostRecentDelq | -7 | -7 | +0  |
| MaxDelq2PublicRecLast12M | 7 | -9 | -16 * |
| MaxDelqEver | 8 | -9 | -17 * |
| NumTotalTrades | 12 | 12 | +0  |
| NumTradesOpeninLast12M | 0 | 0.01353 | +0.01353 * |
| PercentInstallTrades | 42 | 25.09 | -16.91 * |
| MSinceMostRecentInqexcl7days | 0 | 0 | +0  |
| NumInqLast6M | 0 | 0.9831 | +0.9831 * |
| NumInqLast6Mexcl7days | 0 | 0.9742 | +0.9742 * |
| NetFractionRevolvingBurden | 44 | 36.01 | -7.987 * |
| NetFractionInstallBurden | 6 | 76.03 | +70.03 * |
| NumRevolvingTradesWBalance | 3 | 2.18 | -0.8202 * |
| NumInstallTradesWBalance | 2 | 1.99 | -0.009757 * |
| NumBank2NatlTradesWHighUtilization | 0 | 0.9461 | +0.9461 * |
| PercentTradesWBalance | 71 | 75.14 | +4.138 * |

---

## Example 2 (idx=3, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 75 | 62.23 | -12.77 * |
| MSinceOldestTradeOpen | 360 | 360 | +0  |
| MSinceMostRecentTradeOpen | 4 | 4 | +0  |
| AverageMInFile | 68 | 68 | +0  |
| NumSatisfactoryTrades | 42 | 6.517 | -35.48 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 100 | -9 | -109 * |
| MSinceMostRecentDelq | -7 | -7 | +0  |
| MaxDelq2PublicRecLast12M | 7 | -9 | -16 * |
| MaxDelqEver | 8 | 8 | +0  |
| NumTotalTrades | 22 | 22 | +0  |
| NumTradesOpeninLast12M | 3 | 11.55 | +8.546 * |
| PercentInstallTrades | 14 | 37.47 | +23.47 * |
| MSinceMostRecentInqexcl7days | 0 | 0 | +0  |
| NumInqLast6M | 1 | 2.064 | +1.064 * |
| NumInqLast6Mexcl7days | 1 | 1.989 | +0.9892 * |
| NetFractionRevolvingBurden | 11 | 66.09 | +55.09 * |
| NetFractionInstallBurden | 79 | 91.21 | +12.21 * |
| NumRevolvingTradesWBalance | 10 | 2.995 | -7.005 * |
| NumInstallTradesWBalance | 2 | 1.97 | -0.02992 * |
| NumBank2NatlTradesWHighUtilization | 0 | 1.054 | +1.054 * |
| PercentTradesWBalance | 48 | 100 | +52 * |

---

## Example 3 (idx=4, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 73 | 48.28 | -24.72 * |
| MSinceOldestTradeOpen | 158 | 158 | +0  |
| MSinceMostRecentTradeOpen | 4 | 4 | +0  |
| AverageMInFile | 75 | 75 | +0  |
| NumSatisfactoryTrades | 13 | 55.92 | +42.92 * |
| NumTrades60Ever2DerogPubRec | 1 | -9 | -10 * |
| NumTrades90Ever2DerogPubRec | 1 | -9 | -10 * |
| PercentTradesNeverDelq | 93 | -9 | -102 * |
| MSinceMostRecentDelq | 80 | 80 | +0  |
| MaxDelq2PublicRecLast12M | 6 | 9 | +3 * |
| MaxDelqEver | 3 | -9 | -12 * |
| NumTotalTrades | 14 | 14 | +0  |
| NumTradesOpeninLast12M | 1 | -9 | -10 * |
| PercentInstallTrades | 57 | -9 | -66 * |
| MSinceMostRecentInqexcl7days | -7 | -7 | +0  |
| NumInqLast6M | 1 | -9 | -10 * |
| NumInqLast6Mexcl7days | 1 | -9 | -10 * |
| NetFractionRevolvingBurden | 97 | -1.481 | -98.48 * |
| NetFractionInstallBurden | 77 | -8.641 | -85.64 * |
| NumRevolvingTradesWBalance | 2 | -8.578 | -10.58 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 2 | -9 | -11 * |
| PercentTradesWBalance | 100 | 28.45 | -71.55 * |

---

## Example 4 (idx=5, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 90 | 44.12 | -45.88 * |
| MSinceOldestTradeOpen | 427 | 427 | +0  |
| MSinceMostRecentTradeOpen | 1 | 1 | +0  |
| AverageMInFile | 133 | 133 | +0  |
| NumSatisfactoryTrades | 6 | 1.949 | -4.051 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 100 | -7.149 | -107.1 * |
| MSinceMostRecentDelq | -7 | -7 | +0  |
| MaxDelq2PublicRecLast12M | 7 | -9 | -16 * |
| MaxDelqEver | 8 | -9 | -17 * |
| NumTotalTrades | 7 | 7 | +0  |
| NumTradesOpeninLast12M | 1 | 3.038 | +2.038 * |
| PercentInstallTrades | 0 | 25.89 | +25.89 * |
| MSinceMostRecentInqexcl7days | 0 | 0 | +0  |
| NumInqLast6M | 1 | 10.74 | +9.743 * |
| NumInqLast6Mexcl7days | 1 | 10.23 | +9.229 * |
| NetFractionRevolvingBurden | 5 | 68.49 | +63.49 * |
| NetFractionInstallBurden | -8 | 35.33 | +43.33 * |
| NumRevolvingTradesWBalance | 5 | 1.027 | -3.973 * |
| NumInstallTradesWBalance | -8 | 0.9656 | +8.966 * |
| NumBank2NatlTradesWHighUtilization | 0 | 0.9693 | +0.9693 * |
| PercentTradesWBalance | 71 | 100 | +29 * |

---

## Example 5 (idx=7, VALID)
Factual class: 0, CF target: 1, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| ExternalRiskEstimate | 77 | 60.26 | -16.74 * |
| MSinceOldestTradeOpen | 229 | 229 | +0  |
| MSinceMostRecentTradeOpen | 2 | 2 | +0  |
| AverageMInFile | 72 | 72 | +0  |
| NumSatisfactoryTrades | 28 | 8.882 | -19.12 * |
| NumTrades60Ever2DerogPubRec | 0 | -9 | -9 * |
| NumTrades90Ever2DerogPubRec | 0 | -9 | -9 * |
| PercentTradesNeverDelq | 100 | 99.99 | -0.01193 * |
| MSinceMostRecentDelq | -7 | -7 | +0  |
| MaxDelq2PublicRecLast12M | 7 | 9 | +2 * |
| MaxDelqEver | 8 | 8 | +0  |
| NumTotalTrades | 33 | 33 | +0  |
| NumTradesOpeninLast12M | 3 | -9 | -12 * |
| PercentInstallTrades | 48 | 100 | +52 * |
| MSinceMostRecentInqexcl7days | 0 | 0 | +0  |
| NumInqLast6M | 10 | -9 | -19 * |
| NumInqLast6Mexcl7days | 10 | -9 | -19 * |
| NetFractionRevolvingBurden | 30 | -0.1889 | -30.19 * |
| NetFractionInstallBurden | -8 | 20.45 | +28.45 * |
| NumRevolvingTradesWBalance | 6 | -7.73 | -13.73 * |
| NumInstallTradesWBalance | 2 | -9 | -11 * |
| NumBank2NatlTradesWHighUtilization | 2 | -9 | -11 * |
| PercentTradesWBalance | 50 | -6.794 | -56.79 * |

---

