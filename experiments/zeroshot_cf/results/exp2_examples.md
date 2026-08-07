# Experiment 2: CF Examples — GERMAN

Temperature: 1.0, n_permutations: 5, max_context: 256

## Example 1 (idx=0, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| Checking-account | 3 | 0 | -3 * |
| Months | 21 | 8 | -13 * |
| Credit-history | 4 | 0 | -4 * |
| Purpose | 3 | 3 | +0  |
| Credit-amount | 3146 | 806.2 | -2340 * |
| Savings-account | 3 | 0 | -3 * |
| Present-employment-since | 4 | 0 | -4 * |
| Insatllment-rate | 1 | 3 | +2 * |
| Personal-status | 1 | 1 | +0  |
| Other-debtors | 0 | 0 | +0  |
| Present-residence-since | 1 | 0 | -1 * |
| Property | 2 | 0 | -2 * |
| age | 37.5 | 23 | -14.5 * |
| Other-installment-plans | 2 | 2 | +0  |
| Housing | 1 | 0 | -1 * |
| Number-of-existing-credits | 0 | 0 | +0  |
| Job | 2 | 0 | -2 * |
| Number-of-people-being-lible | 0 | 0 | +0  |
| Telephone | 0 | 0 | +0  |
| Foreign-worker | 0 | 0 | +0  |

---

## Example 2 (idx=4, VALID)
Factual class: 1, CF target: 0, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| Checking-account | 1 | 0 | -1 * |
| Months | 21 | 8 | -13 * |
| Credit-history | 4 | 0 | -4 * |
| Purpose | 3 | 3 | +0  |
| Credit-amount | 806.2 | 806.2 | +0  |
| Savings-account | 1 | 0 | -1 * |
| Present-employment-since | 1 | 0 | -1 * |
| Insatllment-rate | 3 | 0 | -3 * |
| Personal-status | 1 | 1 | +0  |
| Other-debtors | 0 | 0 | +0  |
| Present-residence-since | 3 | 0 | -3 * |
| Property | 3 | 3 | +0  |
| age | 37.5 | 58.5 | +21 * |
| Other-installment-plans | 0 | 2 | +2 * |
| Housing | 1 | 2 | +1 * |
| Number-of-existing-credits | 1 | 0 | -1 * |
| Job | 2 | 0 | -2 * |
| Number-of-people-being-lible | 1 | 1 | +0  |
| Telephone | 0 | 0 | +0  |
| Foreign-worker | 0 | 0 | +0  |

---

## Example 3 (idx=1, INVALID)
Factual class: 0, CF target: 1, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| Checking-account | 1 | 0 | -1 * |
| Months | 21 | 8 | -13 * |
| Credit-history | 2 | 0 | -2 * |
| Purpose | 0 | 0 | +0  |
| Credit-amount | 3146 | 806.2 | -2340 * |
| Savings-account | 1 | 0 | -1 * |
| Present-employment-since | 3 | 0 | -3 * |
| Insatllment-rate | 1 | 3 | +2 * |
| Personal-status | 1 | 1 | +0  |
| Other-debtors | 0 | 0 | +0  |
| Present-residence-since | 2 | 3 | +1 * |
| Property | 2 | 0 | -2 * |
| age | 37.5 | 23 | -14.5 * |
| Other-installment-plans | 0 | 0 | +0  |
| Housing | 1 | 0 | -1 * |
| Number-of-existing-credits | 1 | 0 | -1 * |
| Job | 2 | 0 | -2 * |
| Number-of-people-being-lible | 0 | 0 | +0  |
| Telephone | 1 | 0 | -1 * |
| Foreign-worker | 0 | 0 | +0  |

---

## Example 4 (idx=2, INVALID)
Factual class: 0, CF target: 1, CF predicted: 0

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| Checking-account | 1 | 0 | -1 * |
| Months | 21 | 48 | +27 * |
| Credit-history | 0 | 0 | +0  |
| Purpose | 8 | 8 | +0  |
| Credit-amount | 1.12e+04 | 1.12e+04 | +0  |
| Savings-account | 0 | 0 | +0  |
| Present-employment-since | 2 | 4 | +2 * |
| Insatllment-rate | 0 | 0 | +0  |
| Personal-status | 1 | 1 | +0  |
| Other-debtors | 0 | 0 | +0  |
| Present-residence-since | 3 | 3 | +0  |
| Property | 0 | 0 | +0  |
| age | 37.5 | 23 | -14.5 * |
| Other-installment-plans | 2 | 0 | -2 * |
| Housing | 1 | 0 | -1 * |
| Number-of-existing-credits | 2 | 0 | -2 * |
| Job | 1 | 3 | +2 * |
| Number-of-people-being-lible | 0 | 0 | +0  |
| Telephone | 1 | 0 | -1 * |
| Foreign-worker | 0 | 0 | +0  |

---

## Example 5 (idx=3, INVALID)
Factual class: 1, CF target: 0, CF predicted: 1

| Feature | Factual | Counterfactual | Delta |
|---------|---------|---------------|-------|
| Checking-account | 1 | 3 | +2 * |
| Months | 15 | 8 | -7 * |
| Credit-history | 2 | 0 | -2 * |
| Purpose | 0 | 0 | +0  |
| Credit-amount | 1.12e+04 | 806.2 | -1.039e+04 * |
| Savings-account | 0 | 0 | +0  |
| Present-employment-since | 3 | 0 | -3 * |
| Insatllment-rate | 2 | 0 | -2 * |
| Personal-status | 1 | 1 | +0  |
| Other-debtors | 0 | 0 | +0  |
| Present-residence-since | 2 | 0 | -2 * |
| Property | 0 | 0 | +0  |
| age | 30 | 23 | -7 * |
| Other-installment-plans | 2 | 0 | -2 * |
| Housing | 0 | 0 | +0  |
| Number-of-existing-credits | 0 | 0 | +0  |
| Job | 1 | 0 | -1 * |
| Number-of-people-being-lible | 0 | 0 | +0  |
| Telephone | 0 | 0 | +0  |
| Foreign-worker | 0 | 0 | +0  |

---

