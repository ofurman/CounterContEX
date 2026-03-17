# Templates & Examples

## Full Plan Template

Use this template when materializing a complex plan. The plan document is
**self-contained**: any AI agent can read it cold and know exactly how to execute it.

````markdown
# Plan: [Task Name]

## Overview
[What this plan implements and why — one paragraph]

## Execution Protocol

To execute this plan, follow this loop for each stage:

1. **Read the progress tracker** below and find the first stage that is not DONE
2. **Read the stage details** — understand the goal, dependencies, and steps
3. **Clarify ambiguities** — if anything is unclear or multiple approaches exist,
   ask the user before implementing. Do not guess.
4. **Implement** — execute the steps described in the stage
5. **Validate** — run the verification checks listed in the stage.
   If validation fails, fix the issue before proceeding. Do not skip verification.
6. **Update this plan** — mark the stage as DONE in the progress tracker,
   add brief notes about what was done and any deviations from the original steps
7. **Commit** — create an atomic commit with the message specified in the stage.
   Include all changed files (code, config, docs, and this plan file).

Repeat until all stages are DONE or a stage is BLOCKED.

**If a stage cannot be completed**: mark it BLOCKED in the tracker with a note
explaining why, and stop. Do not proceed to subsequent stages.

**If assumptions are wrong**: stop, document the issue in the Issues section below,
revise affected stages, and get user confirmation before continuing.

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Stage Name] | PENDING | | |
| 2 | [Stage Name] | PENDING | | |
| 3 | [Stage Name] | PENDING | | |

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED`

---

## Stage 1: [Name]
**Goal**: [What this stage accomplishes — one sentence]
**Dependencies**: [What must be done first, or "None"]

**Steps**:
1. [Specific action — include file paths, line references, exact values]
2. [Specific action]

**Verification**:
- [ ] [Concrete test command or check with expected outcome]
- [ ] [Another verification step]

**Commit**: `type(scope): description`

---

## Stage 2: [Name]
**Goal**: [What this stage accomplishes]
**Dependencies**: Stage 1

**Steps**:
1. [Specific action]

**Verification**:
- [ ] [Test command or check]

**Commit**: `type(scope): description`

---

## Overall Verification
[End-to-end validation command or checklist to run after all stages are complete]

## Issues
[Document any problems discovered during execution]

### Issue: [Title]
- **Expected**: [What should have happened]
- **Actual**: [What happened]
- **Impact**: [How this affects the plan]
- **Resolution**: [What was done or proposed]

## Decisions
[Record architectural or approach decisions made during planning or execution]

### Decision: [Title]
- **Options**: A) [...] B) [...] C) [...]
- **Chosen**: [Option]
- **Rationale**: [Why]
````

---

## Simple Plan Template

For tasks with fewer than 5 steps that don't need stages:

````markdown
# Plan: [Task Name]

## Overview
[What and why — one paragraph]

## Steps
1. [Action verb] [specific change]
   - File: [path]
   - Details: [what to change]
2. [Action verb] [specific change]
   ...

## Verification
- [ ] [Test command or check]

## Commit
`type(scope): brief description`
````

---

## Example 1: Simple Task

**Task**: "Fix the scoring threshold for matching"

**Analysis** — Read config, found threshold at 22.0, evaluation shows 21.5 is optimal.

**Clarification** — Skipped (straightforward change).

**Plan**:

```markdown
# Plan: Lower Scoring Threshold

## Overview
Lower threshold from 22.0 to 21.5 based on evaluation results.
Converts ~130 borderline predictions to matches, improving recall by ~2-3%.

## Steps
1. Update `match_threshold` in `config/base.yaml` line ~45
   - Change: `match_threshold: 22.0` → `match_threshold: 21.5`
2. Run scoring pipeline to validate
3. Run evaluation to confirm recall improvement

## Verification
- [ ] Scoring runs without errors
- [ ] Recall improved to ~71-73%
- [ ] Precision remains >99%

## Commit
`feat(scoring): lower match threshold to 21.5 for improved recall`
```

**Execution** — All steps done, recall 68.6% → 71.2%, committed.

---

## Example 2: Complex Task with Stages

**Task**: "Add email address matching support"

**Analysis** — Existing pipeline has email features but doesn't use them for scoring.

**Clarification** — Asked user to choose between exact-only vs exact+domain matching. User chose exact-only.

**Plan**:

```markdown
# Plan: Add Email Matching

## Overview
Add email as a high-confidence matching signal. Features already exist
in the pipeline; this adds the matching rule and scoring weight.

## Execution Protocol

To execute this plan, follow this loop for each stage:

1. Read the progress tracker and find the first stage that is not DONE
2. Read the stage details — understand the goal, dependencies, and steps
3. Clarify ambiguities — ask the user if anything is unclear
4. Implement — execute the steps described in the stage
5. Validate — run verification checks; fix failures before proceeding
6. Update this plan — mark stage DONE, add notes
7. Commit — atomic commit with message from the stage, including this plan file

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | Add Matching Rule | DONE | Generated 1,171 pairs | abc1234 |
| 2 | Add Scoring Weight | DONE | Weight 15.0 applied | def5678 |
| 3 | Validation | DONE | +0.44% recall, 99.95% precision | ghi9012 |

---

## Stage 1: Add Matching Rule
**Goal**: Generate candidate pairs from exact email matches
**Dependencies**: None (feature data already exists)

**Steps**:
1. Add `email_exact` rule to `config/matching.yaml`
   - Copy pattern from `phone_exact` rule (lines 23-29)
   - Set source to email features table
   - Set confidence tier to highest

**Verification**:
- [x] Config validates: `pytest tests/test_config.py`
- [x] Pipeline generates candidate pairs from email matches

**Commit**: `feat(matching): add exact email matching rule`

---

## Stage 2: Add Scoring Weight
**Goal**: Score email matches as high-confidence signals
**Dependencies**: Stage 1

**Steps**:
1. Add weight for `email_exact` in `config/scoring.yaml`
   - Weight: 15.0 (same tier as phone match)

**Verification**:
- [x] Config validates
- [x] Email matches score >15 in output

**Commit**: `feat(scoring): add weight 15.0 for exact email match`

---

## Stage 3: Validation
**Goal**: Confirm improvement without regression
**Dependencies**: Stage 2

**Steps**:
1. Run full evaluation pipeline
2. Update documentation with results

**Verification**:
- [x] Recall improved (~+1%)
- [x] Precision maintained (>99.5%)
- [x] Docs updated

**Commit**: `docs(matching): add email matching validation results`

---

## Overall Verification
Run end-to-end pipeline and check metrics show improvement
without regression in existing match quality.

## Issues
None encountered.

## Decisions

### Decision: Email matching approach
- **Options**: A) Exact match only B) Exact + domain matching
- **Chosen**: A
- **Rationale**: Higher precision, simpler to validate. Domain matching can be added later.
```
