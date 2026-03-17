---
name: task-planning
description: "Systematic task planning and execution framework. Guides through analysis, clarification, plan materialization with stages, progress tracking, verification, and atomic commits. Supports both interactive stage-by-stage execution and headless automation via companion bash script. Use for multi-step tasks, architectural changes, or complex problem-solving. Trigger when user asks to plan, break down, structure, or execute implementation of a non-trivial task."
---

# Task Planning

Systematic workflow for planning and executing non-trivial tasks.

## Contents

- [Quick Start Checklist](#quick-start-checklist)
- [Phase 1: Analysis](#phase-1-analysis)
- [Phase 2: Clarification](#phase-2-clarification)
- [Phase 3: Plan Materialization](#phase-3-plan-materialization)
- [Phase 4: Execution](#phase-4-execution)
- [Headless Execution](headless.md)
- [Templates & Examples](templates.md)

## Quick Start Checklist

Copy and track progress:

```
Planning Progress:
- [ ] Phase 1: Analyze context (read docs, explore code, identify root cause)
- [ ] Phase 2: Clarify ambiguity (ask questions if needed)
- [ ] Phase 3: Write plan (stages, steps, verification)
- [ ] Phase 4: Execute (implement, verify, commit per stage)
```

---

## Phase 1: Analysis

**Goal**: Understand the problem space before proposing solutions.

1. **Read relevant docs** -- project README, CLAUDE.md, architecture docs, recent work summaries
2. **Explore the codebase** -- entry points, execution flow, data structures, existing tests
3. **Identify root cause** -- don't treat symptoms; use git history, dependency analysis, impact analysis
4. **Synthesize findings** -- document problem statement, context, and 2-3 potential approaches with trade-offs

Present findings before moving to Phase 2:

```markdown
**Problem**: [What is actually broken or missing]
**Context**: [Relevant architecture, patterns, constraints]
**Approaches**:
  A. [Description] -- Pros: [...] Cons: [...]
  B. [Description] -- Pros: [...] Cons: [...]
**Recommendation**: [Which approach and why]
```

---

## Phase 2: Clarification

**Goal**: Resolve ambiguity before writing the plan.

### Ask questions when

- Multiple valid approaches exist and trade-offs need user input
- Scope is uncertain (minimal fix vs complete solution)
- Architecture choices affect future work
- Breaking changes are involved

### Don't ask about

- Standard patterns (follow existing code)
- Code formatting (use project conventions)
- How to use tools (read docs first)

### Question format

Use AskUserQuestion with options structured as:

```
Option 1: [Brief description]
- Pros: [benefits]
- Cons: [drawbacks]

Option 2: [Brief description]
- Pros: [benefits]
- Cons: [drawbacks]

Recommendation: Option [X] because [reasoning]
```

Skip Phase 2 entirely if the task is straightforward with clear requirements.

---

## Phase 3: Plan Materialization

**Goal**: Write a clear, actionable plan.

### Simple tasks (<5 steps)

```markdown
## Plan: [Task Name]

### Overview
[What and why -- one paragraph]

### Steps
1. [Action verb] [specific change]
   - File: [path]
   - Details: [what to change]
2. [Action verb] [specific change]
   ...

### Verification
- [ ] [Test command or check]

### Commit
`type(scope): brief description`
```

### Complex tasks (>=5 steps or multiple areas)

Split into stages. Each stage must be:

- **Independently testable** -- can verify without completing later stages
- **Committable** -- leaves codebase in a working state
- **Logically cohesive** -- groups related changes together

The plan document must include an **Execution Protocol** section so any AI agent
can pick it up and execute it autonomously. See [templates.md](templates.md) for
the full plan template with embedded execution instructions.

### Writing effective steps

Each step must be:

- **Specific** -- "Add `timeout: 30` to `config/settings.yaml` line 45" not "Update config"
- **Actionable** -- can be done without further research
- **Testable** -- has a verification criterion

### Stage design principles

1. **Logical dependencies** -- Stage 2 depends on Stage 1 explicitly
2. **Incremental value** -- each stage adds usable functionality, not just "setup"
3. **Clear boundaries** -- no overlap between stages

For detailed templates and examples, see [templates.md](templates.md).

---

## Phase 4: Execution

**Goal**: Implement the plan stage by stage with progress tracking, verification, and atomic commits.

Works in two modes:
- **Interactive**: Within a Claude Code session (this workflow)
- **Headless**: Via `scripts/plan_runner.sh` for unattended execution -- see [headless.md](headless.md)

### Step 1: Plan Analysis

1. **Read the plan file** completely
2. **Identify all stages** -- look for `## Stage N:` headings or numbered top-level sections
3. **Locate the progress tracker** -- search for "Progress Tracker"
4. **If no tracker exists**, create one at the bottom of the plan using the template from [templates.md](templates.md)
5. **Report to the user**: number of stages, which are already DONE (if resuming), which stage is next
6. **Ask for confirmation** before starting execution

### Step 2: Stage Execution Loop

For each stage (in order), starting from the first non-DONE stage:

#### Pre-flight Check
- Read the progress tracker to confirm prior stages are DONE
- If a dependency is not met, mark the current stage BLOCKED and stop
- Set the current stage status to `IN_PROGRESS` in the tracker

#### Implement
- Read the stage's steps from the plan
- Implement each step, following the project's CLAUDE.md coding discipline
- Only implement what the current stage specifies -- do not skip ahead

#### Verify
- Run the verification commands/checks listed in the stage
- Run project tests if available (`poetry run pytest`, `npm test`, etc.)
- If verification fails, attempt to fix. If unfixable, mark BLOCKED with details

#### Update Tracker
- Set the stage status to `DONE` in the tracker table
- Add brief notes about what was implemented
- Update "Last stage completed" and "Last updated by"

#### Commit
- Stage all changed files relevant to this stage
- Commit with the message specified in the plan stage
- If pre-commit hooks reformat files, re-stage and retry (do NOT use `--no-verify`)
- Append `Co-Authored-By: Claude <model> <noreply@anthropic.com>`
- One stage = one commit. Never batch multiple stages.
- Include the plan file update (tracker status change) in the commit

#### Continue or Stop
- If all stages are DONE, report summary and create signal file: `echo 'DONE' > .plan_runner_done`
- If BLOCKED, create signal file with reason and stop
- Otherwise, proceed to the next stage

### Step 3: Completion

After all stages are done:
1. Update the tracker -- all stages DONE
2. Create signal file `.plan_runner_done` with contents `DONE`
3. Report summary: stages completed, commits made, any issues encountered

### Handling Issues

If assumptions turn out wrong during execution:

1. **Stop** -- don't proceed with an invalid plan
2. **Document** -- update the plan's Issues section (expected vs actual)
3. **Revise** -- update affected stages, get user confirmation if the change is significant
4. **Continue** -- execute the revised plan

### Commit Conventions

```
type(scope): brief description

- Specific change 1
- Specific change 2

Co-Authored-By: Claude <model> <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `chore`

### Tips

- **Resumable**: If execution is interrupted, just run again. The tracker persists state -- completed stages won't be re-executed.
- **Incremental plans**: You can add stages while execution is in progress. New stages will be picked up in the next iteration.
- **Stage isolation**: Each stage should leave the codebase in a working state. If a stage would break things mid-way, restructure it.
- **Validation stage**: Consider adding a final "Validation" stage that runs the full test suite to confirm everything works end-to-end.
