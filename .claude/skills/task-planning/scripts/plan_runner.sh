#!/usr/bin/env bash
# Generic plan runner — executes implementation plans stage by stage using
# Claude Code CLI or OpenAI Codex CLI.
#
# Repeatedly invokes the chosen agent, each time asking it to:
#   1. Read the plan and progress tracker
#   2. Find the next incomplete stage
#   3. Implement that stage
#   4. Update the tracker, run verification, and commit
#
# The loop terminates when:
#   - All stages are DONE (signal file contains "DONE")
#   - A stage is BLOCKED (signal file contains "BLOCKED: <reason>")
#   - Max stage count is reached
#   - The agent exits with an error
#
# Usage:
#   plan_runner.sh --plan <path-to-plan.md> [options]
#
# Options:
#   --plan PATH          Path to the plan markdown file (required)
#   --agent AGENT        Agent to use: "claude" (default) or "codex"
#   --model MODEL        Model to use (default: opus for claude, o4-mini for codex)
#   --max-turns N        Max agentic turns per stage (default: 200, claude only)
#   --max-stages N       Max stages to execute before stopping (default: 20)
#   --signal-file PATH   Signal file path (default: .plan_runner_done)
#   --test-command CMD   Test command to run after each stage (default: auto-detect)
#   --dry-run            Print prompts without executing
#   --verbose            Pass --verbose to claude CLI
#
# Prerequisites:
#   - claude or codex CLI on PATH (depending on --agent)
#   - Working git checkout on an implementation branch

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
PLAN_FILE=""
AGENT="${AGENT:-claude}"
MODEL=""
MAX_TURNS=200
MAX_STAGES=20
SIGNAL_FILE=""
TEST_COMMAND=""
DRY_RUN=false
VERBOSE="--verbose"

# ── Argument parsing ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan)          PLAN_FILE="$2"; shift 2 ;;
        --agent)         AGENT="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --max-turns)     MAX_TURNS="$2"; shift 2 ;;
        --max-stages)    MAX_STAGES="$2"; shift 2 ;;
        --signal-file)   SIGNAL_FILE="$2"; shift 2 ;;
        --test-command)  TEST_COMMAND="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --verbose)       VERBOSE="--verbose"; shift ;;
        -h|--help)
            sed -n '2,/^$/{ s/^# //; s/^#$//; p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# ── Validate agent choice ───────────────────────────────────────────
case "$AGENT" in
    claude|codex) ;;
    *)
        echo "ERROR: --agent must be 'claude' or 'codex', got: $AGENT" >&2
        exit 1
        ;;
esac

# Set default model based on agent if not provided
if [[ -z "$MODEL" ]]; then
    case "$AGENT" in
        claude) MODEL="opus" ;;
        codex)  MODEL="o4-mini" ;;
    esac
fi

# Verify the chosen CLI is available
if ! command -v "$AGENT" &>/dev/null; then
    echo "ERROR: '$AGENT' CLI not found on PATH" >&2
    exit 1
fi

if [[ -z "$PLAN_FILE" ]]; then
    echo "ERROR: --plan is required" >&2
    echo "Usage: plan_runner.sh --plan <path-to-plan.md> [options]" >&2
    exit 1
fi

if [[ ! -f "$PLAN_FILE" ]]; then
    echo "ERROR: Plan file not found: $PLAN_FILE" >&2
    exit 1
fi

# ── Derived configuration ────────────────────────────────────────────
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PLAN_FILE="$(cd "$(dirname "$PLAN_FILE")" && pwd)/$(basename "$PLAN_FILE")"
PLAN_REL="${PLAN_FILE#"$ROOT_DIR/"}"

if [[ -z "$SIGNAL_FILE" ]]; then
    SIGNAL_FILE="$ROOT_DIR/.plan_runner_done"
fi

LOG_DIR="$ROOT_DIR/logs/plan_runner"
mkdir -p "$LOG_DIR"

# Auto-detect test command if not provided
if [[ -z "$TEST_COMMAND" ]]; then
    if [[ -f "$ROOT_DIR/pyproject.toml" ]]; then
        TEST_COMMAND="poetry run pytest"
    elif [[ -f "$ROOT_DIR/package.json" ]]; then
        TEST_COMMAND="npm test"
    elif [[ -f "$ROOT_DIR/Cargo.toml" ]]; then
        TEST_COMMAND="cargo test"
    elif [[ -f "$ROOT_DIR/go.mod" ]]; then
        TEST_COMMAND="go test ./..."
    else
        TEST_COMMAND=""
    fi
fi

# ── Helpers ──────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_agent() {
    local stage_num="$1"
    local prompt="$2"
    local log_file="$LOG_DIR/stage_${stage_num}_$(date '+%Y%m%d_%H%M%S').log"

    log "Stage $stage_num ($AGENT) — log: $log_file"

    if $DRY_RUN; then
        log "[DRY-RUN] Agent: $AGENT | Model: $MODEL"
        log "[DRY-RUN] Prompt (first 300 chars):"
        log "  ${prompt:0:300}..."
        return 0
    fi

    local exit_code=0
    case "$AGENT" in
        claude)
            claude -p "$prompt" \
                --dangerously-skip-permissions \
                --model "$MODEL" \
                --max-turns "$MAX_TURNS" \
                $VERBOSE \
                2>&1 | tee "$log_file"
            exit_code=${PIPESTATUS[0]}
            ;;
        codex)
            codex exec \
                --model "$MODEL" \
                --full-auto \
                -s danger-full-access \
                "$prompt" \
                2>&1 | tee "$log_file"
            exit_code=${PIPESTATUS[0]}
            ;;
    esac

    if [[ $exit_code -ne 0 ]]; then
        log "WARNING: $AGENT exited with code $exit_code"
        log "Check log: $log_file"
        return $exit_code
    fi

    log "Stage $stage_num completed"
    return 0
}

check_signal_file() {
    if [[ -f "$SIGNAL_FILE" ]]; then
        log "Signal file found: $SIGNAL_FILE"
        return 0
    fi
    return 1
}

# ── Stage prompt ─────────────────────────────────────────────────────
# Each invocation gets the same generic prompt. It reads the plan,
# finds the next incomplete stage, implements it, and updates the tracker.
# This avoids any plan-specific logic in bash.

build_stage_prompt() {
    local test_cmd_section=""
    if [[ -n "$TEST_COMMAND" ]]; then
        test_cmd_section="
- After implementing, run the project test suite: \`$TEST_COMMAND\`
- If tests fail, fix the issues before proceeding."
    fi

    cat <<PROMPT_END
You are executing an implementation plan stage by stage.

## Plan Location
The plan is at: $PLAN_REL

## Instructions

1. **Read the plan file** at \`$PLAN_REL\` completely.
2. **Find the Automation Progress Tracker** table in the plan.
   - If no tracker exists, create one at the bottom of the plan using this format:
     | # | Stage | Status | Notes | Updated |
     |---|-------|--------|-------|---------|
     (one row per stage found in the plan, all set to PENDING)
3. **Identify the next stage** whose status is PENDING (not DONE, BLOCKED, or SKIPPED).
   - If ALL stages are DONE, create the signal file and exit (see step 8).
   - If a stage is BLOCKED, do NOT skip it — stop and report.
4. **Check dependencies**: all prior stages must be DONE. If not, mark this stage BLOCKED.
5. **Set the stage status to IN_PROGRESS** in the tracker.
6. **Implement the stage**:
   - Read the stage's steps from the plan.
   - Implement each step carefully, following the project's coding conventions.
   - Only implement what the current stage specifies. Do not skip ahead.
$test_cmd_section
   - Run any stage-specific verification commands listed in the plan.
7. **Update the progress tracker**:
   - Set the stage status to DONE.
   - Add brief notes about what was implemented.
   - Update "Last stage completed" to this stage's name.
   - Update "Last updated by" to "plan-runner-agent".
8. **Commit your work**:
   - Use the commit message specified in the plan for this stage.
   - Stage all changed files with \`git add -A\`.
   - If pre-commit hooks modify files, re-stage and retry. Do NOT use --no-verify.
9. **Signal file**:
   - If this was the LAST stage (all stages now DONE):
     \`echo 'DONE' > $SIGNAL_FILE\`
   - If you hit an unresolvable blocker:
     \`echo 'BLOCKED: <reason>' > $SIGNAL_FILE\`
   - Otherwise, do NOT create the signal file (the loop will invoke you again for the next stage).

## Rules
- Implement ONLY ONE stage per invocation.
- Do not modify stages that are already DONE.
- Follow the project's existing code style, test patterns, and conventions.
- If the plan references specific files, read them before modifying.
- If something is unclear, mark the stage BLOCKED with a note rather than guessing.
PROMPT_END
}

# ── Main loop ────────────────────────────────────────────────────────
PROMPT="$(build_stage_prompt)"

log "═══════════════════════════════════════════════════════════"
log "Plan Runner — Automated Stage Execution"
log "═══════════════════════════════════════════════════════════"
log "Plan:        $PLAN_REL"
log "Agent:       $AGENT"
log "Signal file: $SIGNAL_FILE"
log "Model:       $MODEL"
[[ "$AGENT" == "claude" ]] && log "Max turns:   $MAX_TURNS"
log "Max stages:  $MAX_STAGES"
log "Test cmd:    ${TEST_COMMAND:-<none>}"
log "Dry run:     $DRY_RUN"
log ""

# Clean up signal file from previous runs
rm -f "$SIGNAL_FILE"

for (( stage_num=1; stage_num<=MAX_STAGES; stage_num++ )); do
    log "═══════════════════════════════════════════════════════════"
    log "Iteration $stage_num / $MAX_STAGES"
    log "═══════════════════════════════════════════════════════════"

    # Check if a previous stage created the signal file
    if check_signal_file; then
        cat "$SIGNAL_FILE"
        break
    fi

    if ! run_agent "$stage_num" "$PROMPT"; then
        log "ERROR: $AGENT failed on iteration $stage_num. Stopping."
        echo "FAILED: iteration $stage_num" > "$SIGNAL_FILE"
        break
    fi

    # Check for signal file after stage
    if check_signal_file; then
        cat "$SIGNAL_FILE"
        break
    fi

    log ""
done

# ── Summary ──────────────────────────────────────────────────────────
log ""
log "═══════════════════════════════════════════════════════════"
if [[ -f "$SIGNAL_FILE" ]]; then
    RESULT=$(cat "$SIGNAL_FILE")
    if [[ "$RESULT" == "DONE" ]]; then
        log "SUCCESS: All plan stages completed!"
    else
        log "STOPPED: $RESULT"
    fi
else
    log "WARNING: Loop ended without signal file (hit max-stages=$MAX_STAGES?)"
    log "Re-run to continue from where it left off."
fi
log "═══════════════════════════════════════════════════════════"
log "Logs: $LOG_DIR"
