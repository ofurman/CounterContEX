# Headless Execution

For unattended multi-stage execution, use the companion bash script. Supports
both Claude Code CLI and OpenAI Codex CLI as backends.

## Usage

```bash
~/.claude/skills/task-planning/scripts/plan_runner.sh \
  --plan path/to/plan.md \
  [--agent claude|codex] \
  [--model MODEL] \
  [--max-turns 200] \
  [--max-stages 10] \
  [--dry-run] \
  [--test-command "poetry run pytest"] \
  [--signal-file .plan_runner_done]
```

The script launches one agent invocation per stage. Each invocation:
1. Reads the plan and progress tracker
2. Identifies the next incomplete stage
3. Implements it, runs verification, updates the tracker, and commits
4. Exits -- the bash loop then checks the signal file and continues

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--plan PATH` | (required) | Path to the plan markdown file |
| `--agent AGENT` | `claude` | Agent backend: `claude` or `codex` |
| `--model MODEL` | `opus` (claude) / `o4-mini` (codex) | Model to use |
| `--max-turns N` | `200` | Max agentic turns per stage (claude only) |
| `--max-stages N` | `20` | Max stages to execute before stopping |
| `--signal-file PATH` | `.plan_runner_done` | Signal file path |
| `--test-command CMD` | auto-detect | Test command to run after each stage |
| `--dry-run` | off | Print prompts without executing |
| `--verbose` | on | Pass --verbose to claude CLI |

## Agent Backends

### Claude (default)

Runs via `claude -p "<prompt>"` with `--dangerously-skip-permissions`.

```bash
plan_runner.sh --plan plan.md --agent claude --model opus
```

### Codex

Runs via `codex exec "<prompt>"` with `--full-auto` (auto-approve + workspace-write sandbox).

```bash
plan_runner.sh --plan plan.md --agent codex --model o4-mini
```

Requires `CODEX_API_KEY` or prior `codex login`. Common Codex models: `o4-mini`, `o3`, `gpt-5.4`.

Environment variable `AGENT` can also set the default:

```bash
AGENT=codex plan_runner.sh --plan plan.md
```

## Test Command Auto-detection

If `--test-command` is not provided, the script detects:
- `pyproject.toml` -> `poetry run pytest`
- `package.json` -> `npm test`
- `Cargo.toml` -> `cargo test`
- `go.mod` -> `go test ./...`

## Signal File Protocol

The script uses a signal file to communicate completion:
- `DONE` -- all stages completed successfully
- `BLOCKED: <reason>` -- a stage could not be completed
- `FAILED: iteration N` -- the agent exited with an error

## Prerequisites

- `claude` or `codex` CLI on PATH (depending on `--agent`)
- Working git checkout on an implementation branch

## Tips

- **Dry run**: Use `--dry-run` to see what would be executed without making changes
- **Resumable**: If interrupted, re-run. The tracker persists state.
- **Logs**: Stored in `logs/plan_runner/` in the project root
- **Mix agents**: You can resume a plan started with one agent using the other, since both follow the same plan format and signal file protocol

See `scripts/plan_runner.sh` for full implementation.
