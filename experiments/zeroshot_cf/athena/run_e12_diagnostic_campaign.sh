#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MATRIX="${1:-}"
CAMPAIGN_MODE="${CAMPAIGN_MODE:-dry-run}"
AUTHORIZATION_RECORD="${AUTHORIZATION_RECORD:-}"
RESUME="${RESUME:-0}"

if [[ -z "$MATRIX" || ! -f "$MATRIX" ]]; then
    echo "usage: $0 PATH_TO_E12_MATRIX" >&2
    exit 2
fi
cd "$PROJECT_DIR"

PYTHON=(uv run --locked python)
CLI=("${PYTHON[@]}" -m experiments.zeroshot_cf.cli)
DIAGNOSTIC=("${PYTHON[@]}" -m experiments.zeroshot_cf.diagnostics.proposal_pushforward)
readarray -t MATRIX_FACTS < <("${PYTHON[@]}" - "$MATRIX" <<'PY'
import hashlib
import sys
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config

config = load_matrix_config(sys.argv[1])
print(hashlib.sha256(config.source.read_bytes()).hexdigest())
print(config.execution.output_root)
print(len(config.runs))
partitions = {run.protocol.factual_partition for run in config.runs}
if len(partitions) != 1:
    raise ValueError("diagnostic matrix mixes factual partitions")
print("confirmation" if partitions == {"test"} else "pilot")
for run in config.runs:
    print(run.cell_id)
PY
)
MATRIX_SHA="${MATRIX_FACTS[0]}"
OUTPUT_ROOT="${MATRIX_FACTS[1]}"
CELL_COUNT="${MATRIX_FACTS[2]}"
REQUIRED_SCOPE="${MATRIX_FACTS[3]}"
CELL_IDS=("${MATRIX_FACTS[@]:4}")

echo "matrix=$MATRIX"
echo "matrix_sha256=$MATRIX_SHA"
echo "output_root=$OUTPUT_ROOT"
echo "cells=$CELL_COUNT"
echo "mode=$CAMPAIGN_MODE"

if [[ "$CAMPAIGN_MODE" == "dry-run" ]]; then
    "${CLI[@]}" matrix --config "$MATRIX" --dry-run
    exit 0
fi
if [[ "$CAMPAIGN_MODE" != "execute" ]]; then
    echo "CAMPAIGN_MODE must be dry-run or execute" >&2
    exit 2
fi
if [[ -z "$AUTHORIZATION_RECORD" || ! -f "$AUTHORIZATION_RECORD" ]]; then
    echo "execute mode requires AUTHORIZATION_RECORD" >&2
    exit 2
fi
if ! grep -Fqx "approved_matrix_sha256=$MATRIX_SHA" "$AUTHORIZATION_RECORD"; then
    echo "authorization record does not approve this exact matrix" >&2
    exit 2
fi
if [[ "$REQUIRED_SCOPE" == "confirmation" ]]; then
    if ! grep -Fqx "scope=confirmation" "$AUTHORIZATION_RECORD"; then
        echo "confirmation execution requires scope=confirmation authorization" >&2
        exit 2
    fi
else
    if ! grep -Fqx "scope=pilot" "$AUTHORIZATION_RECORD"; then
        echo "pilot execution requires scope=pilot authorization" >&2
        exit 2
    fi
fi
if [[ -d "$OUTPUT_ROOT" ]] && find "$OUTPUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
    if [[ "$RESUME" != "1" ]]; then
        echo "refusing non-empty output root without RESUME=1: $OUTPUT_ROOT" >&2
        exit 2
    fi
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
"${PYTHON[@]}" -c \
    'from experiments.zeroshot_cf.tabicl_checkpoints import require_checkpoints; print("verified_checkpoints=" + ",".join(map(str, require_checkpoints())))'
mkdir -p "$OUTPUT_ROOT"
STARTED=$(date +%s)
for ((index = 0; index < CELL_COUNT; index++)); do
    cell_id="${CELL_IDS[$index]}"
    cell_output="$OUTPUT_ROOT/$cell_id"
    echo "cell_index=$index cell_id=$cell_id status=starting"
    "${DIAGNOSTIC[@]}" run \
        --matrix "$MATRIX" \
        --cell-index "$index" \
        --output "$cell_output"
    echo "cell_index=$index cell_id=$cell_id status=complete"
done
FINISHED=$(date +%s)
echo "campaign_elapsed_s=$((FINISHED - STARTED))"

"${PYTHON[@]}" - "$MATRIX" "$OUTPUT_ROOT" <<'PY'
import sys
from experiments.zeroshot_cf.diagnostics.proposal_pushforward import run_matrix_cell
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config

matrix_path, output_root = sys.argv[1:]
config = load_matrix_config(matrix_path)
expected = {run.cell_id for run in config.runs}
actual = {path.name for path in config.execution.output_root.iterdir() if path.is_dir()}
if actual != expected:
    raise ValueError(
        f"diagnostic cell mismatch: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}"
    )
for index, run in enumerate(config.runs):
    run_matrix_cell(
        config.source,
        index,
        config.execution.output_root / run.cell_id,
    )
print(f"verified {len(expected)} diagnostic cells in {output_root}")
PY
