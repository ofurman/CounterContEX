#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MATRIX="${1:-}"
CAMPAIGN_MODE="${CAMPAIGN_MODE:-dry-run}"
AUTHORIZATION_RECORD="${AUTHORIZATION_RECORD:-}"
RESUME="${RESUME:-0}"

if [[ -z "$MATRIX" || ! -f "$MATRIX" ]]; then
    echo "usage: $0 PATH_TO_MATRIX" >&2
    exit 2
fi
cd "$PROJECT_DIR"

PYTHON=(uv run --locked python)
CLI=("${PYTHON[@]}" -m experiments.zeroshot_cf.cli)
MATRIX_SHA=$("${PYTHON[@]}" - "$MATRIX" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
readarray -t MATRIX_FACTS < <("${PYTHON[@]}" - "$MATRIX" <<'PY'
import sys
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config

config = load_matrix_config(sys.argv[1])
partitions = {run.protocol.factual_partition for run in config.runs}
if len(partitions) != 1:
    raise ValueError("campaign matrix mixes factual partitions")
print(config.execution.output_root)
print("confirmation" if partitions == {"test"} else "pilot")
PY
)
OUTPUT_ROOT="${MATRIX_FACTS[0]}"
REQUIRED_SCOPE="${MATRIX_FACTS[1]}"

echo "matrix=$MATRIX"
echo "matrix_sha256=$MATRIX_SHA"
echo "output_root=$OUTPUT_ROOT"
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
        echo "confirmation requires a separate scope=confirmation authorization" >&2
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

RUN_ARGS=(matrix --config "$MATRIX")
if [[ "$RESUME" == "1" ]]; then
    RUN_ARGS+=(--resume)
fi
STARTED=$(date +%s)
"${CLI[@]}" "${RUN_ARGS[@]}"
FINISHED=$(date +%s)
echo "campaign_elapsed_s=$((FINISHED - STARTED))"

"${CLI[@]}" aggregate --config "$MATRIX"
"${PYTHON[@]}" -m experiments.zeroshot_cf.orchestration.campaign_inventory \
    seal --matrix "$MATRIX"
