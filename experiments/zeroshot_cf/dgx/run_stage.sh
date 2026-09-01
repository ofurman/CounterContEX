#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 MARKER MATRIX [MATRIX ...]" >&2
    exit 2
fi

MARKER=$1
shift
PROJECT_DIR=${PROJECT_DIR:-$(git rev-parse --show-toplevel)}
cd "$PROJECT_DIR"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

for config in "$@"; do
    uv run python -m experiments.zeroshot_cf.cli matrix \
        --config "$config" --resume
    uv run python -m experiments.zeroshot_cf.cli aggregate \
        --config "$config"
done

touch "$MARKER"
