#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$(git rev-parse --show-toplevel)}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/zeroshot_cf/results/campaign/launch}
mkdir -p "$RUN_DIR"
MARKER=$RUN_DIR/stage09.DONE
rm -f "$MARKER"
nohup "$PROJECT_DIR/experiments/zeroshot_cf/dgx/run_stage.sh" \
  "$MARKER" \
  experiments/zeroshot_cf/configs/matrices/campaign_e4_confidence.yaml \
  >"$RUN_DIR/stage09.log" 2>&1 &
echo "$!" >"$RUN_DIR/stage09.pid"
