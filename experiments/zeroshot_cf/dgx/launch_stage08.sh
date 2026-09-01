#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$(git rev-parse --show-toplevel)}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/zeroshot_cf/results/campaign/launch}
mkdir -p "$RUN_DIR"
MARKER=$RUN_DIR/stage08.DONE
rm -f "$MARKER"
nohup "$PROJECT_DIR/experiments/zeroshot_cf/dgx/run_stage.sh" \
  "$MARKER" \
  experiments/zeroshot_cf/configs/matrices/campaign_e2_diverse.yaml \
  experiments/zeroshot_cf/configs/matrices/campaign_e3_backend.yaml \
  >"$RUN_DIR/stage08.log" 2>&1 &
echo "$!" >"$RUN_DIR/stage08.pid"
