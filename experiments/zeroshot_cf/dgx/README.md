# GB10 campaign operations

Run from the repository root on `gx10-bdc5`. Verify the pinned vendor checkout and staged
checkpoints before launching:

```bash
uv sync --locked
uv run python experiments/zeroshot_cf/vendor_setup.py --check
uv run python -m experiments.zeroshot_cf.tabicl_checkpoints
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.tabicl_smoke_test
```

Each `launch_stageNN.sh` starts a detached `nohup` process, writes its PID and log beneath
`results/campaign/launch/`, runs each owned matrix with `--resume`, strictly aggregates it, and
publishes `stageNN.DONE` only after every command succeeds. Stage 11 is the exception: its
launcher publishes `stage11.E9_DONE`, because E8 has no generation matrix and its read-only
rescoring gate must pass before the plan stage itself is complete.

`launch_stage08b.sh` runs the prepared E2b diversity-budget sweep
(`campaign_e2b_budget.yaml`, 36 cells, marker `stage08b.DONE`, log `stage08b.log`). It is not
part of any plan stage and costs an estimated 6 to 9 GPU hours, so launch it only on an explicit
instruction.

```bash
bash experiments/zeroshot_cf/dgx/launch_stage07.sh
tail -f experiments/zeroshot_cf/results/campaign/launch/stage07.log
test -f experiments/zeroshot_cf/results/campaign/launch/stage07.DONE
```

Retrieve canonical artifacts, including COMPLETE markers, with:

```bash
rsync -av ofurman@gx10-bdc5:/home/ofurman/CounterContEX/experiments/zeroshot_cf/results/campaign/ \
  experiments/zeroshot_cf/results/campaign/
```
