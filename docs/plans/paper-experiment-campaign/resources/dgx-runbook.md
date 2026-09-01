# DGX runbook — gx10-bdc5

Derived from the `dgx-remote-experiments` memory, **adapted for this repository**. That memory
was written for the older TabPFN-era layout; the differences are marked. Stage 6 verifies this
runbook end-to-end and corrects anything that has drifted — treat it as a starting point, not
a guarantee.

**Host**: `ssh ofurman@gx10-bdc5` — NVIDIA GB10 (Blackwell, aarch64 Ubuntu). GitHub SSH auth
already works. ~660 G free.

## Provisioning

```bash
git clone --branch paper-experiment-campaign git@github.com:ofurman/CounterContEX.git
cd CounterContEX
uv sync --locked          # DIFFERENT from the memory: this repo is a locked uv workspace
uv run python experiments/zeroshot_cf/vendor_setup.py
uv run python experiments/zeroshot_cf/vendor_setup.py --check
```

`vendor_setup.py` replaces the memory's manual CEL clone-and-patch sequence: it pins CEL to
revision `3587f943826f6b087a0d198c8c4aa4373712c7ee`, applies `patches/cel_init.py`, and
verifies the four benchmark datasets are present. Do not pass `--repin`.

Checkpoints: stage once from a network-enabled machine, then copy to
`experiments/zeroshot_cf/models/tabicl/`.

```bash
uv run python -m experiments.zeroshot_cf.tabicl_checkpoints
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.tabicl_smoke_test
```

Never disable checksum or content-identity verification to make checkpoint loading pass.

## Environment

```bash
export HF_HUB_OFFLINE=1
export ZEROSHOT_CF_MODELS_DIR=$PWD/experiments/zeroshot_cf/models
```

Device selection is owned by `methods/countercontex/runtime.py`; pass `--device` through the
CLI rather than setting a device environment variable directly. Verify with
`uv run python -c "import torch; print(torch.cuda.is_available())"`.

`vendor/`, `models/` and `results/` are gitignored, so `git fetch && git reset --hard
origin/paper-experiment-campaign` syncs code without disturbing them.

## Launch pattern

A `claude -p` per-stage runner cannot survive a multi-hour job — it exits and orphans the
child. Launch detached and poll for the marker:

```bash
nohup bash -c '
  cd ~/pwr/CounterContEX
  HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.cli matrix \
    --config experiments/zeroshot_cf/configs/matrices/campaign_e1_main.yaml \
    --resume
  touch ~/e1_main.DONE
' > ~/e1_main.log 2>&1 &
```

Then poll `~/e1_main.DONE` over SSH. `--resume` revalidates the complete manifest against the
freshly resolved identity, so an interrupted campaign restarts safely; it is execution
metadata and does not change `run_id`.

Expect roughly 60 s of CUDA kernel compilation on the first factual of a fresh process, then
warm per-factual costs matching [compute-budget.md](compute-budget.md).

## Retrieving artifacts

```bash
rsync -av ofurman@gx10-bdc5:~/pwr/CounterContEX/experiments/zeroshot_cf/results/campaign/ \
  experiments/zeroshot_cf/results/campaign/
```

Copy the whole run directory including `manifest.json` and `COMPLETE`. A run directory without
its `COMPLETE` marker is an incomplete run and aggregation will reject it — which is the
intended behavior, not a problem to work around.
