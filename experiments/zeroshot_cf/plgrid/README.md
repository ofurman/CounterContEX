# Helios GH200: clean E3

This launcher reuses the existing `countercontex-campaign` environment,
checkpoints, and group-storage layout. From the Helios login node, in this
checkout:

```bash
bash experiments/zeroshot_cf/plgrid/submit.sh --test-only \
  experiments/zeroshot_cf/plgrid/run_e3.sbatch
job=$(bash experiments/zeroshot_cf/plgrid/submit.sh --parsable \
  experiments/zeroshot_cf/plgrid/run_e3.sbatch)
echo "$job"
```

Override a renewed allocation at submission time with `PLG_ACCOUNT` and
`PLG_PARTITION`. Canonical results are stored through the
`experiments/zeroshot_cf/results` symlink under group storage. Slurm logs stay
in the checkout's `logs/` directory, and paired E3 products are written beside
the canonical matrix root in `e3_clean_backend_analysis/`.

Submission is not completion. Verify the job state and exit code, inspect both
logs, and require all 36 `COMPLETE` markers plus the aggregate and four E3
analysis products before interpreting results:

```bash
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed
```
