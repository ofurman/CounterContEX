#!/bin/bash
###############################################################################
# Push the CounterContEX beam-search worktree to PLGrid.
#
#   bash plgrid/sync-to-plgrid.sh          # sync
#   DRY=1 bash plgrid/sync-to-plgrid.sh    # show what would be transferred
#
# Deliberately syncs the WORKING TREE, not a git ref: the beam-search work lives
# in uncommitted modifications (vectorised prune/select, chunking, new metrics)
# plus the untracked experiments/zeroshot_cf/recompute_metrics.py. A
# `git archive` of the branch would silently ship the wrong code.
#
# Code goes to $HOME/projects (10 GiB quota, backed up). Nothing heavy goes to
# $HOME, and nothing bulky goes to the 250 GB group storage that four people
# share — bulk output lives in personal $SCRATCH, written by the jobs.
#
# Locally-computed results (results/arrays/*.npz for the already-finished Law
# and HELOC-frozen cells) are pushed too, so the cluster run resumes instead of
# recomputing them. Set PUSH_RESULTS=0 to skip that.
###############################################################################
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${PWD}"

# shellcheck source=plg-config.sh
source "${REPO}/plgrid/plg-config.sh"
plg_require PLG_LOGIN PLG_HOST PLG_GROUP PROJECT_NAME

REMOTE="${PLG_LOGIN}@${PLG_HOST}"
RSYNC_OPTS=(-a --human-readable)
[ "${DRY:-0}" = "1" ] && RSYNC_OPTS+=(--dry-run --itemize-changes)

echo "=== resolving group storage on ${PLG_HOST} ==="
HEAVY="$(ssh "${REMOTE}" "
    set -eu
    : \"\${PLG_GROUPS_STORAGE:?PLG_GROUPS_STORAGE unset}\"
    # Site convention in plggcfsgenwro is <group>/<login>/, NOT
    # <group>/users/<login>/: the group root is not writable directly and a
    # users/ dir already exists owned by another member.
    mine=\"\${PLG_GROUPS_STORAGE}/${PLG_GROUP}/\${USER}\"
    root=\"\${mine}/${PROJECT_NAME}\"
    test -d \"\${mine}\" || { echo \"no personal dir at \${mine}\" >&2; exit 1; }
    test -w \"\${mine}\" || { echo \"\${mine} is not writable\" >&2; exit 1; }
    umask 077
    mkdir -p \"\${root}\"/{envs,cache,models,results}
    chmod 700 \"\${root}\"
    printf '%s\n' \"\${root}\"
")"
echo "    ${HEAVY}"

# rsync ignores .gitignore, which is what we want here: the things the
# experiment cannot run without are gitignored and would be absent from any
# git-based transfer.
#   experiments/zeroshot_cf/models/*.ckpt          TabPFN v2 weights, ~70 MiB
#   experiments/zeroshot_cf/vendor/counterfactuals the cel library + its CSVs
#   experiments/zeroshot_cf/recompute_metrics.py   untracked salvage tool
echo ""
echo "=== checking local prerequisites ==="
for required in \
    "experiments/zeroshot_cf/models/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt" \
    "experiments/zeroshot_cf/models/tabpfn-v2-regressor.ckpt" \
    "experiments/zeroshot_cf/vendor/counterfactuals/cel" \
    "experiments/zeroshot_cf/recompute_metrics.py"; do
    if [ ! -e "${REPO}/${required}" ]; then
        echo "MISSING locally: ${required}" >&2
        exit 1
    fi
    echo "    ok  ${required}"
done

# Code provenance. The sync excludes .git, so `git rev-parse` on the cluster
# returns nothing and a cluster-generated array would carry no record of the code
# that produced it. Stamp the working-tree state into a file that travels with the
# sync; exp4_beam_search.py reads it into every npz's config_json.
#
# The -dirty suffix is the honest part: this syncs the WORKING TREE, not a git ref,
# so a clean hash alone would overstate reproducibility if there are uncommitted
# edits. Push the branch before syncing to keep runs attributable.
COMMIT="$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if ! git -C "${REPO}" diff --quiet HEAD 2>/dev/null; then
    COMMIT="${COMMIT}-dirty"
fi
printf '%s\n' "${COMMIT}" > "${REPO}/.plgrid-commit"
echo ""
echo "=== code provenance stamp ==="
echo "    ${COMMIT}"

echo ""
echo "=== code -> \$HOME/projects/${PROJECT_NAME} ==="
ssh "${REMOTE}" "mkdir -p projects/${PROJECT_NAME}/logs"
rsync "${RSYNC_OPTS[@]}" \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '.cache' \
    --exclude '.ruff_cache' \
    --exclude '.pytest_cache' \
    --exclude '.mypy_cache' \
    --exclude '.env' \
    --exclude '.DS_Store' \
    --exclude '__pycache__' \
    --exclude '*.egg-info' \
    --exclude 'site/' \
    --exclude 'logs' \
    --exclude '*.out' \
    --exclude '*.err' \
    --exclude '/experiments/zeroshot_cf/models/.cache' \
    --exclude '/experiments/zeroshot_cf/results/arrays' \
    "${REPO}/" "${REMOTE}:projects/${PROJECT_NAME}/"

# Completed cells, pushed separately so PUSH_RESULTS=0 can skip them and so the
# exclude above keeps them out of the generic code sync.
ARRAYS="${REPO}/experiments/zeroshot_cf/results/arrays"
if [ "${PUSH_RESULTS:-1}" = "1" ] && [ -d "${ARRAYS}" ]; then
    echo ""
    echo "=== completed cells -> group storage (resume state) ==="
    echo "    $(find "${ARRAYS}" -name '*.npz' | wc -l | tr -d ' ') npz, $(du -sh "${ARRAYS}" | cut -f1)"
    ssh "${REMOTE}" "mkdir -p '${HEAVY}/results/arrays'"
    rsync "${RSYNC_OPTS[@]}" "${ARRAYS}/" "${REMOTE}:${HEAVY}/results/arrays/"
fi

if [ "${DRY:-0}" = "1" ]; then
    echo ""
    echo "Dry run only — nothing was written."
    exit 0
fi

echo ""
echo "=== verifying ==="
ssh "${REMOTE}" "
    set -eu
    cd projects/${PROJECT_NAME}
    echo 'entrypoint ' \$(test -f experiments/zeroshot_cf/exp4_beam_search.py && echo present || echo MISSING)
    echo 'recompute  ' \$(test -f experiments/zeroshot_cf/recompute_metrics.py && echo present || echo MISSING)
    echo 'code size  ' \$(du -sh . | cut -f1)
    echo 'home quota ' \$(hpc-fs 2>/dev/null | awk '/HOME/{print \$2\" / \"\$3}')
    echo 'arrays     ' \$(ls '${HEAVY}/results/arrays' 2>/dev/null | wc -l) 'files'
"

cat <<EOF

Synced. Next, on PLGrid:

  ssh helios
  cd projects/${PROJECT_NAME}
  bash plgrid/submit.sh --test-only plgrid/00_setup_env.sbatch
  bash plgrid/submit.sh plgrid/00_setup_env.sbatch
EOF
