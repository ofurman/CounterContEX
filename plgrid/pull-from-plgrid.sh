#!/bin/bash
###############################################################################
# Pull generated counterfactual arrays back from PLGrid for local scoring.
#
#   bash plgrid/pull-from-plgrid.sh          # sweep arrays + logs
#   DRY=1 bash plgrid/pull-from-plgrid.sh    # show what would transfer
#   LEGACY=1 bash plgrid/pull-from-plgrid.sh # also pull the untagged Exp-4 arrays
#
# Pulls from the durable MIRROR in group storage, not from $HOME/projects: the
# mirror is what the jobs write to after every config, so it is complete even if
# a job died partway. $SCRATCH is not involved — nothing durable lives there.
#
# Standing rule: generation on PLGrid, scoring local. This is the handoff.
#
# LEGACY is off by default. The four untagged arrays were generated locally on
# MPS and pulling the cluster's copies over them would silently swap the backend
# under results that PROJECT_STATE.md attributes to MPS.
###############################################################################
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${PWD}"

# shellcheck source=plg-config.sh
source "${REPO}/plgrid/plg-config.sh"
plg_require PLG_LOGIN PLG_HOST PLG_GROUP PROJECT_NAME

REMOTE="${PLG_LOGIN}@${PLG_HOST}"
RSYNC_OPTS=(-a --human-readable --partial-dir=.rsync-partial)
[ "${DRY:-0}" = "1" ] && RSYNC_OPTS+=(--dry-run --itemize-changes)

echo "=== resolving mirror on ${PLG_HOST} ==="
MIRROR="$(ssh "${REMOTE}" "
    set -eu
    : \"\${PLG_GROUPS_STORAGE:?PLG_GROUPS_STORAGE unset}\"
    root=\"\${PLG_GROUPS_STORAGE}/${PLG_GROUP}/\${USER}/${PROJECT_NAME}/results\"
    test -d \"\${root}\" || { echo \"no mirror at \${root}\" >&2; exit 1; }
    printf '%s\n' \"\${root}\"
")"
echo "    ${MIRROR}"

LOCAL_ARRAYS="${REPO}/experiments/zeroshot_cf/results/arrays"
mkdir -p "${LOCAL_ARRAYS}/sweep" "${REPO}/experiments/zeroshot_cf/results/sweep"

echo ""
echo "=== sweep arrays -> results/arrays/sweep/ ==="
rsync "${RSYNC_OPTS[@]}" \
    "${REMOTE}:${MIRROR}/arrays/sweep/" "${LOCAL_ARRAYS}/sweep/"

echo ""
echo "=== per-config metrics CSVs + summaries -> results/sweep/ ==="
rsync "${RSYNC_OPTS[@]}" \
    "${REMOTE}:${MIRROR}/sweep/" \
    "${REPO}/experiments/zeroshot_cf/results/sweep/" 2>/dev/null || \
    echo "    (none yet)"

if [ "${LEGACY:-0}" = "1" ]; then
    echo ""
    echo "=== untagged Exp-4 arrays -> results/arrays/ (LEGACY=1) ==="
    echo "    WARNING: overwrites the locally-generated MPS arrays."
    rsync "${RSYNC_OPTS[@]}" --include='exp4_*_cfs.npz' --exclude='*' \
        "${REMOTE}:${MIRROR}/arrays/" "${LOCAL_ARRAYS}/"
fi

if [ "${DRY:-0}" = "1" ]; then
    echo ""
    echo "Dry run only — nothing was written."
    exit 0
fi

echo ""
echo "=== verifying ==="
# Byte counts, not exit status: a truncated npz looks entirely plausible in ls,
# and np.load would fail only at score time, after the transfer looked fine.
n_local=$(find "${LOCAL_ARRAYS}/sweep" -name '*.npz' | wc -l | tr -d ' ')
echo "    ${n_local} sweep npz locally"
ssh "${REMOTE}" "cd '${MIRROR}/arrays/sweep' 2>/dev/null && sha256sum *.npz 2>/dev/null" \
    > /tmp/plg-remote-sums.$$ || true
if [ -s /tmp/plg-remote-sums.$$ ]; then
    bad=0
    while read -r sum name; do
        [ -f "${LOCAL_ARRAYS}/sweep/${name}" ] || { echo "    MISSING ${name}"; bad=1; continue; }
        local_sum=$(shasum -a 256 "${LOCAL_ARRAYS}/sweep/${name}" | cut -d' ' -f1)
        if [ "${local_sum}" != "${sum}" ]; then
            echo "    CHECKSUM MISMATCH ${name}"
            bad=1
        fi
    done < /tmp/plg-remote-sums.$$
    rm -f /tmp/plg-remote-sums.$$
    if [ "${bad}" -eq 0 ]; then
        echo "    all sweep npz match the cluster by SHA-256"
    else
        echo "    VERIFICATION FAILED — re-run the pull" >&2
        exit 1
    fi
fi

cat <<EOF

Pulled. Next, locally:

  uv run python experiments/zeroshot_cf/exp7_sweep_table.py
  uv run python experiments/zeroshot_cf/exp7_report.py
EOF
