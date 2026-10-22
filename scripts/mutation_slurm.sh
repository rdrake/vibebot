#!/bin/bash
# Mutation sweep as a Slurm job on the OTU cluster (gpu05, `scavenge`
# partition: preemptible, so the job requeues and resumes from checkpoints).
#
#   sbatch -c 96 scripts/mutation_slurm.sh RUN SRC_TAR [MODULE...]
#
# RUN       results go to ~/vibebot-mut/runs/RUN/mut/<module>.json
# SRC_TAR   a `git archive` tarball in ~/vibebot-mut/ (the source to mutate)
# MODULE    module paths to sweep; default: every source module, smallest first
#
# Worker worktrees live on the node's local /tmp (SQLite and pytest imports
# over NFS are slow); only results and checkpoints are written to home.
# A `<module>.json.partial` checkpoint in the run dir is resumed.
#SBATCH -p scavenge
#SBATCH --requeue
#SBATCH --mem=256G
#SBATCH -t 08:00:00
#SBATCH -J vibemut
#SBATCH -o /home/rdrake/vibebot-mut/logs/%x-%j.out
set -euo pipefail
# Jobs start with a soft limit of 1024 descriptors; the suite needs more.
ulimit -n "$(ulimit -Hn)"

RUN=$1
SRC=$2
shift 2
HOMEDIR=$HOME/vibebot-mut
OUT=$HOMEDIR/runs/$RUN
WORKERS=$((SLURM_CPUS_PER_TASK - 2))
export MUTATION_DIR=/tmp/vibemut-$SLURM_JOB_ID
export UV_CACHE_DIR=$MUTATION_DIR/uv-cache UV_LINK_MODE=hardlink
mkdir -p "$MUTATION_DIR" "$OUT/mut"
trap 'rm -rf "$MUTATION_DIR"' EXIT
cd "$MUTATION_DIR"

echo "$(date -Is) building $WORKERS worktrees from $SRC on $(hostname)"
mkdir wt0
tar -xf "$HOMEDIR/$SRC" -C wt0
(cd wt0 && uv sync --frozen -q --python 3.14)
seq 1 $((WORKERS - 1)) | xargs -P 24 -I{} bash -c \
  'mkdir wt{} && tar -xf "$0" -C wt{} && cd wt{} && uv sync --frozen -q --python 3.14' "$HOMEDIR/$SRC"
echo "$(date -Is) worktrees ready"

if [[ ! -f $OUT/cov.json ]]; then
  (cd wt0 && .venv/bin/python -m pytest -q -p no:cacheprovider -m "not slow" \
    --cov=llm --cov=nickinmiddle --cov-report=json:"$OUT/cov.json" --cov-fail-under=0 plugins >/dev/null)
  echo "$(date -Is) coverage written"
fi

if [[ $# -eq 0 ]]; then
  set -- $(cd wt0 && find plugins/*/src -name '*.py' ! -name '__init__.py' -exec wc -l {} + \
    | grep -v ' total$' | sort -n | awk '{print $2}')
fi
for m in "$@"; do
  name=$(echo "$m" | sed 's#plugins/[a-z]*/src/##; s#/#__#g; s#\.py$##')
  [[ -f $OUT/mut/$name.json ]] && continue
  # The harness parses the source with ast, so it needs the 3.14 interpreter.
  wt0/.venv/bin/python "$HOMEDIR/mutate.py" "$m" --out "$OUT/mut/$name.json" --cov "$OUT/cov.json" --workers "$WORKERS"
done
echo "$(date -Is) SWEEPDONE"
