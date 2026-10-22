#!/bin/zsh
# Mutation-test every source module, smallest first. Safe to re-run: modules
# with a result file are skipped. Survives terminal exit when started with
#   nohup scripts/mutation_sweep.sh > ~/.cache/vibebot-mutation/sweep.log 2>&1 &
#
# One-time setup (WORKERS worktrees, each with its own venv, plus coverage):
#   scripts/mutation_sweep.sh --setup <commit>
set -e
REPO=${0:a:h:h}
D=${MUTATION_DIR:-$HOME/.cache/vibebot-mutation}
WORKERS=${WORKERS:-10}
if [[ $1 == --setup ]]; then
  mkdir -p $D/mut
  for i in $(seq 0 $((WORKERS - 1))); do
    [[ -d $D/wt$i ]] || git -C $REPO worktree add -q --detach $D/wt$i ${2:-HEAD}
    (cd $D/wt$i && git checkout -q -f --detach ${2:-HEAD} && uv sync -q)
  done
  (cd $D/wt0 && uv run --no-sync pytest -q -p no:cacheprovider -m "not slow" \
    --cov=llm --cov=nickinmiddle --cov-report=json:$D/cov.json --cov-fail-under=0 plugins)
  exit 0
fi
cd $D
for m in $(cd wt0 && find plugins/*/src -name '*.py' ! -name '__init__.py' -exec wc -l {} + | grep -v ' total$' | sort -n | awk '{print $2}'); do
  out=mut/$(echo $m | sed 's#plugins/[a-z]*/src/##; s#/#__#g; s#\.py$##').json
  [[ -f $out ]] && continue
  python3 $REPO/scripts/mutate.py $m --out $out --cov cov.json --workers $WORKERS
done
echo SWEEPDONE
