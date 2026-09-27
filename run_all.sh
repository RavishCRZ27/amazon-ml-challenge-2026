#!/bin/bash
# Resume by default. --from <stage> | --only <stage> | --force <stage|all> | --limit-s1 N
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
PLAN="s0_prepare:train s1_gt:train s2_block:train s3_features:train s0_prepare:test s2_block:test s3_features:test s4_train:train s4x_xenc:train s5_infer:test s6_stage2:test"
LIMIT=""; FROM=""; ONLY=""; FORCE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --limit-s1) LIMIT="--limit-s1 $2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
mkdir -p logs
bash preflight.sh > logs/preflight.txt 2>&1 || { tail -30 logs/preflight.txt; echo "preflight FAILED"; exit 1; }
started=0; [ -z "$FROM" ] && started=1
for item in $PLAN; do
  st=${item%%:*}; split=${item##*:}
  [ "$st" = "$FROM" ] && started=1
  [ $started -eq 1 ] || continue
  [ -z "$ONLY" ] || [ "$st" = "$ONLY" ] || continue
  f=""; { [ "$FORCE" = all ] || [ "$FORCE" = "$st" ]; } && f="--force"
  echo ">>> $st ($split) $LIMIT $f"
  $PY src/$st.py --split $split $LIMIT $f
done
if [ -n "$LIMIT" ]; then
  echo "validator skipped: --limit-s1 run (slice outputs are in work/eval/s5_*)"
elif [ -f output/matching_results.tsv ] && [ -f output/candidate_pairs.tsv ]; then
  $PY data/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/dataset/test
else
  echo "validator skipped: output TSVs not produced yet (stubs)"
fi
echo "=== DONE ==="
