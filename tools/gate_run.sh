#!/bin/bash
# Submission gate (specs/output.md) for a finished S6 run in THIS checkout -> submissions/<n>/.
# usage: bash tools/gate_run.sh <n> [reuse|s6only]
#   reuse: S5 reused an earlier run's cross-encoder scores (run 3), so the GPU half of item 5 rebuilds the
#          smallest cross-encoder part per country through S5 itself (same reuse path) instead of
#          tools/check_xenc_determinism.py, which re-scores every band pair of a part from scratch.
#   s6only: only S6 changed since the last full gate of this checkout (S5 checkpoint untouched): item 5 = S6 rerun + md5.
# Items: 1 ids/rows/subset + validator, 3 HOLDOUT (+ per-S1 dump), 4 France sanity, 6 secret scan,
#        5 reproducibility (S6 rerun -> identical md5s; cross-encoder parts bitwise). Logs: logs/gate_<n>.log
set -uo pipefail
cd "$(dirname "$0")/.."
n=$1; mode=${2:-}; D=submissions/$n; PY=.venv/bin/python; W6=work/test/s6_stage2; W5=work/test/s5_infer
exec > >(tee -a logs/gate_$n.log) 2>&1
fail=0; ok() { echo "GATE $1: PASS"; }; bad() { echo "GATE $1: FAIL"; fail=1; }
echo "=== gate $n ($(pwd), $(git rev-parse --short HEAD), $(date -u +%H:%M:%S)) ==="
[ -f $W6/_DONE.json ] || { echo "S6 not done"; exit 1; }
mkdir -p $D
cp output/matching_results.tsv output/candidate_pairs.tsv $D/
cp config.yaml output/artifacts/thresholds.json output/artifacts/thresholds_final.json \
   output/artifacts/thresholds_stage2.json $W6/stage2_report.json $W6/sanity.json $D/
cp logs/timings.tsv $D/timings.tsv
git rev-parse HEAD > $D/git_commit.txt
(cd $D && md5sum matching_results.tsv candidate_pairs.tsv | tee md5.txt)

echo "--- 1. rows, ids, matches ⊆ candidates, validator"
$PY tools/gate_check.py $D && ok "1a" || bad "1a"
$PY data/utils/validate_submission.py --matching $D/matching_results.tsv --candidate $D/candidate_pairs.tsv \
    --test-dir data/dataset/test --check-ids > logs/gate_${n}_validator.log 2>&1
tail -5 logs/gate_${n}_validator.log
grep -q "PASS" logs/gate_${n}_validator.log && ! grep -q "FAIL" logs/gate_${n}_validator.log && ok "1b validator" || bad "1b validator"

echo "--- 3. HOLDOUT (report only)"
$PY tools/compare_holdout.py $W6 --dump $D/holdout_f.parquet || bad "3"

echo "--- 4. sanity by country (France vs US/India)"
$PY -c "import json; [print(r) for r in json.load(open('$D/sanity.json'))]"

echo "--- 6. secret scan (TSVs, code, docs, config)"
if grep -rIlE 'hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|PRIVATE KEY|aws_secret_access_key|sk-[A-Za-z0-9_-]{20,}' \
     $D src/ code/ docs/ config*.yaml 2>/dev/null; then bad "6"; else ok "6"; fi

echo "--- 5. reproducibility"
# one GPU job at a time: another checkout's gate or run may hold the GPU (up to 60 min)
for i in $(seq 120); do
  [ "$mode" = s6only ] && break
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break
  [ $i -eq 1 ] && echo "waiting for the GPU to be free ($(date -u +%T))"; sleep 30
done
if [ "$mode" = s6only ]; then
  # only S6 changed since this checkout's last full gate: S5's checkpoint (cross-encoder parts) is the gated one
  echo "s6only: S5 checkpoint unchanged (S5 _DONE written $(date -u -r $W5/_DONE.json '+%F %T')); cross-encoder check not repeated"
elif [ "$mode" = reuse ]; then
  parts=$(ls -S $W5/pred-part-*.parquet | awk -F/ '{print $NF}' | sed 's/^pred-//' \
          | awk -F- '{c=$2; last[c]=$0} END {for (c in last) print last[c]}')
  mkdir -p $D/xenc_ref
  for p in $parts; do cp $W5/xenc-$p $D/xenc_ref/ && rm $W5/xenc-$p; done
  rm -f $W5/_DONE.json
  if $PY src/s5_infer.py --split test > logs/gate_${n}_s5.log 2>&1; then
    for p in $parts; do
      $PY -c "import polars as pl,sys; a=pl.read_parquet('$D/xenc_ref/xenc-$p'); b=pl.read_parquet('$W5/xenc-$p'); sys.exit(0 if a.equals(b) else 1)" \
        && echo "xenc $p: bitwise identical" || { echo "xenc $p: DIFFERS"; bad "5 xenc"; }
    done
    rm -rf $D/xenc_ref
  else
    bad "5 S5 rerun"; tail -2 logs/gate_${n}_s5.log
    for p in $parts; do [ -f $W5/xenc-$p ] || cp $D/xenc_ref/xenc-$p $W5/; done
    echo "reference cross-encoder parts restored from $D/xenc_ref; S5 _DONE is gone: rerun S5 + S6 before using this checkout"
  fi
else
  $PY tools/check_xenc_determinism.py > logs/gate_${n}_xenc.log 2>&1 && ok "5 xenc" || bad "5 xenc"
  tail -4 logs/gate_${n}_xenc.log
fi
rm -f $W6/_DONE.json $W6/p2-*.parquet
# the md5 check only means something if S6 rewrote the TSVs
if $PY src/s6_stage2.py --split test > logs/gate_${n}_s6.log 2>&1; then
  (cd output && md5sum matching_results.tsv candidate_pairs.tsv) | diff - $D/md5.txt && ok "5 md5" || bad "5 md5"
else
  bad "5 S6 rerun"; tail -2 logs/gate_${n}_s6.log
fi

if [ -n "${BER_S3_BUCKET:-}" ]; then
  aws s3 cp $D/matching_results.tsv "s3://$BER_S3_BUCKET/${RUN_ID:-v1}/submissions/$n/" --only-show-errors \
    && aws s3 cp $D/candidate_pairs.tsv "s3://$BER_S3_BUCKET/${RUN_ID:-v1}/submissions/$n/" --only-show-errors \
    && echo "S3: s3://$BER_S3_BUCKET/${RUN_ID:-v1}/submissions/$n/" || echo "S3 upload failed (files are in $D)"
fi
[ $fail -eq 0 ] && echo "=== GATE $n: ALL PASS ($(date -u +%H:%M:%S)) ===" || echo "=== GATE $n: FAILED ==="
exit $fail
