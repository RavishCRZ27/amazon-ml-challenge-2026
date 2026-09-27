#!/bin/bash

echo "=== PREFLIGHT CHECK ==="
FAILURES=0
WARNINGS=0
PASSES=0

pass() { echo "PASS: $1"; PASSES=$((PASSES+1)); }
warn() { echo "WARN: $1"; WARNINGS=$((WARNINGS+1)); }
fail() { echo "FAIL: $1"; FAILURES=$((FAILURES+1)); }

# Hardware
echo ""
echo "--- Hardware ---"
NPROC=$(nproc)
pass "nproc: $NPROC"

FREE_GB=$(free -g 2>/dev/null | grep Mem | awk '{print $7}')
if [ -z "$FREE_GB" ]; then
    FREE_GB=$(free -g 2>/dev/null | tail -1 | awk '{print $NF}')
fi
if [ "$FREE_GB" -lt 60 ] 2>/dev/null; then
    fail "free: $FREE_GB GB < 60 GB"
else
    pass "free: $FREE_GB GB"
fi

FREE_DISK=$(df -h . 2>/dev/null | tail -1 | awk '{print $4}')
pass "df: $FREE_DISK available"

nvidia-smi &>/dev/null && pass "GPU detected" || warn "GPU not detected"

# Python packages
echo ""
echo "--- Python Packages ---"
PYTHON=".venv/bin/python"
$PYTHON -c "import polars; print('polars:', polars.__version__)" && pass "polars" || fail "polars"
$PYTHON -c "import pyarrow; print('pyarrow:', pyarrow.__version__)" && pass "pyarrow" || fail "pyarrow"
$PYTHON -c "import numpy; print('numpy:', numpy.__version__)" && pass "numpy" || fail "numpy"
$PYTHON -c "import scipy; print('scipy:', scipy.__version__)" && pass "scipy" || fail "scipy"
$PYTHON -c "import rapidfuzz; from rapidfuzz import process; print('rapidfuzz:', rapidfuzz.__version__)" && pass "rapidfuzz" || fail "rapidfuzz"
$PYTHON -c "import sparse_dot_topn as s; from sparse_dot_topn import sp_matmul_topn; print('sparse_dot_topn:', s.__version__)" && pass "sparse_dot_topn" || fail "sparse_dot_topn (sp_matmul_topn)"
$PYTHON -c "import lightgbm; print('lightgbm:', lightgbm.__version__)" && pass "lightgbm" || fail "lightgbm"
$PYTHON -c "import xgboost; print('xgboost:', xgboost.__version__)" && pass "xgboost" || fail "xgboost"
$PYTHON -c "import yaml; print('yaml OK')" && pass "pyyaml" || fail "pyyaml"
$PYTHON src/translit.py &>/dev/null && pass "translit.py self-check" || fail "translit.py"
$PYTHON src/threshold.py --selftest &>/dev/null && pass "threshold.py self-check" || fail "threshold.py self-check"
# optional stages that are enabled must find their local weights and a working GPU stack
XENC=$($PYTHON -c "import yaml; c=yaml.safe_load(open('config.yaml'))['optional']['cross_encoder']; print(c['model_dir'] if c['enabled'] else '')")
if [ -n "$XENC" ]; then
  [ -f "$XENC/config.json" ] && pass "cross_encoder weights: $XENC" || fail "cross_encoder enabled but $XENC/config.json missing"
  $PYTHON -c "import torch, transformers, sklearn; assert torch.cuda.is_available()" &>/dev/null \
    && pass "cross_encoder stack: torch+CUDA, transformers, sklearn" || fail "cross_encoder stack (torch CUDA / transformers / sklearn)"
fi

# Data
echo ""
echo "--- Data ---"
for tsv in data/dataset/train/train_*.tsv data/dataset/test/test_*.tsv; do
    if [ -f "$tsv" ]; then
        COUNT=$(wc -l < "$tsv" | awk '{print $1-1}')
        pass "$(basename $tsv): $COUNT rows"
    else
        fail "missing: $tsv"
    fi
done
[ -f data/utils/validate_submission.py ] && pass "validator exists" || fail "validator missing"

# AWS
echo ""
echo "--- AWS ---"
if [ -n "$BER_S3_BUCKET" ]; then
    if aws sts get-caller-identity &>/dev/null; then
        pass "AWS credentials valid"
        if aws s3 cp - s3://$BER_S3_BUCKET/$RUN_ID/_preflight/test.txt &>/dev/null < <(echo "test"); then
            aws s3 rm s3://$BER_S3_BUCKET/$RUN_ID/_preflight/test.txt &>/dev/null
            pass "S3 bucket accessible"
        else
            fail "S3 bucket not accessible"
        fi
    else
        fail "AWS credentials invalid (run: aws configure)"
    fi
else
    pass "BER_S3_BUCKET not set (S3 sync skipped)"
fi

# Secrets
echo ""
echo "--- Secrets ---"
if grep -rE 'hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|PRIVATE KEY|sk-[A-Za-z0-9_-]{20,}' src/ code/ docs/ config*.yaml 2>/dev/null; then
    fail "secrets detected in code"
else
    pass "no secrets found"
fi

# Git
echo ""
echo "--- Git ---"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    warn "not a git repository (packaged copy): git checks skipped"
else
    if git status --porcelain | grep -q .; then
        warn "uncommitted changes"
    else
        pass "working tree clean"
    fi
    if grep -q "notebooks/" .gitignore 2>/dev/null; then
        pass "notebooks/ is ignored"
    else
        fail "notebooks/ not in .gitignore"
    fi
fi

# Summary
echo ""
echo "=== SUMMARY ==="
echo "PASS: $PASSES, WARN: $WARNINGS, FAIL: $FAILURES"
[ $FAILURES -eq 0 ] && exit 0 || exit 1
