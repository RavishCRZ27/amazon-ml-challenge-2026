#!/bin/bash
# Build dist/<team>_submission.zip in the challenge layout (rule: specs/output.md).
# Run only after the submission gate passed for the TSVs in output/.
#   usage: bash make_submission.sh <team_name> [methodology.md]
set -euo pipefail
cd "$(dirname "$0")"
TEAM=${1:?usage: bash make_submission.sh <team_name> [methodology.md]}
DOC=${2:-docs/Documentation_filled.md}
PY=.venv/bin/python
PKG=dist/${TEAM}_submission
C=$PKG/code/business_entity_resolution
[ -f output/matching_results.tsv ] && [ -f output/candidate_pairs.tsv ] && [ -f "$DOC" ] || { echo "missing outputs or $DOC"; exit 1; }
! grep -nE '\[(TEST SANITY|FILL|TODO|TBD)' "$DOC" || { echo "unfilled placeholder in $DOC"; exit 1; }
rm -rf "$PKG" "$PKG.zip"
mkdir -p "$PKG/output" "$C/src"
cp output/matching_results.tsv output/candidate_pairs.tsv "$PKG/output/"
cp src/*.py "$C/src/"
cp run_all.sh preflight.sh code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt "$C/"
# packaged config: a clean run retrains the cross-encoder and scores every band pair itself (init_dir and
# reuse_scores point into work/ or another checkout, which are not shipped)
sed -E 's#init_dir: [^,}]*#init_dir: null#; s#init_any_s4: [^,}]*#init_any_s4: false#; s#reuse_scores: [^,}]*#reuse_scores: null#' \
  config.yaml > "$C/config.yaml"
$PY -c "import yaml,sys; x=yaml.safe_load(open(sys.argv[1]))['optional']['cross_encoder']; assert x.get('init_dir') is None and x.get('reuse_scores') is None and not x.get('init_any_s4'), x; print('packaged config: init_dir / reuse_scores unset')" "$C/config.yaml"
cp "$DOC" "$PKG/Documentation_template.md"
# nothing that must never ship
for bad in notebooks .aws work models logs; do
  [ -z "$(find "$PKG" -path "*/$bad/*" -o -name "$bad" -type d | head -1)" ] || { echo "FORBIDDEN: $bad in package"; exit 1; }
done
# secret scan (outputs + code + doc)
if grep -rEIl 'hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|PRIVATE KEY-----|aws_secret_access_key\s*[=:]|sk-[A-Za-z0-9_-]{20,}' "$PKG"; then
  echo "SECRET PATTERN FOUND"; exit 1
fi
echo "secret scan clean"
# the zipped TSVs are byte-identical to output/
for f in matching_results.tsv candidate_pairs.tsv; do
  [ "$(md5sum < output/$f)" = "$(md5sum < "$PKG/output/$f")" ] || { echo "md5 mismatch $f"; exit 1; }
done
$PY - "$PKG" <<'EOF'
import shutil, sys, zipfile
from pathlib import Path
pkg = Path(sys.argv[1])
z = shutil.make_archive(str(pkg), "zip", root_dir=pkg)   # entries at the zip root, as in the README layout
with zipfile.ZipFile(z) as f:
    names = f.namelist()
    print(f"{z}: {len(names)} entries, {Path(z).stat().st_size / 1e6:.1f} MB zipped, "
          f"{sum(i.file_size for i in f.infolist()) / 1e6:.1f} MB unzipped")
    for n in sorted(names):
        if not n.endswith("/") and "/src/" not in n:
            print("  ", n)
    print(f"   + {sum(1 for n in names if '/src/' in n and n.endswith('.py'))} files under src/")
EOF
