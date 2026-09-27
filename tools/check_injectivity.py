#!/usr/bin/env python3
"""Check GT injectivity: count S2/S3 ids appearing under multiple S1s."""

import sys
from pathlib import Path
from collections import defaultdict

import polars as pl

# Read GT
gt_path = Path("student_resource/dataset/train/train_ground_truth.tsv")
if not gt_path.exists():
    print(f"GT not found at {gt_path}")
    sys.exit(1)

print(f"Reading GT from {gt_path}...")
gt = pl.read_csv(gt_path, separator="\t", infer_schema=False)
print(f"GT shape: {gt.shape}")
print(f"Columns: {gt.columns}")

# GT format: source1_entity_id, matched_entity_ids (mixed S2 and S3)
pool_to_s1 = defaultdict(list)

for row in gt.iter_rows():
    s1_id = row[0]
    matched_ids_str = row[1] or ""

    # Split on comma (mixed S2 and S3)
    matched_ids = [x.strip() for x in matched_ids_str.split(",") if x.strip()]

    for matched_id in matched_ids:
        pool_to_s1[matched_id].append(s1_id)

# Count duplicates
duplicates = {k: v for k, v in pool_to_s1.items() if len(v) > 1}

print(f"\nPool ids appearing under multiple S1s: {len(duplicates)}")
for pool_id, s1_ids in sorted(duplicates.items())[:10]:
    print(f"  {pool_id}: {len(s1_ids)} S1s")
if len(duplicates) > 10:
    print(f"  ... and {len(duplicates) - 10} more")

# Verdict
if len(duplicates) > 0:
    print(f"\n⚠ GT is NOT injective on S2/S3 side ({len(duplicates)} duplicates).")
    print(f"Set injective: off in config.yaml")
else:
    print(f"\n✓ GT is injective on S2/S3 side.")
    print(f"Set injective: auto in config.yaml")
