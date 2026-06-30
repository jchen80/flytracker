#!/usr/bin/env python3
"""
One-off patch: rename 'courtship_switch' → 'courtship_auto_switch' in all
processed parquet files under <rootdir>/processed_mats/.

Usage:
    python analyses/triad/src/rename_switch_col.py /Volumes/Julie/fb_MMF_MFF_triad_38mm/
"""
import sys
import os
import glob
import pandas as pd

OLD_COL = 'courtship_switch'
NEW_COL = 'courtship_auto_switch'


def main():
    if len(sys.argv) < 2:
        print("Usage: python rename_switch_col.py <rootdir>")
        sys.exit(1)

    rootdir = sys.argv[1]
    processedmat_dir = os.path.join(rootdir, 'processed_mats')

    parquet_files = sorted(glob.glob(os.path.join(processedmat_dir, '*.parquet')))
    if not parquet_files:
        print(f"No parquet files found in {processedmat_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet files\n")

    renamed = 0
    skipped = 0
    for fp in parquet_files:
        name = os.path.basename(fp)
        df = pd.read_parquet(fp)

        if OLD_COL not in df.columns:
            print(f"  {name}: no '{OLD_COL}' column — skipping")
            skipped += 1
            continue

        if NEW_COL in df.columns:
            print(f"  {name}: '{NEW_COL}' already exists — dropping old '{OLD_COL}'")
            df = df.drop(columns=[OLD_COL])
        else:
            df = df.rename(columns={OLD_COL: NEW_COL})
            print(f"  {name}: renamed '{OLD_COL}' → '{NEW_COL}'")

        df.to_parquet(fp, index=False)
        renamed += 1

    print(f"\nDone. {renamed} file(s) updated, {skipped} skipped.")


if __name__ == "__main__":
    main()
