#!/usr/bin/env python3
"""
Patch already-processed parquet files to add dist_to_other_body_adj.

Reads each parquet, self-joins on (acquisition, pair, frame) to pair each fly
row with the other fly's ellipse parameters, computes the body-adjusted
distance, and writes the result back to the same parquet file.

Skips files that already have the column or are missing required columns
(major_axis_len / minor_axis_len).

Usage:
    python -m analyses.triad.src.patch_body_adj_dist <rootdir>

    rootdir must contain processed_mats/. If processed_mats_backup/ also
    exists inside rootdir it is patched as well.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd

from analyses.triad.src.multi_funcs import compute_dist_body_adj

TARGET_COL    = 'dist_to_other_body_adj'
REQUIRED_COLS = ('pos_x', 'pos_y', 'ori', 'major_axis_len', 'minor_axis_len',
                 'pair', 'frame', 'id', 'acquisition')


def _patch_one(fpath):
    """Patch a single parquet file. Returns True if the file was updated."""
    df = pd.read_parquet(fpath)

    if TARGET_COL in df.columns:
        print(f"    already patched — skipping")
        return False

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"    missing columns {missing} — skipping")
        return False

    ppm = float(df['PPM'].iloc[0]) if 'PPM' in df.columns else None
    if ppm is None:
        print(f"    no PPM column — skipping")
        return False

    # Self-join: for each row find the other fly in the same (acquisition, pair, frame)
    key_cols   = ['acquisition', 'pair', 'frame']
    ellip_cols = ['pos_x', 'pos_y', 'ori', 'major_axis_len', 'minor_axis_len']

    other = (df[key_cols + ['id'] + ellip_cols]
             .rename(columns={
                 'id':             'o_id',
                 'pos_x':          'o_pos_x',
                 'pos_y':          'o_pos_y',
                 'ori':            'o_ori',
                 'major_axis_len': 'o_major_ax',
                 'minor_axis_len': 'o_minor_ax',
             }))

    merged = (df.reset_index()                      # preserve original integer index as column
                .merge(other, on=key_cols, how='left')
                .query('id != o_id'))

    # In rare multi-match cases (should not happen) keep the first match per row
    merged = merged.drop_duplicates(subset='index')

    d_adj = compute_dist_body_adj(
        merged['pos_x'].values,     merged['pos_y'].values,
        merged['ori'].values,       merged['major_axis_len'].values,
        merged['minor_axis_len'].values,
        merged['o_pos_x'].values,   merged['o_pos_y'].values,
        merged['o_ori'].values,     merged['o_major_ax'].values,
        merged['o_minor_ax'].values,
        pix_per_mm=ppm,
    )

    merged = merged.set_index('index')
    df[TARGET_COL] = np.nan
    df.loc[merged.index, TARGET_COL] = d_adj

    df.to_parquet(fpath, engine='pyarrow', compression='snappy')
    return True


def _patch_dir(processed_dir):
    parquet_files = sorted(glob.glob(os.path.join(processed_dir, '*.parquet')))
    if not parquet_files:
        print(f"  No parquet files found in {processed_dir}")
        return

    updated = 0
    for fp in parquet_files:
        acq = os.path.splitext(os.path.basename(fp))[0]
        print(f"  {acq}")
        if _patch_one(fp):
            updated += 1

    print(f"  Updated {updated}/{len(parquet_files)} files in {processed_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analyses.triad.src.patch_body_adj_dist <rootdir>")
        sys.exit(1)

    rootdir = sys.argv[1]

    dirs_to_patch = []
    for name in ('processed_mats', 'processed_mats_backup'):
        d = os.path.join(rootdir, name)
        if os.path.isdir(d):
            dirs_to_patch.append(d)

    if not dirs_to_patch:
        print(f"No processed_mats/ directory found under {rootdir}")
        sys.exit(1)

    for d in dirs_to_patch:
        print(f"\nPatching {d} ...")
        _patch_dir(d)

    print("\nDone.")


if __name__ == "__main__":
    main()