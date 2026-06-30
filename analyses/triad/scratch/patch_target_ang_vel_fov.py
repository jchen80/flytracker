#!/usr/bin/env python3
"""
Patch already-processed parquet files to add (or correct) target_ang_vel_fov.

target_ang_vel_fov is the bearing rate due to the target fly's own motion only
(focal fly held fixed):

    target_ang_vel_fov = (dx * v2y - dy * v2x) / (dx^2 + dy^2)

where d = (dx, dy) = target - focal and v2 is the target's lab-frame velocity
(finite-differenced position). It is the exact derivative of abs_ang_between with
the focal fly's velocity set to zero, isolating the target's contribution to its
angular position in the focal fly's field of view. CCW-positive, matching the
abs_ang_between (atan2) convention.

    target_ang_vel_fov_signed = sign(theta_error) * target_ang_vel_fov
        = d|theta_error|/dt due to the target's own motion
        positive = progressive (target motion increases |theta_error|, carrying it
                   away from the heading axis — front-to-back optic flow)
        negative = regressive (target motion decreases |theta_error|)

Also writes deg/s versions of both: target_ang_vel_fov_deg and
target_ang_vel_fov_signed_deg (the rad/s columns above are kept).

Mirrors the calculation in relative_metrics.calculate_theta_error. This recomputes
and OVERWRITES the columns if present (use it to correct files written by an
earlier, wrong-sign version), so it does not skip already-patched files.

Usage:
    python -m analyses.triad.scratch.patch_target_ang_vel_fov <rootdir>

    rootdir must contain processed_mats/. If processed_mats_backup/ also exists
    inside rootdir it is patched as well.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd

REQUIRED_COLS = ('pos_x', 'pos_y', 'pair', 'theta_error', 'frame', 'id')


def _patch_one(fpath):
    """Patch a single parquet file. Returns True if the file was updated."""
    df = pd.read_parquet(fpath)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"    missing columns {missing} — skipping")
        return False

    fps = float(df['FPS'].iloc[0]) if 'FPS' in df.columns else 60.0
    dt = 1.0 / fps

    # Per-fly lab-frame velocity from finite-differenced position
    pos = (df[['frame', 'id', 'pos_x', 'pos_y']]
           .drop_duplicates(['frame', 'id'])
           .sort_values(['id', 'frame']))
    pos['_vx'] = pos.groupby('id')['pos_x'].diff() / dt
    pos['_vy'] = pos.groupby('id')['pos_y'].diff() / dt

    # Each row's partner (the target) from the pair string
    pair_split = df['pair'].str.split('_')
    fly0 = pair_split.str[0].astype(int)
    fly1 = pair_split.str[1].astype(int)
    partner = np.where(fly0 == df['id'].values, fly1, fly0)

    part = pos.rename(columns={'id': '_partner',
                               'pos_x': '_px', 'pos_y': '_py'})

    merged = (df.reset_index()
                .assign(_partner=partner)
                .merge(part, on=['frame', '_partner'], how='left')
                .drop_duplicates(subset='index')
                .set_index('index'))

    dx = merged['_px'] - merged['pos_x']
    dy = merged['_py'] - merged['pos_y']
    tav = (dx * merged['_vy'] - dy * merged['_vx']) / (dx ** 2 + dy ** 2)

    df['target_ang_vel_fov'] = np.nan
    df.loc[merged.index, 'target_ang_vel_fov'] = tav.values
    df['target_ang_vel_fov_signed'] = (
        np.sign(df['theta_error']) * df['target_ang_vel_fov']
    )
    # deg/s versions (rad/s kept above)
    df['target_ang_vel_fov_deg'] = np.rad2deg(df['target_ang_vel_fov'])
    df['target_ang_vel_fov_signed_deg'] = np.rad2deg(df['target_ang_vel_fov_signed'])

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
        print("Usage: python -m analyses.triad.scratch.patch_target_ang_vel_fov <rootdir>")
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