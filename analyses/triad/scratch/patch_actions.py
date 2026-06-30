#!/usr/bin/env python3
"""
Re-patch action annotations and target assignments in already-processed parquet
files, without re-running flip correction or coordinate transformations.

Reads the updated actions.mat for each acquisition, replaces all action columns
(plus derived _boutnum, _target, _switch columns) in the existing parquet, then
re-runs target assignment exactly as pairwise_transformation_metrics does.

Handles both single-chamber and multi-chamber acquisitions.

Usage:
    python analyses/triad/src/patch_actions.py <rootdir>

    rootdir must contain:
        raw_videos/     -- acquisition directories with updated actions.mat files
        processed_mats/ -- existing .parquet files to update in place
"""
import sys
import os
import glob
import re
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import libs.utils as util
from analyses.triad.src import multi_funcs as mf
from analyses.triad.src import util as tutil

# Match params used in pairwise_transformation_metrics.py
DELTA_ANGLE_DEG       = 30
MIN_COURTSHIP_BOUT_SEC = 2.0

# Actions that get nearest-fly target assignment
NEAREST_TARGET_ACTIONS = ['circling', 'mounting attempt', 'copulation']


def _drop_action_columns(df):
    """Remove all action, boutnum, target, and switch columns from df."""
    drop = [c for c in df.columns if
            c.endswith('_boutnum') or
            c.endswith('_target') or
            c.endswith('_auto_switch') or
            c.endswith('_switch') or          # catches old 'courtship_switch' name
            c in NEAREST_TARGET_ACTIONS + ['courtship', 'switch', 'auto_switch']]
    existing = [c for c in drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


def _assign_targets(df, fps):
    """Re-run target assignment for all action columns present in df."""
    for action in NEAREST_TARGET_ACTIONS:
        if action in df.columns:
            df = tutil.assign_target_nearest(df, action_col=action)

    if 'courtship' in df.columns:
        df = tutil.assign_target_orientation(
            df, action_col='courtship', fps=fps,
            delta_theta_deg=DELTA_ANGLE_DEG,
            min_bout_sec=MIN_COURTSHIP_BOUT_SEC)

    return df


def _patch_parquet(parquet_path, all_actions, ch_flies=None):
    """Replace action columns in one parquet and save in place."""
    df = pd.read_parquet(parquet_path)
    fps = int(df['FPS'].iloc[0]) if 'FPS' in df.columns else 60

    # Filter and remap IDs for multi-chamber
    if ch_flies is not None:
        ch_id_remap = {old_id: new_id for new_id, old_id in enumerate(ch_flies)}
        ch_actions = all_actions[all_actions['id'].isin(ch_flies)].copy()
        if len(ch_actions) == 0:
            print(f"    {os.path.basename(parquet_path)}: no actions for this chamber — skipping.")
            return
        ch_actions['id'] = ch_actions['id'].map(ch_id_remap)
    else:
        ch_actions = all_actions.copy()

    print(f"    {os.path.basename(parquet_path)}: actions present: "
          f"{sorted(ch_actions['action'].unique().tolist())}")

    # Drop all old action-derived columns, re-assign fresh
    df = _drop_action_columns(df)
    df = util.assign_action_frames_to_df(df, ch_actions)
    df = _assign_targets(df, fps)

    df.to_parquet(parquet_path, index=False)
    print(f"    {os.path.basename(parquet_path)}: saved ({len(df)} rows).")


def _patch_dir(parquet_dir, acq, all_actions, acq_dir):
    """Patch all parquets for one acquisition in parquet_dir."""
    single_parquet = os.path.join(parquet_dir, f'{acq}.parquet')
    ch_parquets    = sorted(glob.glob(os.path.join(parquet_dir, f'{acq}_ch*.parquet')))

    if os.path.exists(single_parquet):
        _patch_parquet(single_parquet, all_actions, ch_flies=None)

    elif ch_parquets:
        trk_files = glob.glob(os.path.join(acq_dir, '*', '*-track.mat'))
        if not trk_files:
            print(f"  No track.mat — cannot recover chamber assignments, skipping.")
            return

        print(f"  Multi-chamber ({len(ch_parquets)} parquets) — loading track data...")
        try:
            calib, trk, feat = util.load_flytracker_data(
                acq_dir, calib_is_upstream=False, filter_ori=True)
            chambers = mf.split_by_chamber(trk, feat, calib)
        except Exception as e:
            print(f"  Failed to load tracking data: {e} — skipping.")
            return

        for ch_idx, (_trk_ch, _feat_ch, _calib_ch, ch_flies) in enumerate(chambers):
            parquet_path = os.path.join(parquet_dir, f'{acq}_ch{ch_idx}.parquet')
            if not os.path.exists(parquet_path):
                print(f"  Parquet not found for {acq}_ch{ch_idx} — skipping.")
                continue
            _patch_parquet(parquet_path, all_actions, ch_flies=ch_flies)

    else:
        print(f"  No parquet found for {acq} in {os.path.basename(parquet_dir)} — skipping.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_actions.py <rootdir>")
        sys.exit(1)

    rootdir = sys.argv[1]
    acquisition_parentdir = os.path.join(rootdir, 'raw_videos')
    processedmat_dir      = os.path.join(rootdir, 'processed_mats')
    backup_dir            = os.path.join(rootdir, 'processed_mats_backup')

    acqs = sorted([f for f in os.listdir(acquisition_parentdir) if not f.startswith('.')])
    print(f"Found {len(acqs)} acquisitions in {acquisition_parentdir}")
    if os.path.isdir(backup_dir):
        print(f"Backup dir found — will also patch {backup_dir}")

    for acq in acqs:
        print(f"\n{'='*60}\n{acq}")
        acq_dir = os.path.join(acquisition_parentdir, acq)

        # ── Load actions ──────────────────────────────────────────────
        action_files = glob.glob(os.path.join(acq_dir, acq, '*-actions.mat'))
        if not action_files:
            print(f"  No actions.mat found — skipping.")
            continue

        all_actions = util.ft_actions_to_bout_df(action_files[0])
        if all_actions is None or len(all_actions) == 0:
            print(f"  No valid bouts in actions.mat — skipping.")
            continue

        print(f"  Actions loaded: {sorted(all_actions['action'].unique().tolist())}, "
              f"fly IDs: {sorted(all_actions['id'].unique().tolist())}")

        # ── Patch processed_mats/ ─────────────────────────────────────
        print(f"  [processed_mats]")
        _patch_dir(processedmat_dir, acq, all_actions, acq_dir)

        # ── Patch processed_mats_backup/ if present ───────────────────
        if os.path.isdir(backup_dir):
            print(f"  [processed_mats_backup]")
            _patch_dir(backup_dir, acq, all_actions, acq_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
