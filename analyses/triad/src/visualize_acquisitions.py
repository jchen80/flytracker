#!/usr/bin/env python3
"""
Post-processing visualization script. Loads already-saved processed parquets
and generates per-acquisition figures/clips without re-running the pipeline.

Produces per acquisition:
  figures/{acq}/bout_clips/         — courtship bout clips
  figures/{acq}/switch_clips/       — clips around switch events
  figures/{acq}/bout_signals/       — signal timecourse plots per bout

Usage:
    python analyses/triad/src/visualize_acquisitions.py <rootdir>
    python analyses/triad/src/visualize_acquisitions.py <rootdir> <acquisition>
"""
import sys
import os
import re
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import libs.plotting as putil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from analyses.triad.src import putil as tputil

ACTION_COL = 'courtship'

PLOT_STYLE = 'dark'
MIN_FONT_SIZE = 12
putil.set_sns_style(style=PLOT_STYLE, min_fontsize=MIN_FONT_SIZE)
BG_COLOR = [0.7]*3 if PLOT_STYLE == 'dark' else 'k'


def _base_acq(acq):
    """Strip _ch{N} suffix added for multi-chamber parquets."""
    return re.sub(r'_ch\d+$', '', acq)


def _find_avi(rootdir, acq):
    # For multi-chamber parquets (e.g. my_acq_ch0) the video lives under the
    # base acquisition directory (my_acq/), not the chamber-suffixed one.
    base = _base_acq(acq)
    matches = glob.glob(os.path.join(rootdir, 'raw_videos', base, '*.avi'))
    if not matches:
        return None
    return matches[0]


def visualize_one(df, avi_path, save_dir, acq, fps):
    '''Run all visualization steps for a single acquisition.'''
    # ── Video clips ──────────────────────────────────────────────────
    print(f"  Extracting bout clips...")
    #tputil.extract_bout_clips(df, avi_path, save_dir,
    #                           mark_flies=True, max_frames=-1, n_clips=1,
    #                           action_cols=[ACTION_COL])

    has_manual_switch = (
        'switching' in df.columns
        and ((df['switching'] == df['id']) & (df['switching'] != -1)).any()
    )
    if has_manual_switch:
        print(f"  Extracting manual switch clips...")
        tputil.extract_switch_clips(df, avi_path, save_dir, action_col=ACTION_COL,
                                    switch_source='manual', slowdown=2)
    else:
        print(f"  No manual switch events — skipping switch clips.")

    # ── Signal timecourse plots per bout ─────────────────────────────
    boutnum_col = f'{ACTION_COL}_boutnum'
    if boutnum_col not in df.columns:
        print(f"  No '{boutnum_col}' column — skipping signal plots.")
        return

    if 'theta_error' not in df.columns:
        print(f"  No 'theta_error' column — skipping signal plots.")
        return

    bout_nums = sorted(df[(df[boutnum_col].notna()) & (df[boutnum_col] != -1)][boutnum_col].unique())
    if not bout_nums:
        print(f"  No bouts found — skipping signal plots.")
        return

    signals_dir = os.path.join(save_dir, 'bout_signals')
    switch_signals_dir = os.path.join(save_dir, 'bout_signals_with_switches')
    os.makedirs(signals_dir, exist_ok=True)
    print(f"  Plotting signals for {len(bout_nums)} bouts...")

    for bout_num in bout_nums:
        bout_mask = df[boutnum_col] == bout_num
        actor_id  = df.loc[bout_mask & (df[ACTION_COL] == df['id']), 'id']
        actor_id  = int(actor_id.iloc[0]) if len(actor_id) > 0 else None

        # route to the switch folder only for MANUAL (annotated) switches
        has_manual_switch = (
            actor_id is not None and
            'switching' in df.columns and
            (bout_mask & (df['switching'] == actor_id)).any()
        )

        out_dir = switch_signals_dir if has_manual_switch else signals_dir
        if has_manual_switch:
            os.makedirs(switch_signals_dir, exist_ok=True)

        fig, _ = tputil.plot_bout_signals(df, bout_num,
                                           action_col=ACTION_COL,
                                           fps=fps,
                                           mark_switches=True,
                                           save_dir=out_dir,
                                           acq=acq)
        if fig is not None:
            plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_acquisitions.py <rootdir> [acquisition]")
        sys.exit(1)

    rootdir = sys.argv[1]
    target_acq = sys.argv[2] if len(sys.argv) >= 3 else None
    processedmat_dir = os.path.join(rootdir, 'processed_mats')
    figdir = os.path.join(rootdir, 'figures')

    if target_acq is not None:
        parquet_files = [os.path.join(processedmat_dir, f'{target_acq}.parquet')]
        if not os.path.exists(parquet_files[0]):
            print(f"No parquet file found for acquisition '{target_acq}'")
            sys.exit(1)
    else:
        parquet_files = sorted(glob.glob(os.path.join(processedmat_dir, '*.parquet')))
        if not parquet_files:
            print(f"No parquet files found in {processedmat_dir}")
            sys.exit(1)

    print(f"Found {len(parquet_files)} acquisition(s)")

    for fp in parquet_files:
        acq = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n{'='*60}\n{acq}")

        df = pd.read_parquet(fp)

        if ACTION_COL not in df.columns:
            print(f"  No '{ACTION_COL}' column — skipping.")
            continue

        avi_path = _find_avi(rootdir, acq)
        if avi_path is None:
            print(f"  No .avi found — skipping.")
            continue

        fps = int(df['FPS'].iloc[0]) if 'FPS' in df.columns else 60

        save_dir = os.path.join(figdir, acq)
        os.makedirs(save_dir, exist_ok=True)

        visualize_one(df, avi_path, save_dir, acq, fps)

    print("\nDone.")


if __name__ == "__main__":
    main()
