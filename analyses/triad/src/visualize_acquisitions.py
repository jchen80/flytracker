#!/usr/bin/env python3
"""
Post-processing visualization script. Loads already-saved processed parquets
and generates per-acquisition figures/clips without re-running the pipeline.

Produces per acquisition:
  figures/{acq}/bout_clips/         — courtship bout clips
  figures/{acq}/switch_clips/       — clips around switch events
  figures/{acq}/bout_signals/       — signal timecourse plots per bout

Usage:
    python analyses/triad/src/visualize_acquisitions.py /Volumes/Julie/fb_MMF_MFF_triad_38mm/
"""
import sys
import os
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


def _find_avi(rootdir, acq):
    matches = glob.glob(os.path.join(rootdir, 'raw_videos', acq, '*.avi'))
    if not matches:
        return None
    return matches[0]


def visualize_one(df, avi_path, save_dir, acq, fps):
    '''Run all visualization steps for a single acquisition.'''
    # ── Video clips ──────────────────────────────────────────────────
    print(f"  Extracting bout clips...")
    tputil.extract_bout_clips(df, avi_path, save_dir,
                               mark_flies=True, max_frames=-1, n_clips=2,
                               action_cols=[ACTION_COL])

    switch_col = f'{ACTION_COL}_switch'
    if switch_col in df.columns and (df[switch_col] == 1).any():
        print(f"  Extracting switch clips...")
        tputil.extract_switch_clips(df, avi_path, save_dir, action_col=ACTION_COL)
    else:
        print(f"  No switch events — skipping switch clips.")

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

    switch_col = f'{ACTION_COL}_switch'
    for bout_num in bout_nums:
        has_switch = (
            switch_col in df.columns and
            ((df[boutnum_col] == bout_num) & (df[switch_col] == 1)).any()
        )
        out_dir = switch_signals_dir if has_switch else signals_dir
        if has_switch:
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
        print("Usage: python visualize_acquisitions.py <rootdir>")
        sys.exit(1)

    rootdir = sys.argv[1]
    processedmat_dir = os.path.join(rootdir, 'processed_mats')
    figdir = os.path.join(rootdir, 'figures')

    parquet_files = sorted(glob.glob(os.path.join(processedmat_dir, '*.parquet')))
    if not parquet_files:
        print(f"No parquet files found in {processedmat_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} acquisitions")

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

        # if the save_dir is not empty, skip to avoid overwriting existing outputs (can be re-run after fixing code)
        if os.listdir(save_dir):
            print(f"  WARNING: {save_dir} is not empty — skipping to avoid overwriting existing outputs.")
            continue

        visualize_one(df, avi_path, save_dir, acq, fps)

    print("\nDone.")


if __name__ == "__main__":
    main()
