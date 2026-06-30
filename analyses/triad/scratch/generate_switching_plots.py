#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Julie Chen
Date: 2026-05-19

Switching analysis: where is the new courtship target relative to the male
at the moment of a target switch?

Uses filter_to_switch_frames() to isolate the first frame of each new target
run, keeping only the pair row where the partner IS the new target — so
targ_rel_pos_x/y describes exactly where the switched-to fly was at switch time.

Usage:
    python -m analyses.triad.src.switching_plots <rootdir>

    rootdir must contain:
        raw_videos/       -- source acquisition directories (used only to list acqs)
        processed_mats/   -- .parquet files written by pairwise_transformation_metrics
        figures/          -- output parent directory (created if missing)

Outputs are saved to <rootdir>/figures/switching/.
"""

import sys
import os
from matplotlib import pyplot as plt
import pandas as pd
from analyses.triad.src import data_io
from analyses.triad.src import util as tutil
from analyses.triad.src import putil as tputil

import libs.plotting as putil

FOCAL_FLIES = {
    'MMF':  [0, 1],
    'MFF':  [0],
    'MMMF': [0, 1, 2],
}

SEX_MAP = {
    'MMF':  {0: 'M', 1: 'M', 2: 'F'},
    'MFF':  {0: 'M', 1: 'F', 2: 'F'},
    'MMMF': {0: 'M', 1: 'M', 2: 'M', 3: 'F'},
}

PPM_VARIATION_THRESHOLD = 0.05

ASSAY_COLORS = {
    'Dmel_MMF': 'mediumpurple',
    'Dmel_MFF': 'fuchsia',
    'Dyak_MMF': 'gold',
    'Dyak_MFF': 'orangered',
}

PLOT_STYLE = 'dark'
MIN_FONT_SIZE = 12
putil.set_sns_style(style=PLOT_STYLE, min_fontsize=MIN_FONT_SIZE)

ACTION_COL = 'courtship'


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analyses.triad.src.switching_plots <rootdir>")
        sys.exit(1)

    rootdir = sys.argv[1]
    processedmat_dir = os.path.join(rootdir, 'processed_mats')
    figdir = os.path.join(rootdir, 'figures', 'switching')
    os.makedirs(figdir, exist_ok=True)
    print(f"Saving figures to {figdir}")

    # Load all processed data
    combined_df = data_io.load_all_processed_dfs(processedmat_dir, verbose=False)
    if combined_df is None or len(combined_df) == 0:
        print("No processed data found. Exiting.")
        sys.exit(1)

    combined_df['abs_theta_error_deg'] = combined_df['theta_error_deg'].abs()

    FPS = int(combined_df['FPS'].iloc[0])
    print(f"FPS: {FPS}")

    # Compute ppm from full data (reliable across all acquisitions)
    full_assay_dfs = data_io.get_assay_dfs(combined_df)
    ppm_dict = {}
    for assay_type, assay_df in full_assay_dfs.items():
        ppm = tutil.get_assay_ppm(assay_df, assay_type, threshold=PPM_VARIATION_THRESHOLD)
        if ppm is not None:
            ppm_dict[assay_type] = ppm

    # Filter to switch frames — one row per switch event, the pair row where
    # the partner IS the new target (so targ_rel_pos_x/y = new target position)
    print(f"\nFiltering to {ACTION_COL} switch frames...")
    switch_df = tutil.filter_to_switch_frames(combined_df, action_col=ACTION_COL)
    if len(switch_df) == 0:
        print("No switch frames found. Check that switch detection ran correctly.")
        sys.exit(1)

    print(f"  {len(switch_df)} switch-frame rows across "
          f"{switch_df['acquisition'].nunique()} acquisitions")

    switch_df['abs_theta_error_deg'] = switch_df['theta_error_deg'].abs()
    switch_assay_dfs = data_io.get_assay_dfs(switch_df)

    # Only keep ppm entries for assay types present in the switch data
    switch_ppm_dict = {k: v for k, v in ppm_dict.items() if k in switch_assay_dfs}

    def _save(fig, name):
        savepath = os.path.join(figdir, name)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
        plt.close(fig)

    # ── Old vs new target position vectors at switch time ────────────────────
    print("\nComputing switch frame vectors (old and new target positions)...")
    vector_df = tutil.get_switch_frame_vectors(combined_df, action_col=ACTION_COL)
    if len(vector_df) > 0:
        vector_assay_dfs = {assay: grp for assay, grp in vector_df.groupby('assay_type')}
        fig, _ = tputil.plot_switch_vectors_across_assays(
            vector_assay_dfs, switch_ppm_dict,
            focal_flies_map=FOCAL_FLIES)
        _save(fig, 'switch_target_vectors.png')
    else:
        print("  No switch vector data found — skipping vector plot.")

    # ── Relative position density of the new target at switch time ────────────
    # 'all frames' in switch_assay_dfs == all switch events (already pre-filtered)
    print("\nPlotting new-target relative position density at switch...")
    fig, axes = tputil.plot_relative_position_density_across_assays(
        switch_assay_dfs, switch_ppm_dict,
        action_cols=None,
        focal_flies_map=FOCAL_FLIES,
        include_non_action=False)
    fig.suptitle(f'Position of new target at {ACTION_COL} switch', fontsize=13)
    plt.tight_layout()
    _save(fig, 'switch_target_position_density.png')

    # ── Metric histograms at switch frames ────────────────────────────────────
    print("\nPlotting metric distributions at switch frames...")

    fig, _ = tputil.plot_metric_distribution_across_assays(
        switch_assay_dfs, metric='abs_theta_error_deg',
        action_cols=None,
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        bins=150,
        include_non_action=False,
        xlim_percentile=95)
    fig.suptitle(f'Orientation error to new target at {ACTION_COL} switch', fontsize=12)
    plt.tight_layout()
    _save(fig, 'switch_abs_theta_error_deg_distribution.png')

    fig, _ = tputil.plot_metric_distribution_across_assays(
        switch_assay_dfs, metric='dist_to_other',
        action_cols=None,
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        bins=150,
        include_non_action=False,
        xlim_percentile=95)
    fig.suptitle(f'Distance to new target at {ACTION_COL} switch', fontsize=12)
    plt.tight_layout()
    _save(fig, 'switch_dist_to_other_distribution.png')

    # ── Metric violins at switch frames ───────────────────────────────────────
    print("\nPlotting metric violins at switch frames...")

    fig, _ = tputil.plot_metric_violin_across_assays(
        switch_assay_dfs, metric='abs_theta_error_deg',
        action_cols=None,
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        include_non_action=False,
        ylim_percentile=95)
    fig.suptitle(f'Orientation error to new target at {ACTION_COL} switch', fontsize=12)
    plt.tight_layout()
    _save(fig, 'switch_abs_theta_error_deg_violin.png')

    fig, _ = tputil.plot_metric_violin_across_assays(
        switch_assay_dfs, metric='dist_to_other',
        action_cols=None,
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        include_non_action=False,
        ylim_percentile=95)
    fig.suptitle(f'Distance to new target at {ACTION_COL} switch', fontsize=12)
    plt.tight_layout()
    _save(fig, 'switch_dist_to_other_violin.png')

    # ── Switch count and rate across assays ───────────────────────────────────
    print("\nPlotting switch count and rate across assays...")

    fig, ax = tputil.plot_switch_rate_across_assays(
        full_assay_dfs, action_col=ACTION_COL, fps=FPS,
        focal_flies_map=FOCAL_FLIES, norm_minutes=None,
        assay_colors=ASSAY_COLORS)
    if fig is not None:
        _save(fig, 'switch_count_across_assays.png')

    fig, ax = tputil.plot_switch_rate_across_assays(
        full_assay_dfs, action_col=ACTION_COL, fps=FPS,
        focal_flies_map=FOCAL_FLIES, norm_minutes=30,
        assay_colors=ASSAY_COLORS)
    if fig is not None:
        _save(fig, 'switch_rate_per_courtship_across_assays.png')

    # ── Sex of the switched-to target ─────────────────────────────────────────
    print("\nPlotting sex of switched-to target across assays...")
    fig, ax = tputil.plot_target_sex_fraction_across_assays(
        switch_assay_dfs, action_col=ACTION_COL,
        sex_map=SEX_MAP,
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS)
    if fig is not None:
        ax.set_title(f'Sex of switched-to target ({ACTION_COL})')
        plt.tight_layout()
        _save(fig, 'switch_target_sex_fraction.png')


if __name__ == "__main__":
    main()
