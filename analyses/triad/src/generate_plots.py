#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Julie Chen
Date: 2026-05-06

Usage:
    python -m analyses.triad.src.generate_plots <rootdir>

    rootdir must contain:
        raw_videos/       -- source .avi files (used only to list acquisitions)
        processed_mats/   -- .parquet files written by generate_pairwise_transformations
        figures/          -- output directory (created if missing)
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

# fly id → sex per triad/assay type
SEX_MAP = {
    'MMF':  {0: 'M', 1: 'M', 2: 'F'},
    'MFF':  {0: 'M', 1: 'F', 2: 'F'},
    'MMMF': {0: 'M', 1: 'M', 2: 'M', 3: 'F'},
}

PPM_VARIATION_THRESHOLD = 0.05  # 5%

ASSAY_COLORS = {
    'Dmel_MMF': 'dodgerblue',
    'Dmel_MFF': 'tomato',
    'Dyak_MMF': 'limegreen',
    'Dyak_MFF': 'orange',
}

PLOT_STYLE = 'dark'
MIN_FONT_SIZE = 12
putil.set_sns_style(style=PLOT_STYLE, min_fontsize=MIN_FONT_SIZE)
BG_COLOR = [0.7] * 3 if PLOT_STYLE == 'dark' else 'k'

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analyses.triad.src.generate_plots <rootdir>")
        sys.exit(1)

    rootdir = sys.argv[1]

    acquisition_parentdir = os.path.join(rootdir, 'raw_videos')
    acqs = sorted([f for f in os.listdir(acquisition_parentdir) if not f.startswith('.')])
    print(f"Found {len(acqs)} acquisitions")

    processedmat_dir = os.path.join(rootdir, 'processed_mats')
    figdir = os.path.join(rootdir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    # Load data
    combined_df = data_io.load_all_processed_dfs(processedmat_dir, verbose=False)
    combined_df['abs_theta_error_deg'] = combined_df['theta_error_deg'].abs()
    assay_dfs = data_io.get_assay_dfs(combined_df)

    FPS = int(combined_df['FPS'].iloc[0])
    print(f"FPS: {FPS}")

    # build ppm_dict per assay type
    ppm_dict = {}
    for assay_type, assay_df in assay_dfs.items():
        ppm = tutil.get_assay_ppm(assay_df, assay_type, threshold=PPM_VARIATION_THRESHOLD)
        if ppm is not None:
            ppm_dict[assay_type] = ppm

    fig, axes = tputil.plot_relative_position_density_across_assays(
        assay_dfs, ppm_dict,
        action_cols=['courtship'],
        focal_flies_map=FOCAL_FLIES,
        save_dir=figdir, include_non_action=True, target_action_col="courtship")

    fig, axes = tputil.plot_metric_distribution_across_assays(
        assay_dfs, metric='abs_theta_error_deg',
        action_cols=['courtship'],
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        save_dir=figdir, include_non_action=False, bins=200, 
        target_action_col='courtship', xlim_percentile=80)

    fig, axes = tputil.plot_metric_distribution_across_assays(
        assay_dfs, metric='dist_to_other',
        action_cols=['courtship'],
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        save_dir=figdir, include_non_action=False, bins=200, 
        target_action_col='courtship', xlim_percentile=80)

    fig, _ = tputil.plot_action_fraction_across_assays(
        assay_dfs, action_col='courtship',
        assay_colors=ASSAY_COLORS,
        save_dir=figdir)

    fig, _ = tputil.plot_action_rate_across_assays(
        assay_dfs, action_col='mounting attempt', fps=FPS,
        norm_minutes=30,
        assay_colors=ASSAY_COLORS,
        save_dir=figdir)

    fig, _ = tputil.plot_action_rate_across_assays(
        assay_dfs, action_col='circling', fps=FPS,
        norm_minutes=30,
        assay_colors=ASSAY_COLORS,
        save_dir=figdir)

    fig, ax = tputil.plot_target_sex_fraction_across_assays(
        assay_dfs, action_col='courtship',
        sex_map=SEX_MAP,
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        save_dir=figdir)

if __name__ == "__main__":
    main()
