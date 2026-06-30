#!/usr/bin/env python3
"""
Example-frame sampling for a triad courtship experiment.

For each pursuit metric (distance to target, |θ error|), sample real video frames
at stepped metric values across assays — a qualitative companion to the
quantitative pursuit-metric distributions in generate_pursuit_metric_plots.py.

Usage:
    python -m analyses.triad.src.figures.generate_frame_sample_plots <rootdir>

    rootdir must contain raw_videos/ + processed_mats/ (parquet) and gets figures/ written.
Run with the flytracker env.
"""

import sys
import matplotlib
matplotlib.use('Agg')

from analyses.triad.src import putil as tputil
from analyses.triad.src.figures.plotting_helpers import (
    ACTION_COL, _dist_metrics, load_and_prepare,
)


def _sample_example_frames(assay_dfs, rootdir, figdir):
    print("\n── Sample example frames by metric ──")
    for dist_metric in _dist_metrics(assay_dfs):
        tputil.sample_frames_by_metric_across_assays(
            assay_dfs, rootdir, metric=dist_metric, figdir=figdir,
            action_col=ACTION_COL, step=1.0, n_samples=5, metric_range=(0, 10))
    tputil.sample_frames_by_metric_across_assays(
        assay_dfs, rootdir, metric='abs_theta_error_deg', figdir=figdir,
        action_col=ACTION_COL, step=10.0, n_samples=5, metric_range=(0, 45))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analyses.triad.src.figures.generate_frame_sample_plots <rootdir>")
        sys.exit(1)

    rootdir = sys.argv[1]
    data = load_and_prepare(rootdir)
    if data is None:
        sys.exit(0)

    _sample_example_frames(data.assay_dfs, rootdir, data.figdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
