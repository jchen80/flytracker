#!/usr/bin/env python3
"""
Category 1 — Pursuit metric figures for a triad courtship experiment.

Distributions / violins / per-acquisition means of the pursuit metrics (focal &
target velocity, distance to target, |θ error|), both in aggregate and binned by
focal/target velocity, plus how the metrics relate to each other (distance vs
|θ error|). Runs on the 6-way 2M/1M split (Dmel/Dyak × {MMF-2M, MMF-1M, MFF}).

Usage:
    python -m analyses.triad.src.figures.generate_pursuit_metric_plots <rootdir>

    rootdir must contain processed_mats/ (parquet) and will get figures/ written.
"""

import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analyses.triad.src import util as tutil
from analyses.triad.src import putil as tputil
from analyses.triad.src.figures.plotting_helpers import (
    ACTION_COL, FOCAL_FLIES, BODY_ADJ_DIST, VEL_SOURCES, DIST_BIN_MM,
    _save, _dist_metrics, _range_tag, _bin_label, build_velocity_bins,
    load_and_prepare, resolve_rootdir_and_theme,
)
import libs.plotting as putil

# Reassigned in main() to the lightened 2M/1M split scheme.
ASSAY_COLORS = putil.ASSAY_TYPE_COLORS


# ── Plot sections ─────────────────────────────────────────────────────────────

def _plot_velocity_distributions(assay_dfs, figdir):
    print("\n── Velocity distributions ──")
    target_assay_dfs = {k: tutil.filter_to_target_pairs(v, action_col=ACTION_COL)
                        for k, v in assay_dfs.items()}

    for metric, title in [('target_vel', 'Target fly velocity during courtship (mm/s)'),
                           ('vel',        'Focal fly velocity during courtship (mm/s)')]:
        fig, _ = tputil.plot_metric_distribution_across_assays(
            target_assay_dfs, metric=metric,
            action_cols=None,
            focal_flies_map=FOCAL_FLIES,
            assay_colors=ASSAY_COLORS,
            bins=100,
            xlim_percentile=80)
        fig.suptitle(title, fontsize=12)
        _save(fig, figdir, f'{metric}_distribution_courtship.png')

        fig, _ = tputil.plot_metric_violin_across_assays(
            target_assay_dfs, metric=metric,
            action_cols=None,
            focal_flies_map=FOCAL_FLIES,
            assay_colors=ASSAY_COLORS)
        fig.suptitle(title, fontsize=12)
        _save(fig, figdir, f'{metric}_violin_courtship.png')

        # per-acquisition view: one point per acquisition (acq = unit of replication)
        fig, _ = tputil.plot_metric_per_acquisition_across_assays(
            target_assay_dfs, metric=metric,
            action_cols=None,
            focal_flies_map=FOCAL_FLIES,
            assay_colors=ASSAY_COLORS)
        fig.suptitle(f'{title} — per acquisition', fontsize=12)
        _save(fig, figdir, f'{metric}_per_acquisition_courtship.png')


def _plot_velocity_bin_distributions(assay_dfs, figdir):
    """One combined figure per metric: columns = velocity bins, assays overlaid.

    Reuses plot_metric_distribution_across_assays (via its ax= argument), drawing
    each velocity bin into its own panel of a single 1×N_bins figure.
    """
    print("\n── Velocity-bin metric distributions ──")
    metrics = _dist_metrics(assay_dfs) + ['abs_theta_error_deg']

    for vel_source in VEL_SOURCES:
        binned = build_velocity_bins(
            assay_dfs, vel_source, lambda lo, hi: f'{_range_tag(lo, hi)} mm/s')
        if not binned:
            print(f"  {vel_source}: no data — skipping.")
            continue

        for metric in metrics:
            if not any(metric in df.columns
                       for _, ad in binned for df in ad.values()):
                continue

            n_panels = len(binned)
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5),
                                     squeeze=False)
            axes = axes[0]
            for ax, (bin_label, bin_assay_dfs) in zip(axes, binned):
                tputil.plot_metric_distribution_across_assays(
                    bin_assay_dfs, metric=metric,
                    action_cols=None,
                    focal_flies_map=FOCAL_FLIES,
                    assay_colors=ASSAY_COLORS,
                    bins=100, xlim_percentile=80,
                    ax=ax)
                ax.set_title(bin_label)
            for ax in axes[1:]:
                ax.set_ylabel('')

            fig.suptitle(
                f'{metric} by {vel_source} velocity bin (courtship)', fontsize=13)
            plt.tight_layout()
            _save(fig, figdir, f'{metric}_by_{vel_source}_vel_bins_distribution.png')


def _plot_metric_distributions(assay_dfs, figdir):
    """Aggregate distance and |θ error| distributions during courtship.

    Binned-histogram + KDE-smoothed distributions, plus a per-acquisition view
    (one point per acquisition) for each distance metric and for |θ error|.
    """
    print("\n── Metric distributions (distance, |θ error|) ──")

    for dist_metric in _dist_metrics(assay_dfs):
        for use_kde in (False, True):   # binned histogram + KDE-smoothed version
            fig, _ = tputil.plot_metric_distribution_across_assays(
                assay_dfs, metric=dist_metric,
                action_cols=[ACTION_COL],
                focal_flies_map=FOCAL_FLIES,
                assay_colors=ASSAY_COLORS,
                save_dir=figdir, include_non_action=False,
                bins=50, target_action_col=ACTION_COL, xlim_percentile=80, kde=use_kde)
            plt.close(fig)

        # per-acquisition view: one point per acquisition (acq = unit of replication)
        fig, _ = tputil.plot_metric_per_acquisition_across_assays(
            assay_dfs, metric=dist_metric,
            action_cols=[ACTION_COL],
            focal_flies_map=FOCAL_FLIES,
            assay_colors=ASSAY_COLORS,
            save_dir=figdir, include_non_action=False,
            target_action_col=ACTION_COL)
        plt.close(fig)

    for use_kde in (False, True):   # binned histogram + KDE-smoothed version
        fig, _ = tputil.plot_metric_distribution_across_assays(
            assay_dfs, metric='abs_theta_error_deg',
            action_cols=[ACTION_COL],
            focal_flies_map=FOCAL_FLIES,
            assay_colors=ASSAY_COLORS,
            save_dir=figdir, include_non_action=False,
            bins=50, target_action_col=ACTION_COL, xlim_percentile=80, kde=use_kde)
        plt.close(fig)

    fig, _ = tputil.plot_metric_per_acquisition_across_assays(
        assay_dfs, metric='abs_theta_error_deg',
        action_cols=[ACTION_COL],
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        save_dir=figdir, include_non_action=False,
        target_action_col=ACTION_COL)
    plt.close(fig)


def _plot_velocity_bin_profiles(assay_dfs, figdir):
    print("\n── Velocity bin profiles ──")
    metrics = ['abs_theta_error_deg'] + _dist_metrics(assay_dfs)

    for vel_source in VEL_SOURCES:
        binned = build_velocity_bins(assay_dfs, vel_source, _bin_label)
        if not binned:
            continue

        for metric in metrics:
            if not any(metric in df.columns
                       for _, ad in binned for df in ad.values()):
                continue
            fig, _ = tputil.plot_metric_by_velocity_bin_across_assays(
                binned, metric=metric,
                focal_flies_map=FOCAL_FLIES,
                assay_colors=ASSAY_COLORS)
            fig.suptitle(
                f'{metric} by {vel_source} velocity bin (courtship)', fontsize=12)
            plt.tight_layout()
            _save(fig, figdir, f'{metric}_by_{vel_source}_vel_bins.png')

            fig, _ = tputil.plot_metric_by_assay_then_velocity_bin(
                binned, metric=metric,
                focal_flies_map=FOCAL_FLIES)
            fig.suptitle(
                f'{metric} by assay type / {vel_source} velocity bin (courtship)', fontsize=12)
            plt.tight_layout()
            _save(fig, figdir, f'{metric}_by_assay_{vel_source}_vel_bins.png')


def _plot_velocity_bin_per_acquisition(assay_dfs, figdir):
    """Per-acquisition metric points, one panel per velocity bin, split by assay.

    Per-acquisition analog of _plot_velocity_bin_profiles: each acquisition is one
    point (its mean over courtship/target frames in that speed bin), so assays can
    be compared as replicates within each focal/target velocity bin.
    """
    print("\n── Velocity-bin per-acquisition metrics ──")
    metrics = ['abs_theta_error_deg'] + _dist_metrics(assay_dfs)

    for vel_source in VEL_SOURCES:
        binned = build_velocity_bins(
            assay_dfs, vel_source, lambda lo, hi: f'{_bin_label(lo, hi)} mm/s')
        if not binned:
            print(f"  {vel_source}: no data — skipping.")
            continue

        for metric in metrics:
            if not any(metric in df.columns
                       for _, ad in binned for df in ad.values()):
                continue
            fig, _ = tputil.plot_metric_per_acquisition_by_velocity_bin_across_assays(
                binned, metric=metric,
                focal_flies_map=FOCAL_FLIES,
                assay_colors=ASSAY_COLORS,
                agg='median')
            fig.suptitle(
                f'Per-acquisition {metric} by {vel_source} velocity bin (courtship)',
                fontsize=12)
            plt.tight_layout()
            _save(fig, figdir,
                  f'{metric}_by_{vel_source}_vel_bins_per_acquisition.png')


def _plot_dist_theta_relationship(assay_dfs, figdir):
    """Relationship between focal–target distance (dist_to_other_body_adj) and orientation
    error (|θ error|) during courtship, per assay type: a per-assay scatter (with a
    binned-mean trend) and an across-assay binned mean±SEM comparison."""
    print("\n── Distance vs |θ error| relationship ──")
    # (a) x = distance, y = |θ error|
    # (b) x = signed θ error, y = distance
    # Fixed DIST_BIN_MM-wide bins when DISTANCE is on the x-axis (round, interpretable);
    # the signed-θ-on-x version keeps equal-count bins (degrees, can be negative).
    pairs = [(BODY_ADJ_DIST, 'abs_theta_error_deg'),
             ('theta_error_deg', BODY_ADJ_DIST)]
    for x_metric, y_metric in pairs:
        bw = DIST_BIN_MM if x_metric == BODY_ADJ_DIST else None
        fig, _ = tputil.plot_metric_xy_scatter_across_assays(
            assay_dfs, x_metric=x_metric, y_metric=y_metric,
            action_col=ACTION_COL, focal_flies_map=FOCAL_FLIES, assay_colors=ASSAY_COLORS,
            bin_width=bw, save_dir=figdir)
        if fig is not None:
            plt.close(fig)
        for style in ('band', 'points'):
            fig, _ = tputil.plot_metric_xy_binned_across_assays(
                assay_dfs, x_metric=x_metric, y_metric=y_metric,
                action_col=ACTION_COL, focal_flies_map=FOCAL_FLIES, assay_colors=ASSAY_COLORS,
                bin_width=bw, style=style, save_dir=figdir)
            if fig is not None:
                plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rootdir = resolve_rootdir_and_theme(
        sys.argv, "Usage: python -m analyses.triad.src.figures.generate_pursuit_metric_plots <rootdir> [--light]")

    data = load_and_prepare(rootdir)
    if data is None:
        sys.exit(0)

    global ASSAY_COLORS
    ASSAY_COLORS = data.assay_colors_split
    split_assay_dfs = data.split_assay_dfs
    figdir = data.figdir

    _plot_velocity_distributions(split_assay_dfs, figdir)
    _plot_velocity_bin_distributions(split_assay_dfs, figdir)
    _plot_metric_distributions(split_assay_dfs, figdir)
    _plot_velocity_bin_profiles(split_assay_dfs, figdir)
    _plot_velocity_bin_per_acquisition(split_assay_dfs, figdir)
    _plot_dist_theta_relationship(split_assay_dfs, figdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
