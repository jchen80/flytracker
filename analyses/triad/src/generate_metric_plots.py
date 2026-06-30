#!/usr/bin/env python3
"""
Generate all summary figures for a triad courtship experiment.

Usage:
    python -m analyses.triad.src.generate_plots <rootdir>

    rootdir must contain:
        raw_videos/     -- source acquisition directories
        processed_mats/ -- .parquet files from pairwise_transformation_metrics
        figures/        -- output directory (created if missing)
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analyses.triad.src import data_io
from analyses.triad.src import util as tutil
from analyses.triad.src import putil as tputil
import libs.plotting as putil

# ── Constants ────────────────────────────────────────────────────────────────

ACTION_COL = 'courtship'

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

ASSAY_COLORS = putil.ASSAY_TYPE_COLORS   # shared Ruta-lab courtship scheme

PLOT_STYLE = 'courtship'
MIN_FONT_SIZE = 12
putil.set_sns_style(style=PLOT_STYLE, min_fontsize=MIN_FONT_SIZE)

# Velocity range bins: (min_mm_s, max_mm_s), None = open bound
VEL_RANGES = [
    (None, 5),
    (5,    None),
]
VEL_SOURCES = ['focal', 'target']

# Fixed bin width (mm) for the distance axis of the dist-vs-θ relationship plots.
DIST_BIN_MM = 1.0


# ── Helpers ──────────────────────────────────────────────────────────────────

BODY_ADJ_DIST = 'dist_to_other_body_adj'


def _dist_metrics(assay_dfs):
    """Distance metrics present in at least one assay df."""
    metrics = ['dist_to_other']
    if any(BODY_ADJ_DIST in df.columns for df in assay_dfs.values()):
        metrics.append(BODY_ADJ_DIST)
    return metrics


def _save(fig, figdir, name):
    savepath = os.path.join(figdir, name)
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    print(f"  Saved {name}")
    plt.close(fig)


def _range_tag(lo, hi):
    if lo is None:
        return f'lt{hi}'
    if hi is None:
        return f'gt{lo}'
    return f'{lo}to{hi}'


def _bin_label(lo, hi):
    """Human-readable velocity bin label for axis ticks."""
    if lo is None:
        return f'< {hi}'
    if hi is None:
        return f'> {lo}'
    return f'{lo}–{hi}'


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
        binned = []
        for lo, hi in VEL_RANGES:
            bin_assay_dfs = {
                k: tutil.filter_pursuit_frames(
                    v, action_col=ACTION_COL,
                    min_vel_mm_s=lo, max_vel_mm_s=hi,
                    vel_source=vel_source)
                for k, v in assay_dfs.items()
            }
            bin_assay_dfs = {k: v for k, v in bin_assay_dfs.items() if len(v) > 0}
            if bin_assay_dfs:
                binned.append((f'{_range_tag(lo, hi)} mm/s', bin_assay_dfs))

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


def _plot_position_and_metrics(assay_dfs, ppm_dict, figdir):
    print("\n── Position density and metric distributions ──")

    for density_method in ('hist', 'kde'):   # binned + smoothed (KDE) versions
        fig, _ = tputil.plot_relative_position_density_across_assays(
            assay_dfs, ppm_dict,
            action_cols=[ACTION_COL],
            focal_flies_map=FOCAL_FLIES,
            save_dir=figdir, include_non_action=True,
            target_action_col=ACTION_COL, method=density_method)
        plt.close(fig)

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


def _plot_action_rates(assay_dfs, split_assay_dfs, figdir, fps):
    """Action fractions and behavior rates across assays.

    Courtship fraction and target-sex fraction run on the 4-way set (a 2M/1M-subset
    courtship fraction would be degenerate). Behavior *rates* (mounting attempt,
    circling) run on the 2M/1M split set so MMF is broken into MMF-2M and MMF-1M —
    i.e. the rate while both males court vs while one does. ASSAY_COLORS already
    carries the expanded _2M/_1M keys (set in main).
    """
    print("\n── Action fractions and rates ──")

    fig, _ = tputil.plot_action_fraction_across_assays(
        assay_dfs, action_col=ACTION_COL,
        assay_colors=ASSAY_COLORS,
        save_dir=figdir)
    plt.close(fig)

    for action in ['mounting attempt', 'circling']:
        if not any(action in df.columns for df in split_assay_dfs.values()):
            continue
        fig, _ = tputil.plot_action_rate_across_assays(
            split_assay_dfs, action_col=action, fps=fps,
            norm_minutes=30,
            assay_colors=ASSAY_COLORS,
            save_dir=figdir)
        plt.close(fig)

    fig, _ = tputil.plot_target_sex_fraction_across_assays(
        assay_dfs, action_col=ACTION_COL,
        sex_map=SEX_MAP,
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        save_dir=figdir)
    plt.close(fig)


def _plot_velocity_bin_profiles(assay_dfs, figdir):
    print("\n── Velocity bin profiles ──")
    metrics = ['abs_theta_error_deg'] + _dist_metrics(assay_dfs)

    for vel_source in VEL_SOURCES:
        binned = []
        for lo, hi in VEL_RANGES:
            bin_assay_dfs = {
                k: tutil.filter_pursuit_frames(
                    v, action_col=ACTION_COL,
                    min_vel_mm_s=lo, max_vel_mm_s=hi,
                    vel_source=vel_source)
                for k, v in assay_dfs.items()
            }
            bin_assay_dfs = {k: v for k, v in bin_assay_dfs.items() if len(v) > 0}
            if bin_assay_dfs:
                binned.append((_bin_label(lo, hi), bin_assay_dfs))

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
        binned = []
        for lo, hi in VEL_RANGES:
            bin_assay_dfs = {
                k: tutil.filter_pursuit_frames(
                    v, action_col=ACTION_COL,
                    min_vel_mm_s=lo, max_vel_mm_s=hi,
                    vel_source=vel_source)
                for k, v in assay_dfs.items()
            }
            bin_assay_dfs = {k: v for k, v in bin_assay_dfs.items() if len(v) > 0}
            if bin_assay_dfs:
                binned.append((f'{_bin_label(lo, hi)} mm/s', bin_assay_dfs))

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


def _plot_velocity_bin_density_grid(assay_dfs, ppm_dict, figdir, zoom_mm=15):
    """Combined grid: rows = velocity bins, cols = assays; ±{zoom_mm} mm target density."""
    print("\n── Velocity-bin position-density grid ──")

    for vel_source in VEL_SOURCES:
        binned = []
        for lo, hi in VEL_RANGES:
            bin_assay_dfs = {
                k: tutil.filter_pursuit_frames(
                    v, action_col=ACTION_COL,
                    min_vel_mm_s=lo, max_vel_mm_s=hi,
                    vel_source=vel_source)
                for k, v in assay_dfs.items()
            }
            bin_assay_dfs = {k: v for k, v in bin_assay_dfs.items() if len(v) > 0}
            if bin_assay_dfs:
                binned.append((f'{_bin_label(lo, hi)} mm/s', bin_assay_dfs))

        if not binned:
            print(f"  {vel_source}: no data — skipping.")
            continue

        for density_method in ('hist', 'kde'):   # binned + smoothed (KDE) versions
            fig, _ = tputil.plot_relative_position_density_by_velocity_bin_across_assays(
                binned, ppm_dict,
                focal_flies_map=FOCAL_FLIES,
                zoom_mm=zoom_mm, draw_focal_ellipse=True, method=density_method)
            mtag = ' (KDE)' if density_method == 'kde' else ''
            msuf = '_kde' if density_method == 'kde' else ''
            fig.suptitle(
                f'Target position density (focal frame) by {vel_source} speed bin '
                f'(±{zoom_mm} mm){mtag}', fontsize=13)
            plt.tight_layout()
            _save(fig, figdir, f'rel_pos_density_by_{vel_source}_vel_bins_pm{zoom_mm}mm{msuf}.png')


def _sample_example_frames(assay_dfs, rootdir, figdir):
    print("\n── Sample example frames by metric ──")
    for dist_metric in _dist_metrics(assay_dfs):
        tputil.sample_frames_by_metric_across_assays(
            assay_dfs, rootdir, metric=dist_metric, figdir=figdir,
            action_col=ACTION_COL, step=1.0, n_samples=5, metric_range=(0, 10))
    tputil.sample_frames_by_metric_across_assays(
        assay_dfs, rootdir, metric='abs_theta_error_deg', figdir=figdir,
        action_col=ACTION_COL, step=10.0, n_samples=5, metric_range=(0, 45))


def _plot_dist_theta_relationship(split_assay_dfs, figdir):
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
            split_assay_dfs, x_metric=x_metric, y_metric=y_metric,
            action_col=ACTION_COL, focal_flies_map=FOCAL_FLIES, assay_colors=ASSAY_COLORS,
            bin_width=bw, save_dir=figdir)
        if fig is not None:
            plt.close(fig)
        for style in ('band', 'points'):
            fig, _ = tputil.plot_metric_xy_binned_across_assays(
                split_assay_dfs, x_metric=x_metric, y_metric=y_metric,
                action_col=ACTION_COL, focal_flies_map=FOCAL_FLIES, assay_colors=ASSAY_COLORS,
                bin_width=bw, style=style, save_dir=figdir)
            if fig is not None:
                plt.close(fig)


def _plot_courtship_multiplicity(assay_dfs, figdir):
    """MMF only: fraction of courtship frames that are 2-male vs 1-male, per species."""
    print("\n── Courtship multiplicity (2M vs 1M) ──")
    fig, _ = tputil.plot_courtship_multiplicity_fraction(
        assay_dfs, focal_flies_map=FOCAL_FLIES, triad='MMF',
        action_col=ACTION_COL, assay_colors=ASSAY_COLORS, save_dir=figdir)
    if fig is not None:
        plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analyses.triad.src.generate_plots <rootdir>")
        sys.exit(1)

    rootdir = sys.argv[1]
    processedmat_dir = data_io.resolve_data_dir(rootdir)
    figdir = os.path.join(rootdir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    print("Loading data...")
    combined_df = data_io.load_all_processed_dfs(processedmat_dir, verbose=False)
    combined_df['abs_theta_error_deg'] = combined_df['theta_error_deg'].abs()
    assay_dfs = data_io.get_assay_dfs(combined_df)

    FPS = int(combined_df['FPS'].iloc[0])
    print(f"FPS: {FPS}")

    ppm_dict = {}
    for assay_type, assay_df in assay_dfs.items():
        ppm = tutil.get_assay_ppm(assay_df, assay_type, threshold=PPM_VARIATION_THRESHOLD)
        if ppm is not None:
            ppm_dict[assay_type] = ppm

    # Split each MMF assay into 2M (both males courting) and 1M (one male) frame-subsets
    # so the metric/pursuit/position comparisons run on the 6-way set
    # (Dmel/Dyak × {MMF-2M, MMF-1M, MFF}). MFF passes through unchanged.
    split_assay_dfs = tutil.split_assay_dfs_by_courting_count(
        assay_dfs, FOCAL_FLIES, split_triad='MMF', action_col=ACTION_COL)
    split_ppm = tutil.expand_keyed_dict_for_split(ppm_dict, assay_dfs, split_triad='MMF')
    global ASSAY_COLORS
    ASSAY_COLORS = tutil.expand_keyed_dict_for_split(   # 2M = base color, 1M = lighter
        ASSAY_COLORS, assay_dfs, split_triad='MMF',
        transform=lambda c, sub: c if sub == '2M' else putil.lighten(c, 0.45))

    # Distance vs |θ error| relationship during courtship, per assay type.
    _plot_dist_theta_relationship(split_assay_dfs, figdir)

    # TEMP: other plot sections disabled to iterate quickly on the dist-vs-θ plot above.
    # Re-enable these (and _plot_courtship_multiplicity) for a full run.
    # _plot_courtship_multiplicity(assay_dfs, figdir)   # uses the original 4-way MMF assays
    # _plot_velocity_distributions(split_assay_dfs, figdir)
    # _plot_velocity_bin_distributions(split_assay_dfs, figdir)
    # _plot_velocity_bin_density_grid(split_assay_dfs, split_ppm, figdir, zoom_mm=10)
    # _plot_velocity_bin_profiles(split_assay_dfs, figdir)
    # _plot_velocity_bin_per_acquisition(split_assay_dfs, figdir)
    # _plot_position_and_metrics(split_assay_dfs, split_ppm, figdir)
    # Action fractions/rates stay on the 4-way set: the 2M/1M split is itself defined by
    # courtship, so a courtship-fraction within a subset would be degenerate (~1 for 2M).
    # _plot_action_rates(assay_dfs, split_assay_dfs, figdir, FPS)
    # _sample_example_frames(assay_dfs, rootdir, figdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
