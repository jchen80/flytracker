#!/usr/bin/env python3
"""
Category 2 — Action rate and timing figures for a triad courtship experiment.

Courtship fraction, target-sex fraction, behavior rates (mounting attempt,
circling), courtship multiplicity (2M vs 1M), and when behaviors happen relative
to cumulative courtship pseudotime, plus target velocity during each behavior.

Behavior *rates*, *timing* and *multiplicity* use the 2M/1M MMF split; courtship
and target-sex *fractions* stay on the 4-way assays (a courtship fraction within
a courtship-defined subset would be degenerate).

Usage:
    python -m analyses.triad.src.figures.generate_action_timing_plots <rootdir>

    rootdir must contain processed_mats/ (parquet) and will get figures/ written.
"""

import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd

from analyses.triad.src import data_io
from analyses.triad.src import util as tutil
from analyses.triad.src import putil as tputil
from analyses.triad.src.figures.plotting_helpers import (
    ACTION_COL, FOCAL_FLIES, SEX_MAP, _save, load_and_prepare,
)
import libs.plotting as putil

TARGET_COLS = [
    'mounting attempt',
    'circling',
]

# Reassigned in main() to the lightened 2M/1M split scheme.
ASSAY_COLORS = putil.ASSAY_TYPE_COLORS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fname(target_col):
    """Convert a target column name to a safe filename stem."""
    return target_col.replace(' ', '_')


# ── Plot sections ─────────────────────────────────────────────────────────────

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


def _plot_courtship_multiplicity(assay_dfs, figdir):
    """MMF only: fraction of courtship frames that are 2-male vs 1-male, per species."""
    print("\n── Courtship multiplicity (2M vs 1M) ──")
    fig, _ = tputil.plot_courtship_multiplicity_fraction(
        assay_dfs, focal_flies_map=FOCAL_FLIES, triad='MMF',
        action_col=ACTION_COL, assay_colors=ASSAY_COLORS, save_dir=figdir)
    if fig is not None:
        plt.close(fig)


def _plot_relative_timing(split_assay_dfs, figdir, target_col):
    print(f"\n── {target_col} relative timing ──")

    # One timing table per (split) assay: MMF is split into MMF-2M / MMF-1M frame
    # subsets, so pseudotime and event onsets are computed within each subset. Running
    # per subset (rather than on one combined df) keeps each acquisition's 2M and 1M
    # frames from being pooled by the per-acquisition grouping inside the helper.
    parts = []
    for df in split_assay_dfs.values():
        t = tutil.get_action_relative_timing(
            df, action_col=ACTION_COL, target_col=target_col,
            focal_flies_map=FOCAL_FLIES)
        if len(t) > 0:
            parts.append(t)

    if not parts:
        print(f"  No '{target_col}' events found — skipping.")
        return

    timing_df = pd.concat(parts, ignore_index=True)
    timing_assay_dfs = {assay: grp for assay, grp in timing_df.groupby('assay_type')}

    print(f"  {len(timing_df)} events across "
          f"{timing_df['acquisition'].nunique()} acquisitions")
    for assay, grp in timing_assay_dfs.items():
        print(f"    {assay}: {len(grp)} events, "
              f"{grp['acquisition'].nunique()} acquisitions")

    fig, _ = tputil.plot_action_relative_timing_across_assays(
        timing_assay_dfs,
        assay_colors=ASSAY_COLORS,
    )
    fig.suptitle(
        f'{target_col.capitalize()} timing relative to {ACTION_COL} pseudotime',
        fontsize=13,
    )
    plt.tight_layout()
    _save(fig, figdir, f'{_fname(target_col)}_relative_{ACTION_COL}_timing.png')


def _plot_target_velocity_during_action(combined_df, assay_dfs, figdir, target_col):
    print(f"\n── Target velocity: courtship vs {target_col} ──")

    if target_col not in combined_df.columns:
        print(f"  '{target_col}' column not found — skipping.")
        return

    target_pair_df = tutil.filter_to_target_pairs(combined_df, action_col=ACTION_COL)

    if 'target_vel' not in target_pair_df.columns:
        print(f"  'target_vel' column missing — skipping.")
        return

    court_assay_dfs = {k: v for k, v in data_io.get_assay_dfs(target_pair_df).items()
                       if k in assay_dfs}
    if not court_assay_dfs:
        print(f"  No assay data — skipping.")
        return

    # Frames where focal fly is performing target_col (within courtship target pairs)
    action_assay_dfs = {
        assay_type: df[df[target_col] != -1]
        for assay_type, df in court_assay_dfs.items()
        if target_col in df.columns and (df[target_col] != -1).any()
    }

    print(f"  {sum(len(v) for v in court_assay_dfs.values())} focal→target pair rows "
          f"across {target_pair_df['acquisition'].nunique()} acquisitions")
    for assay_type, df in action_assay_dfs.items():
        print(f"    {assay_type}: {len(df)} {target_col} rows")

    condition_assay_dfs = {'all courtship': court_assay_dfs, target_col: action_assay_dfs}

    fig, _ = tputil.plot_metric_condition_distribution_across_assays(
        condition_assay_dfs,
        metric='target_vel',
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        bins=50,
        xlim_percentile=95,
    )
    if fig is not None:
        fig.suptitle(f'Target velocity: all courtship vs {target_col} (mm/s)', fontsize=12)
        plt.tight_layout()
        _save(fig, figdir, f'{_fname(target_col)}_target_vel_distribution.png')

    fig, _ = tputil.plot_metric_condition_violin_across_assays(
        condition_assay_dfs,
        metric='target_vel',
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS,
        ylim_percentile=95,
    )
    if fig is not None:
        fig.suptitle(f'Target velocity: all courtship vs {target_col} (mm/s)', fontsize=12)
        plt.tight_layout()
        _save(fig, figdir, f'{_fname(target_col)}_target_vel_violin.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analyses.triad.src.figures.generate_action_timing_plots <rootdir>")
        sys.exit(1)

    data = load_and_prepare(sys.argv[1])
    if data is None:
        sys.exit(0)

    global ASSAY_COLORS
    ASSAY_COLORS = data.assay_colors_split
    assay_dfs = data.assay_dfs
    split_assay_dfs = data.split_assay_dfs
    figdir = data.figdir

    # Action fractions/rates: fractions on the 4-way assays, rates on the split set.
    _plot_action_rates(assay_dfs, split_assay_dfs, figdir, data.fps)
    _plot_courtship_multiplicity(assay_dfs, figdir)   # uses the original 4-way MMF assays

    for target_col in TARGET_COLS:
        _plot_relative_timing(split_assay_dfs, figdir, target_col)
        _plot_target_velocity_during_action(data.combined_df, assay_dfs, figdir, target_col)

    print("\nDone.")


if __name__ == "__main__":
    main()
