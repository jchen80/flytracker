#!/usr/bin/env python3
"""
Category 4 — Relative position density figures for a triad courtship experiment.

Where the target sits relative to the focal fly during courtship: the relative
position density (lab/focal frame) in aggregate and binned by velocity, the
egocentric "two targets on the FOV" densities (pursued vs other), the pursued-vs-
other θ-error joint density, and p(courtship of each target) over the target θ–θ
field-of-view space.

Densities run on the 6-way 2M/1M split (plus the MMF-2M female-target subcase);
the courtship-probability maps use the full 4-way assays (every frame is a valid
p(courtship) denominator).

Usage:
    python -m analyses.triad.src.figures.generate_position_density_plots <rootdir>

    rootdir must contain processed_mats/ (parquet) and will get figures/ written.
Run with the flytracker env.
"""

import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analyses.triad.src import putil as tputil
from analyses.triad.src.figures.plotting_helpers import (
    ACTION_COL, FOCAL_FLIES, SEX_MAP, VEL_SOURCES,
    _save, _bin_label, build_velocity_bins, add_female_target_subcases,
    load_and_prepare,
)


# ── Plot sections ─────────────────────────────────────────────────────────────

def _plot_position_density(assay_dfs, ppm_dict, figdir):
    print("\n── Relative position density ──")
    for density_method in ('hist', 'kde'):   # binned + smoothed (KDE) versions
        fig, _ = tputil.plot_relative_position_density_across_assays(
            assay_dfs, ppm_dict,
            action_cols=[ACTION_COL],
            focal_flies_map=FOCAL_FLIES,
            save_dir=figdir, include_non_action=True,
            target_action_col=ACTION_COL, method=density_method)
        plt.close(fig)


def _plot_velocity_bin_density_grid(assay_dfs, ppm_dict, figdir, zoom_mm=15):
    """Combined grid: rows = velocity bins, cols = assays; ±{zoom_mm} mm target density."""
    print("\n── Velocity-bin position-density grid ──")

    for vel_source in VEL_SOURCES:
        binned = build_velocity_bins(
            assay_dfs, vel_source, lambda lo, hi: f'{_bin_label(lo, hi)} mm/s')
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


def _plot_fov_densities(split_assay_dfs, split_ppm, figdir):
    """Egocentric target-on-FOV densities (pursued vs other) and the pursued-vs-other
    θ-error joint density, per scenario."""
    print("\n── Target-on-FOV densities (pursued vs other) ──")
    for method in ('hist', 'kde'):
        fig, _ = tputil.plot_target_fov_density_across_assays(
            split_assay_dfs, split_ppm, focal_flies_map=FOCAL_FLIES,
            method=method, action_col=ACTION_COL, save_dir=figdir)
        if fig is not None:
            plt.close(fig)

    print("\n── Pursued-vs-other θ-error joint densities ──")
    for method in ('hist', 'kde'):
        fig, _ = tputil.plot_target_theta_joint_density_across_assays(
            split_assay_dfs, focal_flies_map=FOCAL_FLIES,
            method=method, action_col=ACTION_COL, save_dir=figdir)
        if fig is not None:
            plt.close(fig)


def _plot_courtship_prob_maps(assay_dfs, figdir):
    """p(courtship of each target) over the target θ–θ space (full 4-way assays)."""
    print("\n── p(courtship of each target) over the target θ–θ space ──")
    fig, _ = tputil.plot_target_courtship_prob_maps(
        assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES,
        action_col=ACTION_COL, save_dir=figdir)
    if fig is not None:
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analyses.triad.src.figures.generate_position_density_plots <rootdir>")
        sys.exit(1)

    data = load_and_prepare(sys.argv[1])
    if data is None:
        sys.exit(0)

    figdir = data.figdir
    # FOV densities use the augmented split (adds the MMF-2M female-target subcase).
    fov_split_dfs, fov_split_ppm = add_female_target_subcases(
        data.split_assay_dfs, data.split_ppm)

    _plot_position_density(data.split_assay_dfs, data.split_ppm, figdir)
    _plot_velocity_bin_density_grid(data.split_assay_dfs, data.split_ppm, figdir, zoom_mm=10)
    _plot_fov_densities(fov_split_dfs, fov_split_ppm, figdir)
    _plot_courtship_prob_maps(data.assay_dfs, figdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
