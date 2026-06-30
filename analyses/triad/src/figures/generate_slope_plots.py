#!/usr/bin/env python3
"""
Category 5 — Turning: θ-error ↔ angular-velocity slope figures (triad courtship).

How the focal fly's turning relates to its orientation error to the pursued
target: signed angular velocity painted over the target θ–θ field-of-view space
(turn maps), the θ-error vs angular-velocity scatter with a per-panel linear fit
(in aggregate and binned by focal speed), the per-acquisition regression slope
across assays, and how that slope varies with focal/target speed.

Runs on the 6-way 2M/1M split (plus the MMF-2M female-target subcase); the
all-frames turn map uses the full 4-way assays.

Usage:
    python -m analyses.triad.src.figures.generate_slope_plots <rootdir>

    rootdir must contain processed_mats/ (parquet) and will get figures/ written.
Run with the flytracker env.
"""

import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analyses.triad.src import putil as tputil
from analyses.triad.src.figures.plotting_helpers import (
    ACTION_COL, FOCAL_FLIES, SEX_MAP, _vel_tag, _vel_label,
    add_female_target_subcases, load_and_prepare,
)
import libs.plotting as putil

ASSAY_COLORS = putil.ASSAY_TYPE_COLORS   # shared Ruta-lab courtship scheme

# Focal-fly speed bins (mm/s) for the velocity-binned θ→ang-vel scatter.
# Each (lo, hi); None = open bound. One figure per bin.
VEL_BINS = [(None, 10), (10, None)]


# ── Plot sections ─────────────────────────────────────────────────────────────

def _plot_turn_maps(assay_dfs, split_assay_dfs, figdir):
    """Signed fly angular velocity over the target θ–θ space.

    All-frames uses the full 4-way assays; the courtship version splits MMF into
    2M/1M (the split is courtship-defined, so it's only valid for courtship frames).
    """
    print("\n── Signed fly angular velocity over the target θ–θ space ──")
    fig, _ = tputil.plot_target_turn_maps(
        assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES,
        courtship_only=False, action_col=ACTION_COL, save_dir=figdir)
    if fig is not None:
        plt.close(fig)
    fig, _ = tputil.plot_target_turn_maps(
        split_assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES,
        courtship_only=True, action_col=ACTION_COL, save_dir=figdir)
    if fig is not None:
        plt.close(fig)


def _plot_theta_angvel_scatter(split_assay_dfs, figdir):
    """θ-error to pursued target vs fly angular velocity (courtship), with a per-panel
    linear regression: in aggregate, then binned by the focal fly's own speed."""
    print("\n── θ-error to pursued target vs fly angular velocity (courtship) ──")
    fig, _ = tputil.plot_courtship_theta_vs_angvel_scatter(
        split_assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES,
        action_col=ACTION_COL, assay_colors=ASSAY_COLORS, save_dir=figdir,
        xlim_display=60)
    if fig is not None:
        plt.close(fig)

    # ...the same scatter, binned by the focal fly's own speed (one figure per VEL_BINS bin)
    print("\n── θ-error vs fly angular velocity, binned by focal speed ──")
    for lo, hi in VEL_BINS:
        fig, _ = tputil.plot_courtship_theta_vs_angvel_scatter(
            split_assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES,
            action_col=ACTION_COL, assay_colors=ASSAY_COLORS, save_dir=figdir,
            xlim_display=60, vel_range=(lo, hi), vel_label=_vel_label(lo, hi),
            save_suffix=f'_vel_{_vel_tag(lo, hi)}')
        if fig is not None:
            plt.close(fig)


def _plot_angvel_slope_per_acquisition(split_assay_dfs, figdir):
    """Per-acquisition regression slope (θ-error → angular velocity) across assays."""
    print("\n── Per-acquisition θ→ang-vel slope across assays ──")
    fig, _ = tputil.plot_courtship_angvel_slope_per_acquisition(
        split_assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES,
        action_col=ACTION_COL, assay_colors=ASSAY_COLORS, save_dir=figdir,
        xlim_display=60)
    if fig is not None:
        plt.close(fig)


def _plot_angvel_slope_by_velocity(split_assay_dfs, figdir):
    """How the θ→ang-vel slope changes with the fly's own speed (binned), per species,
    one line per assay. Versions over: binning speed (focal vs pursued target), slope
    mode (pooled across acqs vs per-acq points + mean±SEM), and two θ-error fit windows:
    |θ|≤60° (matches the scatter) and the full ±180°."""
    print("\n── θ→ang-vel slope vs speed bin (focal & target; pooled / per-acquisition) ──")
    for vel_src in ('focal', 'target'):
        for theta_win, suf in [(60, ''), (180, '_theta180')]:
            for slope_mode in ('pooled', 'per_acq'):
                fig, _ = tputil.plot_courtship_angvel_slope_by_velocity(
                    split_assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES,
                    mode=slope_mode, vel_source=vel_src, action_col=ACTION_COL,
                    assay_colors=ASSAY_COLORS, xlim_display=theta_win, min_points=500,
                    save_dir=figdir, save_suffix=suf)
                if fig is not None:
                    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analyses.triad.src.figures.generate_slope_plots <rootdir>")
        sys.exit(1)

    data = load_and_prepare(sys.argv[1])
    if data is None:
        sys.exit(0)

    figdir = data.figdir
    # Slope/turn figures use the augmented split (adds the MMF-2M female-target subcase).
    fov_split_dfs, _ = add_female_target_subcases(data.split_assay_dfs, data.split_ppm)

    _plot_turn_maps(data.assay_dfs, fov_split_dfs, figdir)
    _plot_theta_angvel_scatter(fov_split_dfs, figdir)
    _plot_angvel_slope_per_acquisition(fov_split_dfs, figdir)
    _plot_angvel_slope_by_velocity(fov_split_dfs, figdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
