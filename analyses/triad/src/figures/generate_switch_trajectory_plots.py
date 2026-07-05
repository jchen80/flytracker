#!/usr/bin/env python3
"""
Category 3a — Switch trajectory figures for MFF triad acquisitions (MFF only).

Everything time-resolved around a target switch: the aggregate new-target
trajectory across assays, the FOV-angular-velocity / θ-error views around the
switch, and per-event sampled ego-/allo-centric trajectory figures.

Switch-as-a-point figures (rates, switch-frame metric distributions, old-vs-new
deltas, θ–θ switch maps) live in generate_switch_plots.py.

Usage:
    python -m analyses.triad.src.figures.generate_switch_trajectory_plots <rootdir>
    # run with the flytracker env, e.g.
    #   ~/miniconda3/envs/flytracker/bin/python -m analyses.triad.src.figures.generate_switch_trajectory_plots <rootdir>

    rootdir must contain reviewed_mats/ (or processed_mats/) and gets figures/ written.
"""

import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analyses.triad.src import util as tutil
from analyses.triad.src import putil as tputil
from analyses.triad.src.figures.plotting_helpers import (
    ACTION_COL, FOCAL_FLIES, _save, load_and_prepare, resolve_rootdir_and_theme,
)
import libs.plotting as putil

ASSAY_COLORS = putil.ASSAY_TYPE_COLORS   # shared Ruta-lab courtship scheme

# Half-windows (seconds each side of the switch) for the sampled target-trajectory
# figures. One figure per assay is produced for each value.
SWITCH_TRAJ_HALF_WINDOWS = (1.0, 2.0)


# ── Plot sections ─────────────────────────────────────────────────────────────

def _plot_switch_trajectory(combined_df, ppm_dict, figdir, fps):
    print("\n── New-target trajectory around switch ──")

    traj_df = tutil.get_switch_new_target_trajectories(
        combined_df, action_col=ACTION_COL, fps=fps, window_sec=4.0)
    if len(traj_df) == 0:
        print("  No trajectory data — skipping.")
        return

    traj_assay_dfs = {assay: grp for assay, grp in traj_df.groupby('assay_type')}
    traj_ppm_dict  = {k: v for k, v in ppm_dict.items() if k in traj_assay_dfs}

    n_events = traj_df.drop_duplicates(['acquisition', 'id', 'switch_frame']).shape[0]
    print(f"  {n_events} events across {traj_df['acquisition'].nunique()} acquisitions")

    fig, _ = tputil.plot_switch_trajectory_across_assays(
        traj_assay_dfs, traj_ppm_dict,
        fps=fps,
        focal_flies_map=FOCAL_FLIES,
        assay_colors=ASSAY_COLORS)
    _save(fig, figdir, 'switch_new_target_trajectory.png')

    # New-target FOV angular velocity vs orientation error, rows = assays, at a few times
    if 'new_target_ang_vel_fov_deg' in traj_df.columns and 'new_theta_error_deg' in traj_df.columns:
        fig, _ = tputil.plot_switch_target_ang_vel_fov_vs_theta_error_across_assays(
            traj_assay_dfs,
            t_rel_points=((-0.2, '−0.2 s'), (-0.1, '−0.1 s'), (0.0, 'at switch')),
            focal_flies_map=FOCAL_FLIES, assay_colors=ASSAY_COLORS)
        if fig is not None:
            _save(fig, figdir, 'switch_target_ang_vel_fov_vs_theta_error.png')

    # Old/new positions + vectors colored by target ang. vel. in FOV (3-row figure)
    if 'new_target_ang_vel_fov_signed' in traj_df.columns:
        fig, _ = tputil.plot_switch_positions_colored_by_metric_across_assays(
            traj_assay_dfs, traj_ppm_dict,
            t_rel_points = ((-0.5, '0.5 s before'), (-0.25, '0.25 s before'), (0, 'at switch')),
            focal_flies_map=FOCAL_FLIES, vlim_percentile=90)
        if fig is not None:
            _save(fig, figdir, 'switch_positions_colored_target_ang_vel_fov.png')

        # standalone old→new lines at switch, colored by target ang vel FOV
        fig, _ = tputil.plot_switch_positions_colored_by_metric_across_assays(
            traj_assay_dfs, traj_ppm_dict,
            t_rel_points=((0, 'at switch'),),
            focal_flies_map=FOCAL_FLIES, vlim_percentile=90, single_row=True)
        if fig is not None:
            _save(fig, figdir, 'switch_oldnew_lines_colored.png')

        # Same positions at switch with a short 0.25 s tail, colored by the metric
        fig, _ = tputil.plot_switch_target_tail_colored_by_metric_across_assays(
            traj_assay_dfs, traj_ppm_dict,
            focal_flies_map=FOCAL_FLIES, tail_sec=0.25, vlim_percentile=90)
        if fig is not None:
            _save(fig, figdir, 'switch_target_tail_colored_target_ang_vel_fov.png')


def _plot_switch_target_trajectory_samples(combined_df, ppm_dict, figdir, fps,
                                           n_samples=16,
                                           half_windows=SWITCH_TRAJ_HALF_WINDOWS):
    """Sample switch events per assay and save ego-frame trajectory figures.

    One figure per assay is saved for each half-window in half_windows (seconds
    each side of the switch); the same sampled events are reused across windows.
    """
    print("\n── Sampled target trajectories around switch ──")

    # Compute the widest window once; narrower windows just clip t_rel.
    traj_df = tutil.get_switch_new_target_trajectories(
        combined_df, action_col=ACTION_COL, fps=fps,
        window_sec=2 * max(half_windows))
    if len(traj_df) == 0:
        print("  No trajectory data — skipping.")
        return

    for half_window in half_windows:
        win_df = traj_df[traj_df['t_rel'].between(-half_window, half_window)]
        for assay_type, tdf in win_df.groupby('assay_type'):
            ppm = ppm_dict.get(assay_type, 1)
            focal_flies = FOCAL_FLIES.get(tdf['triad_type'].iloc[0])
            # 'ego' = focal-centred frame, 'allo' = lab frame (shows focal fly too)
            for frame, suffix in (('ego', ''), ('allo', '_allo')):
                # fly-colored trajectory only (time gradient)
                fig, _ = tputil.plot_switch_target_trajectory_samples_for_assay(
                    tdf, ppm, assay_type, frame=frame,
                    new_color=ASSAY_COLORS.get(assay_type, 'tomato'),
                    n_samples=n_samples, fps=fps, focal_flies=focal_flies,
                    half_window_sec=half_window)
                if fig is not None:
                    _save(fig, figdir,
                          f'switch_target_trajectory_samples_{assay_type}'
                          f'{suffix}_pm{half_window:g}s.png')

                # paired: trajectory beside a partner colored by target_ang_vel_fov
                fig, _ = tputil.plot_switch_target_trajectory_pairs_for_assay(
                    tdf, ppm, assay_type, frame=frame,
                    new_color=ASSAY_COLORS.get(assay_type, 'tomato'),
                    n_samples=n_samples, fps=fps, focal_flies=focal_flies,
                    half_window_sec=half_window, vlim_percentile=90)
                if fig is not None:
                    _save(fig, figdir,
                          f'switch_target_trajectory_samples_{assay_type}'
                          f'{suffix}_paired_pm{half_window:g}s.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rootdir = resolve_rootdir_and_theme(
        sys.argv, "Usage: python -m analyses.triad.src.figures.generate_switch_trajectory_plots <rootdir> [--light]")

    data = load_and_prepare(rootdir, triad_filter='MFF')
    if data is None:
        sys.exit(0)

    combined_df = data.combined_df
    ppm_dict = data.ppm_dict
    figdir = data.figdir
    fps = data.fps

    _plot_switch_trajectory(combined_df, ppm_dict, figdir, fps)
    _plot_switch_target_trajectory_samples(combined_df, ppm_dict, figdir, fps, n_samples=28)

    print("\nDone.")


if __name__ == "__main__":
    main()
