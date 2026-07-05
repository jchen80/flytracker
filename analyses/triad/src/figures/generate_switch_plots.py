#!/usr/bin/env python3
"""
Category 3b — Switch figures (switch-as-a-point) for MFF triad acquisitions.

Operates only on MFF triad types (Dmel_MFF, Dyak_MFF, etc.). Covers everything
that treats a target switch as a discrete event rather than a trajectory:
switch rates, switch-frame position/metric distributions, old-vs-new deltas
(overall and split by Δ|θ error| case), and where switches sit in the focal
male's (θ to target x, θ to target y) field-of-view space.

Time-resolved trajectory figures around the switch live in
generate_switch_trajectory_plots.py.

Usage:
    python -m analyses.triad.src.figures.generate_switch_plots <rootdir>
    # run with the flytracker env, e.g.
    #   ~/miniconda3/envs/flytracker/bin/python -m analyses.triad.src.figures.generate_switch_plots <rootdir>

    rootdir must contain reviewed_mats/ (or processed_mats/) and gets figures/ written.

Notable outputs (all MFF only):
    switch_rate_per_courtship_across_assays.png  -- switches per MIN of courtship
    switch_delta_{new_lower,similar,new_higher}.png
        -- per switch-case grid: rows = (distance, |θ error|, signed target
           ang-vel-FOV deg/s) new−old deltas; cols = [KDE histogram | paired
           old→new per assay]  (_plot_switch_metric_deltas_by_case)
    switch_oldnew_lines.png                      -- standalone single-row old→new lines
    switchcase_{case}_target_vectors.png         -- 3-row: old / new / old→new lines
    target_theta_switch_{combined,fovnew,...,distnew,...}.png
        -- switch locations over the target θ–θ space (_plot_switch_theta_maps)
"""

import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analyses.triad.src import data_io
from analyses.triad.src import util as tutil
from analyses.triad.src import putil as tputil
from analyses.triad.src.figures.plotting_helpers import (
    ACTION_COL, FOCAL_FLIES, SEX_MAP, BODY_ADJ_DIST,
    _save, _dist_metrics, load_and_prepare, resolve_rootdir_and_theme,
)
import libs.plotting as putil

ASSAY_COLORS = putil.ASSAY_TYPE_COLORS   # shared Ruta-lab courtship scheme

TRIAD = 'MFF'
SWITCH_SOURCE = 'manual'                  # only manually-annotated switches (MFF)
FOV_METRIC = 'target_ang_vel_fov_signed_deg'
DIST_METRIC = 'dist_to_other_body_adj'

# |theta error| difference (deg) splitting switches into similar / new-higher /
# new-lower cases for the by-case analysis.
SWITCH_THETA_CASE_THRESHOLD_DEG = 15.0


# ── Plot sections ─────────────────────────────────────────────────────────────

def _plot_switch_rates(assay_dfs, figdir, fps):
    print("\n── Switch rates ──")

    fig, _ = tputil.plot_switch_rate_across_assays(
        assay_dfs, action_col=ACTION_COL, fps=fps,
        focal_flies_map=FOCAL_FLIES, norm_minutes=None,
        assay_colors=ASSAY_COLORS)
    if fig is not None:
        _save(fig, figdir, 'switch_count_across_assays.png')

    fig, _ = tputil.plot_switch_rate_across_assays(
        assay_dfs, action_col=ACTION_COL, fps=fps,
        focal_flies_map=FOCAL_FLIES, norm_minutes=1,
        assay_colors=ASSAY_COLORS)
    if fig is not None:
        _save(fig, figdir, 'switch_rate_per_courtship_across_assays.png')

    has_manual = any('switching' in df.columns for df in assay_dfs.values())
    has_auto   = any(f'{ACTION_COL}_auto_switch' in df.columns
                     for df in assay_dfs.values())
    if has_manual:
        fig, _ = tputil.plot_action_rate_across_assays(
            assay_dfs, action_col='switching', fps=fps,
            norm_minutes=1,
            assay_colors=ASSAY_COLORS,
            save_dir=figdir)
        plt.close(fig)
    elif has_auto:
        fig, _ = tputil.plot_switch_rate_across_assays(
            assay_dfs, action_col=ACTION_COL, fps=fps,
            focal_flies_map=FOCAL_FLIES, norm_minutes=1,
            assay_colors=ASSAY_COLORS)
        if fig is not None:
            _save(fig, figdir, 'switch_rate_auto_per_courtship_across_assays.png')


def _plot_switch_positions_and_metrics(combined_df, assay_dfs, ppm_dict, figdir):
    print("\n── Switch positions and metrics ──")

    switch_df = tutil.filter_to_switch_frames(combined_df, action_col=ACTION_COL)
    if len(switch_df) == 0:
        print("  No switch frames found — skipping.")
        return
    switch_df['abs_theta_error_deg'] = switch_df['theta_error_deg'].abs()
    switch_assay_dfs = data_io.get_assay_dfs(switch_df)
    switch_ppm_dict  = {k: v for k, v in ppm_dict.items() if k in switch_assay_dfs}

    print(f"  {len(switch_df)} switch-frame rows across "
          f"{switch_df['acquisition'].nunique()} acquisitions")

    # Switch target position vectors
    vector_df = tutil.get_switch_frame_vectors(combined_df, action_col=ACTION_COL)
    if len(vector_df) > 0:
        vector_assay_dfs = {assay: grp for assay, grp in vector_df.groupby('assay_type')}
        fig, _ = tputil.plot_switch_vectors_across_assays(
            vector_assay_dfs, switch_ppm_dict,
            focal_flies_map=FOCAL_FLIES,
            old_color='dimgray', assay_colors=ASSAY_COLORS)
        _save(fig, figdir, 'switch_target_vectors.png')

        # standalone old→new lines (vanilla: old grey, grey line, new assay-color)
        fig, _ = tputil.plot_switch_vectors_across_assays(
            vector_assay_dfs, switch_ppm_dict, focal_flies_map=FOCAL_FLIES,
            old_color='dimgray', assay_colors=ASSAY_COLORS, vectors_only=True)
        _save(fig, figdir, 'switch_oldnew_lines.png')
    else:
        print("  No switch vector data — skipping vector plots.")

    # New target position density at switch
    fig, _ = tputil.plot_relative_position_density_across_assays(
        switch_assay_dfs, switch_ppm_dict,
        action_cols=None,
        focal_flies_map=FOCAL_FLIES,
        include_non_action=False)
    fig.suptitle(f'Position of new target at {ACTION_COL} switch', fontsize=13)
    plt.tight_layout()
    _save(fig, figdir, 'switch_target_position_density.png')

    # Metric distributions at switch frames
    for metric, fname, title, bins, pct in [
        ('abs_theta_error_deg', 'switch_abs_theta_error_deg_distribution.png',
         f'Orientation error to new target at {ACTION_COL} switch', 50, 95),
        *[(m, f'switch_{m}_distribution.png',
           f'Distance to new target at {ACTION_COL} switch ({m})', 50, 95)
          for m in _dist_metrics(switch_assay_dfs)],
    ]:
        fig, _ = tputil.plot_metric_distribution_across_assays(
            switch_assay_dfs, metric=metric,
            action_cols=None,
            focal_flies_map=FOCAL_FLIES,
            assay_colors=ASSAY_COLORS,
            bins=bins, xlim_percentile=pct,
            include_non_action=False)
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        _save(fig, figdir, fname)

    # Metric violins at switch frames
    for metric, fname, title in [
        ('abs_theta_error_deg', 'switch_abs_theta_error_deg_violin.png',
         f'Orientation error to new target at {ACTION_COL} switch'),
        *[(m, f'switch_{m}_violin.png',
           f'Distance to new target at {ACTION_COL} switch ({m})')
          for m in _dist_metrics(switch_assay_dfs)],
    ]:
        fig, _ = tputil.plot_metric_violin_across_assays(
            switch_assay_dfs, metric=metric,
            action_cols=None,
            focal_flies_map=FOCAL_FLIES,
            assay_colors=ASSAY_COLORS,
            include_non_action=False,
            ylim_percentile=95)
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        _save(fig, figdir, fname)


def _plot_switch_old_vs_new_deltas(combined_df, figdir):
    """Old-vs-new delta plots at switch for theta error and target ang. vel. in FOV."""
    print("\n── Switch old vs new deltas: theta error and target ang. vel. in FOV ──")

    comparison_df = tutil.get_switch_theta_error_comparison(
        combined_df, action_col=ACTION_COL)
    if len(comparison_df) == 0:
        print("  No switch events with theta_error data — skipping.")
        return

    comparison_assay_dfs = {assay: grp
                            for assay, grp in comparison_df.groupby('assay_type')}
    print(f"  {len(comparison_df)} switch events across "
          f"{comparison_df['acquisition'].nunique()} acquisitions")

    fig, _ = tputil.plot_switch_theta_error_delta_across_assays(
        comparison_assay_dfs, assay_colors=ASSAY_COLORS)
    _save(fig, figdir, 'switch_theta_error_old_vs_new.png')

    ang_df = tutil.get_switch_target_ang_vel_fov_comparison(
        combined_df, action_col=ACTION_COL)
    if len(ang_df) > 0:
        ang_assay_dfs = {assay: grp for assay, grp in ang_df.groupby('assay_type')}
        print(f"  {len(ang_df)} switch events with target_ang_vel_fov data")
        fig, _ = tputil.plot_switch_target_ang_vel_fov_delta_across_assays(
            ang_assay_dfs, assay_colors=ASSAY_COLORS)
        _save(fig, figdir, 'switch_target_ang_vel_fov_old_vs_new.png')
    else:
        print("  No target_ang_vel_fov data — skipping delta plot.")


def _plot_switch_metric_deltas_by_case(combined_df, figdir,
                                       threshold_deg=SWITCH_THETA_CASE_THRESHOLD_DEG):
    """One figure per switch case (new_lower/similar/new_higher); rows = metrics
    (distance, |θ error|, signed target ang vel FOV); cols = [Δ histogram |
    paired old→new per assay]. MFF only (combined_df is already MFF-filtered)."""
    print(f"\n── Switch new−old delta metrics, split by Δ|θ error| case (±{threshold_deg:g}°) ──")

    classified = tutil.classify_switches_by_theta_error(
        combined_df, action_col=ACTION_COL, threshold_deg=threshold_deg)
    if len(classified) == 0:
        print("  No theta-error switch data — skipping.")
        return
    # focal-fly switch events only (consistency with the other switch plots)
    classified = classified[classified.apply(
        lambda r: (FOCAL_FLIES.get(r['triad_type']) is None
                   or r['id'] in FOCAL_FLIES.get(r['triad_type'])), axis=1)]
    if len(classified) == 0:
        print("  No focal-fly switch events — skipping.")
        return

    key = ['acquisition', 'id', 'frame']
    master = classified[key + ['assay_type', 'switch_case',
                               'old_abs_theta_error_deg', 'new_abs_theta_error_deg',
                               'delta_abs_theta_error_deg']].copy()

    dist_cmp = tutil.get_switch_metric_comparison(
        combined_df, 'dist_to_other', action_col=ACTION_COL)
    if len(dist_cmp):
        master = master.merge(
            dist_cmp[key + ['old_dist_to_other', 'new_dist_to_other',
                            'delta_dist_to_other']], on=key, how='left')

    av_cmp = tutil.get_switch_target_ang_vel_fov_comparison(
        combined_df, action_col=ACTION_COL)
    av_cols = [c for c in ['old_target_ang_vel_fov_signed_deg',
                           'new_target_ang_vel_fov_signed_deg',
                           'delta_target_ang_vel_fov_signed_deg'] if c in av_cmp.columns]
    if len(av_cmp) and len(av_cols) == 3:
        master = master.merge(av_cmp[key + av_cols], on=key, how='left')

    metrics = []
    if 'delta_dist_to_other' in master.columns:
        metrics.append({'old': 'old_dist_to_other', 'new': 'new_dist_to_other',
                        'delta': 'delta_dist_to_other',
                        'label': 'distance to target (mm)'})
    metrics.append({'old': 'old_abs_theta_error_deg', 'new': 'new_abs_theta_error_deg',
                    'delta': 'delta_abs_theta_error_deg', 'label': '|θ error| (deg)'})
    if 'delta_target_ang_vel_fov_signed_deg' in master.columns:
        metrics.append({'old': 'old_target_ang_vel_fov_signed_deg',
                        'new': 'new_target_ang_vel_fov_signed_deg',
                        'delta': 'delta_target_ang_vel_fov_signed_deg',
                        'label': 'signed target ang vel FOV (deg/s)'})

    print("  counts by (assay, case):")
    print(master.groupby(['assay_type', 'switch_case']).size().to_string())

    for case in ('new_lower', 'similar', 'new_higher'):
        case_df = master[master['switch_case'] == case]
        if len(case_df) == 0:
            print(f"  {case}: no events — skipping.")
            continue
        fig, _ = tputil.plot_switch_delta_metrics_for_case(
            case_df, metrics, assay_colors=ASSAY_COLORS, case_label=case)
        if fig is not None:
            _save(fig, figdir, f'switch_delta_{case}.png')


def _plot_switch_by_theta_error_case(combined_df, ppm_dict, figdir, fps,
                                     threshold_deg=SWITCH_THETA_CASE_THRESHOLD_DEG,
                                     n_samples=10):
    """Split switches into 3 theta-error cases, then re-run the switch plots per case.

    Cases (delta = new_abs_theta_error - old_abs_theta_error):
        new_lower  -- delta <= -threshold  (new target better oriented)
        similar    -- |delta| < threshold
        new_higher -- delta >=  threshold  (new target worse oriented)
    """
    print(f"\n── Switches split by Δ|theta error| case (±{threshold_deg:g}°) ──")

    classified = tutil.classify_switches_by_theta_error(
        combined_df, action_col=ACTION_COL, threshold_deg=threshold_deg)
    if len(classified) == 0:
        print("  No theta-error switch data — skipping.")
        return

    # keep only focal-fly switch events (consistency with the plots)
    classified = classified[classified.apply(
        lambda r: (FOCAL_FLIES.get(r['triad_type']) is None
                   or r['id'] in FOCAL_FLIES.get(r['triad_type'])), axis=1)]
    if len(classified) == 0:
        print("  No focal-fly switch events — skipping.")
        return

    print("  counts by (assay, case):")
    print(classified.groupby(['assay_type', 'switch_case']).size().to_string())

    # 1) counts across assays
    fig, _ = tputil.plot_switch_case_counts_across_assays(
        classified, threshold_deg=threshold_deg)
    if fig is not None:
        _save(fig, figdir, 'switch_case_counts_across_assays.png')

    # Compute the heavy frames once, then filter per case via event keys.
    vector_df = tutil.get_switch_frame_vectors(combined_df, action_col=ACTION_COL)
    traj_df = tutil.get_switch_new_target_trajectories(
        combined_df, action_col=ACTION_COL, fps=fps, window_sec=4.0)

    for case in ('new_lower', 'similar', 'new_higher'):
        keys = (classified[classified['switch_case'] == case]
                [['acquisition', 'id', 'switch_frame']].drop_duplicates())
        if len(keys) == 0:
            print(f"  case {case}: no events — skipping.")
            continue
        print(f"  case {case}: {len(keys)} events")

        # 2a) target vectors (3-row positions plot with connecting lines)
        vec_case = vector_df.merge(
            keys.rename(columns={'switch_frame': 'frame'}),
            on=['acquisition', 'id', 'frame'], how='inner')
        if len(vec_case) > 0:
            vad = {a: g for a, g in vec_case.groupby('assay_type')}
            vppm = {k: v for k, v in ppm_dict.items() if k in vad}
            fig, _ = tputil.plot_switch_vectors_across_assays(
                vad, vppm, focal_flies_map=FOCAL_FLIES,
                old_color='dimgray', assay_colors=ASSAY_COLORS)
            if fig is not None:
                _save(fig, figdir, f'switchcase_{case}_target_vectors.png')

            # standalone old→new lines (vanilla: old grey, grey line, new assay-color)
            fig, _ = tputil.plot_switch_vectors_across_assays(
                vad, vppm, focal_flies_map=FOCAL_FLIES,
                old_color='dimgray', assay_colors=ASSAY_COLORS, vectors_only=True)
            if fig is not None:
                _save(fig, figdir, f'switchcase_{case}_oldnew_lines.png')

        traj_case = traj_df.merge(keys, on=['acquisition', 'id', 'switch_frame'],
                                  how='inner')
        if len(traj_case) == 0:
            continue
        traj_assay = {a: g for a, g in traj_case.groupby('assay_type')}

        # 2b) old/new positions at switch colored by target ang. vel. in FOV
        if 'new_target_ang_vel_fov_signed' in traj_case.columns:
            fig, _ = tputil.plot_switch_positions_colored_by_metric_across_assays(
                traj_assay, ppm_dict,
                t_rel_points=((0, 'at switch'),),
                focal_flies_map=FOCAL_FLIES, vlim_percentile=90)
            if fig is not None:
                _save(fig, figdir,
                      f'switchcase_{case}_positions_colored_target_ang_vel_fov.png')

            # standalone old→new lines at switch, colored by target ang vel FOV
            fig, _ = tputil.plot_switch_positions_colored_by_metric_across_assays(
                traj_assay, ppm_dict,
                t_rel_points=((0, 'at switch'),),
                focal_flies_map=FOCAL_FLIES, vlim_percentile=90, single_row=True)
            if fig is not None:
                _save(fig, figdir, f'switchcase_{case}_oldnew_lines_colored.png')

        # 2c) tails plot colored by target ang. vel. in FOV
        fig, _ = tputil.plot_switch_target_tail_colored_by_metric_across_assays(
            traj_assay, ppm_dict, focal_flies_map=FOCAL_FLIES, tail_sec=0.25, vlim_percentile=90)
        if fig is not None:
            _save(fig, figdir,
                  f'switchcase_{case}_tail_colored_target_ang_vel_fov.png')

        # 2d) sample trajectory pairs per assay (fly-colored | metric-colored),
        # in allocentric (lab-frame) coordinates, ±1 s around the switch
        for assay_type, tdf in traj_assay.items():
            focal_flies = FOCAL_FLIES.get(tdf['triad_type'].iloc[0])
            tdf_win = tdf[tdf['t_rel'].between(-1.0, 1.0)]
            fig, _ = tputil.plot_switch_target_trajectory_pairs_for_assay(
                tdf_win, ppm_dict.get(assay_type, 1), assay_type, frame='allo',
                new_color=ASSAY_COLORS.get(assay_type, 'tomato'),
                n_samples=n_samples, fps=fps, focal_flies=focal_flies,
                half_window_sec=1.0, vlim_percentile=90)
            if fig is not None:
                _save(fig, figdir, f'switchcase_{case}_pairs_{assay_type}_allo.png')


def _plot_switch_theta_maps(assay_dfs, figdir):
    """Switch locations over the focal male's target θ–θ field-of-view space.

    The basic two-panel map plus richer colorings (by the new target, and by the
    new / old / Δ FOV angular velocity and focal–target distance), and a panel
    colored by the fly's own angular velocity at the switch. Manual switches, MFF.
    """
    print("\n── Switch locations in the target θ–θ space ──")

    # basic two-panel map (switches → target x / target y)
    fig, _ = tputil.plot_target_switch_points(
        assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES, triad=TRIAD,
        action_col=ACTION_COL, switch_source=SWITCH_SOURCE, save_dir=figdir)
    if fig is not None:
        plt.close(fig)

    # all switches on one panel, colored by the new target (pink x / blue y)
    fig, _ = tputil.plot_switch_points_combined(
        assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES, triad=TRIAD,
        action_col=ACTION_COL, switch_source=SWITCH_SOURCE, save_dir=figdir)
    if fig is not None:
        plt.close(fig)

    # two-row maps colored by new / old / Δ target FOV angular velocity
    for _col, fig in tputil.plot_switch_points_by_motion(
            assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES, triad=TRIAD,
            action_col=ACTION_COL, switch_source=SWITCH_SOURCE,
            metric_col=FOV_METRIC, save_dir=figdir):
        plt.close(fig)

    # two-row maps colored by new / old / Δ focal–target distance (blue/red)
    for _col, fig in tputil.plot_switch_points_by_distance(
            assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES, triad=TRIAD,
            action_col=ACTION_COL, switch_source=SWITCH_SOURCE,
            metric_col=DIST_METRIC, save_dir=figdir):
        plt.close(fig)

    # all switches on one panel, colored by the fly's angular velocity at the switch
    fig, _ = tputil.plot_switch_points_by_angvel(
        assay_dfs, SEX_MAP, focal_flies_map=FOCAL_FLIES, triad=TRIAD,
        action_col=ACTION_COL, switch_source=SWITCH_SOURCE, save_dir=figdir)
    if fig is not None:
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rootdir = resolve_rootdir_and_theme(
        sys.argv, "Usage: python -m analyses.triad.src.figures.generate_switch_plots <rootdir> [--light]")

    data = load_and_prepare(rootdir, triad_filter=TRIAD)
    if data is None:
        sys.exit(0)

    combined_df = data.combined_df
    assay_dfs = data.assay_dfs
    ppm_dict = data.ppm_dict
    figdir = data.figdir
    fps = data.fps

    _plot_switch_rates(assay_dfs, figdir, fps)
    _plot_switch_positions_and_metrics(combined_df, assay_dfs, ppm_dict, figdir)
    _plot_switch_old_vs_new_deltas(combined_df, figdir)
    _plot_switch_metric_deltas_by_case(combined_df, figdir)
    _plot_switch_by_theta_error_case(combined_df, ppm_dict, figdir, fps)
    _plot_switch_theta_maps(assay_dfs, figdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
