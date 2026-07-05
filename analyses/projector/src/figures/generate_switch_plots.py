"""CLI: switch analyses from the parquet cache (built with build_assays --switches).

Produces, in {assay_type_dir}/figures/:
  - switch_rate_vs_intensity.png        : switches / courtship-minute vs LED intensity, per species
  - switch_target_vectors.png           : triad-style egocentric old/new target positions, per species
  - switch_target_vectors_by_intensity.png : same, one column per species x LED intensity
  - switch_motion_tails.png             : old/new target at switch + tail colored by retinal FOV motion (prog/reg)
  - switch_positions_by_motion.png      : old/new target positions colored by retinal FOV motion (prog/reg), per species
  - switch_positions_by_motion_by_intensity.png : same, one column block per species x LED intensity
  - switch_target_vectors[_by_intensity]_old{lt30,gt30}.png : vectors split by the OLD target's |theta error|
      at the switch (<30 deg = still well-oriented to the abandoned target, vs >30 deg)
  - switch_positions_by_motion[_by_intensity]_old{lt30,gt30}.png : motion-colored positions, same old-target split
  - switch_{target_vectors,positions_by_motion}[_by_intensity]_old{lt30,gt30}_{prog,reg}.png :
      combinations of the old-target |theta error| bin and the old target's prog/regressive FOV motion
  - switch_fly_pose[_by_intensity][_<colorby>]_<species>.png : fly lab-frame position+heading at the switch
      (all switches, and split by LED intensity), colored by new target (default) or a metric --
      old_theta/new_theta (|theta error| to old/new target), old_fov/new_fov (old/new target FOV motion),
      or led (discrete LED-intensity levels, one equal step along the species color per level; aggregate only)
  - switch_fly_pose_by_case_<species>.png : same, 2x2 grid of the old-target cases
      (|theta error| < / > 30 deg x prog/regressive), colored by new target (inner/outer)
  - switch_trajectory_samples_<species>[_allo][_paired].png : triad-style per-event sampled switch trajectories,
      ego + allo frames; '_paired' adds partner panels with targets colored by retinal FOV bearing-rate (prog/reg)
  - switch_timecourse_theta_error.png   : peri-switch |θ error to dot| (deg), new vs old target
  - switch_timecourse_headdist.png      : peri-switch head-to-dot distance (mm), new vs old target
  - switch_points_by_{newdot,motion_*,angvel}.png : switch locations over the male's (θ inner,
      θ outer) FOV bearing space, colored by new dot / new·old·Δ FOV motion / male angular velocity
      (aggregate = panels per species; split = one figure per species x LED intensity)
  - switch_prob_{chasing,occupancy}.png / switch_prob_by_target.png : p(switch | θ–θ position)
  - switch_rate_per_courtship_min.png   : pooled switch rate per courtship-minute (LED-valid frames)
  - switch_rate.csv                     : the per (species, LED intensity) rate summary

The θ–θ FOV-space switch maps (switch_points_*, switch_prob_*) were moved here from
generate_fov_plots.py; they run on LED-valid frames (led_metadata.valid_led).

Usage:
    python -m analyses.projector.src.figures.generate_switch_plots <assay_type_dir>
    python -m analyses.projector.src.figures.generate_switch_plots --root <projector-male-2dot dir>
    python -m analyses.projector.src.figures.generate_switch_plots <dir> --by-direction

Run with the flytracker env.
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from . import plotting_helpers as ph
from .. import analyze_switches as asw
from .. import led_metadata as lm
from ..putil import fov_plots as fp
from ..putil import switch_plots as swp

# Bins on the OLD (abandoned) target's |theta error| (deg) at the switch frame:
# 'oldlt30' = fly was still well-oriented to the target it left; 'oldgt30' = not.
OLD_THETA_BINS = [('oldlt30', (0, 30)), ('oldgt30', (30, 180))]
# Bins on the OLD target's retinal FOV bearing-rate sign at the switch frame.
OLD_MOTION_BINS = [('prog', 'progressive'), ('reg', 'regressive')]


def main():
    ap = ph.base_arg_parser(__doc__)
    ap.add_argument('--by-direction', action='store_true',
                    help='color the switch-rate plot by stimulus direction (needs target_direction)')
    ap.add_argument('--per-second', action='store_true',
                    help='rate per courtship-second instead of per-minute')
    args = ap.parse_args()

    df, out_dir = ph.load_from_args(args)
    if df is None or df.empty:
        print('No data loaded; build the cache first with build_assays --switches')
        return
    if 'switching' not in df.columns or df['switching'].sum() == 0:
        print('No switch annotations in the cache; rebuild with build_assays --switches '
              '(after annotating in FlyTracker).')
        return

    hue = 'target_direction' if args.by_direction else None
    if hue and (hue not in df.columns or df[hue].notna().sum() == 0):
        print(f"[warn] --by-direction needs 'target_direction'; ignoring")
        hue = None

    print(f"species: {sorted(df['species'].dropna().unique())} | "
          f"total switches: {int(df['switching'].sum())}")

    summary, per = asw.switch_rate(df, hue=hue, per_minute=not args.per_second)
    print('\nswitch rate summary:\n' + summary.to_string())
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    summary.to_csv(os.path.join(fig_dir, 'switch_rate.csv'), index=False)

    swp.plot_switch_rate(df, hue=hue, per_minute=not args.per_second,
                        assay_type_dir=out_dir)
    # fly lab-frame pose at the switch (all switches + split by LED intensity), per
    # species, colored by new target then by old/new |theta error| (FOV colorings
    # are added in the FOV block below since they need the FOV columns)
    species_list = sorted(df['species'].dropna().unique())
    for sp in species_list:
        for cb in ('target', 'old_theta', 'new_theta'):
            suf = '' if cb == 'target' else f'_{cb}'
            swp.plot_switch_fly_pose(df, species=sp, color_by=cb, assay_type_dir=out_dir,
                                    save_name=f'switch_fly_pose{suf}_{sp}.png')
            swp.plot_switch_fly_pose_by_intensity(
                df, species=sp, color_by=cb, assay_type_dir=out_dir,
                save_name=f'switch_fly_pose_by_intensity{suf}_{sp}.png')
        # color the aggregate pose by LED intensity (redundant in the by-intensity grid)
        swp.plot_switch_fly_pose(df, species=sp, color_by='led', assay_type_dir=out_dir,
                                save_name=f'switch_fly_pose_led_{sp}.png')
    # switch-target vectors: one figure by species, one by species x LED intensity
    swp.plot_switch_vectors(df, groupby=('species',), assay_type_dir=out_dir,
                           save_name='switch_target_vectors.png')
    swp.plot_switch_vectors(df, groupby=('species', 'led_intensity'), assay_type_dir=out_dir,
                           save_name='switch_target_vectors_by_intensity.png')

    # ... and the same, split by the OLD target's |theta error| at the switch
    # (oldlt30 = fly still well-oriented to the abandoned target; oldgt30 = not)
    for tag, rng in OLD_THETA_BINS:
        swp.plot_switch_vectors(df, groupby=('species',), old_theta_range=rng,
                               assay_type_dir=out_dir,
                               save_name=f'switch_target_vectors_{tag}.png')
        swp.plot_switch_vectors(df, groupby=('species', 'led_intensity'), old_theta_range=rng,
                               assay_type_dir=out_dir,
                               save_name=f'switch_target_vectors_by_intensity_{tag}.png')

    # retinal FOV motion (progressive/regressive) at switches:
    #   (1) tails colored by motion, (2) old/new positions colored by motion,
    #   (3) triad-style sampled per-event trajectories (ego + allo, plain + FOV-paired)
    if 'target_ang_vel_fov_outerdot_signed_deg' in df.columns:
        swp.plot_switch_motion_tails(df, groupby=('species',), assay_type_dir=out_dir)
        # old/new positions colored by motion: one figure by species, one by species x LED intensity
        swp.plot_switch_positions_by_motion(df, groupby=('species',), assay_type_dir=out_dir,
                                           save_name='switch_positions_by_motion.png')
        swp.plot_switch_positions_by_motion(df, groupby=('species', 'led_intensity'),
                                           assay_type_dir=out_dir,
                                           save_name='switch_positions_by_motion_by_intensity.png')
        # ... and split by the OLD target's |theta error| at the switch
        for tag, rng in OLD_THETA_BINS:
            swp.plot_switch_positions_by_motion(
                df, groupby=('species',), old_theta_range=rng, assay_type_dir=out_dir,
                save_name=f'switch_positions_by_motion_{tag}.png')
            swp.plot_switch_positions_by_motion(
                df, groupby=('species', 'led_intensity'), old_theta_range=rng,
                assay_type_dir=out_dir,
                save_name=f'switch_positions_by_motion_by_intensity_{tag}.png')

        # combinations: OLD target |theta error| bin x OLD target prog/regr motion
        # (needs FOV columns; both vectors and motion-colored positions, per bin)
        for ttag, rng in OLD_THETA_BINS:
            for mtag, motion in OLD_MOTION_BINS:
                tag = f'{ttag}_{mtag}'
                swp.plot_switch_vectors(
                    df, groupby=('species',), old_theta_range=rng, old_motion=motion,
                    assay_type_dir=out_dir, save_name=f'switch_target_vectors_{tag}.png')
                swp.plot_switch_vectors(
                    df, groupby=('species', 'led_intensity'), old_theta_range=rng,
                    old_motion=motion, assay_type_dir=out_dir,
                    save_name=f'switch_target_vectors_by_intensity_{tag}.png')
                swp.plot_switch_positions_by_motion(
                    df, groupby=('species',), old_theta_range=rng, old_motion=motion,
                    assay_type_dir=out_dir, save_name=f'switch_positions_by_motion_{tag}.png')
                swp.plot_switch_positions_by_motion(
                    df, groupby=('species', 'led_intensity'), old_theta_range=rng,
                    old_motion=motion, assay_type_dir=out_dir,
                    save_name=f'switch_positions_by_motion_by_intensity_{tag}.png')

        # fly lab-frame pose at switch, 2x2 old-target case grid, colored by new target;
        # plus the FOV-motion colorings (need FOV cols) for the agg + by-intensity poses
        for sp in species_list:
            swp.plot_switch_fly_pose_by_case(
                df, species=sp, assay_type_dir=out_dir,
                save_name=f'switch_fly_pose_by_case_{sp}.png')
            for cb in ('old_fov', 'new_fov'):
                swp.plot_switch_fly_pose(df, species=sp, color_by=cb, assay_type_dir=out_dir,
                                        save_name=f'switch_fly_pose_{cb}_{sp}.png')
                swp.plot_switch_fly_pose_by_intensity(
                    df, species=sp, color_by=cb, assay_type_dir=out_dir,
                    save_name=f'switch_fly_pose_by_intensity_{cb}_{sp}.png')
                # same motion coloring, split into panels by destination dot (inner/outer)
                swp.plot_switch_fly_pose_by_newdot(
                    df, species=sp, color_by=cb, assay_type_dir=out_dir,
                    save_name=f'switch_fly_pose_by_newdot_{cb}_{sp}.png')
        # big focused figure: just old->new vectors colored by FOV motion, one panel per species
        swp.plot_switch_vectors_by_motion(df, groupby=('species',), assay_type_dir=out_dir)
        # triad-style sampled switch trajectories, per species, ego + allo frames
        swp.plot_switch_trajectory_samples(df, groupby=('species',), assay_type_dir=out_dir)
        # peri-switch time-courses (new vs old target), aligned at the switch,
        # split into switches TO the inner dot vs TO the outer dot for clarity
        swp.plot_switch_peri_timecourse(df, yvar='theta_error', groupby=('species', 'new_target'),
                                       assay_type_dir=out_dir)
        swp.plot_switch_peri_timecourse(df, yvar='headdist', groupby=('species', 'new_target'),
                                       assay_type_dir=out_dir)
    else:
        print('[warn] no FOV columns in cache; rebuild with build_assays for the '
              'motion-colored switch figures')

    # ── Switch locations over the θ–θ FOV bearing space ───────────────────────
    # (moved here from generate_fov_plots). Needs LED-valid frames + the FOV chasing
    # table (get_dot_pair_fov) and the switch-FOV table (get_switch_fov).
    df_led = df[lm.valid_led(df)].reset_index(drop=True)
    tbl = fp.get_dot_pair_fov(df_led, chasing_only=True) if not df_led.empty else df_led
    sw = swp.get_switch_fov(df_led) if not df_led.empty else None
    if df_led.empty or tbl.empty:
        print('[info] no LED-valid chasing frames — skipping θ–θ switch maps')
    else:
        for mode in ('aggregate', 'split'):
            figs = []
            figs += swp.plot_switch_points_by_newdot(tbl, sw, mode=mode, save_dir=fig_dir)
            figs += swp.plot_switch_points_by_motion(tbl, sw, mode=mode, save_dir=fig_dir)
            figs += swp.plot_switch_points_by_angvel(tbl, sw, mode=mode, save_dir=fig_dir)
            # p(switch | θ–θ position), coarse bins, two priors (denominators):
            #   chasing → p(switch | chasing here); occupancy → p(switch | present here)
            figs += swp.plot_switch_prob_map(df_led, mode=mode, prior='chasing', save_dir=fig_dir)
            figs += swp.plot_switch_prob_map(df_led, mode=mode, prior='occupancy', save_dir=fig_dir)
            figs += swp.plot_switch_prob_by_target(df_led, mode=mode, save_dir=fig_dir)
            for f in figs:
                plt.close(f)
        # overall switch rate per courtship-minute, pooled across LED intensities
        fig = swp.plot_switch_rate_per_courtship_min(df_led, save_dir=fig_dir)
        if fig is not None:
            plt.close(fig)


if __name__ == '__main__':
    main()
