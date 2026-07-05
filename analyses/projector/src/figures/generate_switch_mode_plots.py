#!/usr/bin/env python3
"""Dot-configuration "mode" analysis for the projector dot-assay.

Every chasing frame (and every switch) is labelled by a 4×4 "mode" combining:
  - position combo: the sign of θ to each dot — left (θ ≥ 0) / right (θ < 0) for inner·outer
  - motion  combo: the sign of each dot's FOV motion — progressive (+) / regressive (−)

This visualizes how chasing time and target switches distribute over those 16 modes, the
per-mode switch rate relative to the chasing baseline, and example chasing frames per mode:

    switch_modes_positions_{species}.png  -- egocentric old→new switch vectors, per mode
    switch_mode_heatmaps.png              -- 4×4 chasing occupancy / switch share /
                                             p(switch | chasing), per species
    switch_mode_bars[ _by_led].png        -- per-mode bars: chasing-frame fraction (top) +
                                             switch count (bottom), pooled and split by LED
    mode_example_frames_{species}.png     -- sampled real video frames per mode (allocentric),
                                             each dot's past 1 s trajectory overlaid

LED frames are filtered with led_metadata.valid_led (Dmel ≤10%, Dyak ≤99%).

Usage:
    python -m analyses.projector.src.figures.generate_switch_mode_plots <assay_type_dir>
    python -m analyses.projector.src.figures.generate_switch_mode_plots --root <projector-male-2dot dir>
    python -m analyses.projector.src.figures.generate_switch_mode_plots <dir> --n-examples 4 --tail-sec 1.5

Output goes to {dir}/figures/. Run with the flytracker env.
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import libs.plotting as putil
from . import plotting_helpers as ph
from .. import led_metadata as lm
from ..putil import fov_plots as fp
from ..putil import switch_plots as sp

putil.set_sns_style(style='courtship', min_fontsize=11)


def main():
    ap = ph.base_arg_parser(__doc__)
    ap.add_argument('--n-examples', type=int, default=3,
                    help='example chasing frames sampled per mode (default 3)')
    ap.add_argument('--tail-sec', type=float, default=1.0,
                    help='length of the per-dot motion tail, seconds (default 1)')
    args = ap.parse_args()

    df, out_dir = ph.load_from_args(args)
    if df is None or df.empty:
        print('No data loaded; build the cache first with build_assays.')
        return

    df = df[lm.valid_led(df)].reset_index(drop=True)
    if df.empty:
        print('No frames with valid LED intensity.')
        return

    figdir = os.path.join(out_dir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    tbl = fp.get_dot_pair_fov(df, chasing_only=True)   # chasing baseline (per-frame modes)
    sw = sp.get_switch_fov(df)                          # one row per switch
    print(f"species: {sorted(df['species'].dropna().unique())} | "
          f"chasing frames: {len(tbl)} | switches: {0 if sw is None else len(sw)}")
    if tbl.empty:
        print('No chasing frames — nothing to plot.')
        return

    print("\n── egocentric old→new switch vectors per mode ──")
    for f in sp.plot_switch_modes_positions(sw, save_dir=figdir):
        plt.close(f)

    # old→new switch vectors restricted to the modes BOTH species occupy (> 2% chasing),
    # laid out species × mode, in three colorings (species / dot-switched-to / FOV motion)
    print("\n── old→new switch vectors in common modes (3 colorings) ──")
    for cb in ('species', 'newdot', 'fovmotion'):
        f = sp.plot_switch_modes_oldnew(tbl, sw, color_by=cb, save_dir=figdir)
        if f is not None:
            plt.close(f)

    print("\n── chasing baseline / switch share / p(switch|chasing) heatmaps ──")
    f = sp.plot_switch_mode_heatmaps(tbl, sw, save_dir=figdir)
    if f is not None:
        plt.close(f)

    print("\n── per-mode bars (chasing fraction + switch count) ──")
    for by_led in (False, True):
        f = sp.plot_switch_mode_bars(tbl, sw, by_led=by_led, save_dir=figdir)
        if f is not None:
            plt.close(f)
    # by-LED, restricted to the common modes, with switches stacked → inner / → outer
    f = sp.plot_switch_mode_bars(tbl, sw, by_led=True, common_only=True, split_newdot=True,
                                 save_dir=figdir)
    if f is not None:
        plt.close(f)

    print("\n── p(switch | chasing) vs LED intensity, per mode ──")
    f = sp.plot_switch_prob_by_led(tbl, sw, common_only=True, save_dir=figdir)
    if f is not None:
        plt.close(f)

    print("\n── example chasing frames per mode (real frame + motion tails) ──")
    for f in sp.plot_mode_example_frames(df, out_dir, n_examples=args.n_examples,
                                         tail_sec=args.tail_sec, save_dir=figdir):
        plt.close(f)

    print('\nDone.')


if __name__ == '__main__':
    main()
