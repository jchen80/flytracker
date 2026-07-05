#!/usr/bin/env python3
"""
Egocentric "two dots on the FOV" figures for the projector dot-assay.

The projector analog of the triad FOV / θ–θ occupancy plots. During CHASING frames, for
the male's egocentric field of view (inner dot = x, outer dot = y), produces:

    fov_density            -- 2D ego density (mm) of the pursued vs other dot (rows)
    theta_joint_density    -- joint density of (θ to inner, θ to outer)
    theta_fovmotion_maps   -- (θ_inner, θ_outer) mean target FOV motion, inner & outer (rows)
    theta_angvel_map       -- (θ_inner, θ_outer) mean male angular velocity

The switch-location maps over this same θ–θ space (switch_points_by_*, switch_prob_*)
live in generate_switch_plots.py.

Each is produced in two modes: 'aggregate' (panels = species, valid LED pooled) and 'split'
(one figure per species, panels = that species' LED intensities). LED frames are filtered
with led_metadata.valid_led (Dmel ≤10%, Dyak ≤99%).

Usage:
    python -m analyses.projector.src.figures.generate_fov_plots <assay_type_dir>
    python -m analyses.projector.src.figures.generate_fov_plots --root <projector-male-2dot dir>

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

putil.set_sns_style(style='courtship', min_fontsize=11)


def main():
    ap = ph.base_arg_parser(__doc__)
    args = ap.parse_args()

    df, out_dir = ph.load_from_args(args)
    if df is None or df.empty:
        print('No data loaded; build the cache first with build_assays.')
        return

    # keep only frames with a usable LED intensity (per-species ceiling)
    df = df[lm.valid_led(df)].reset_index(drop=True)
    if df.empty:
        print('No frames with valid LED intensity.')
        return

    figdir = os.path.join(out_dir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    tbl = fp.get_dot_pair_fov(df, chasing_only=True)
    print(f"species: {sorted(df['species'].dropna().unique())} | "
          f"chasing frames: {len(tbl)}")
    if tbl.empty:
        print('No chasing frames — nothing to plot.')
        return

    for mode in ('aggregate', 'split'):
        print(f"\n── {mode} ──")
        figs = []
        figs += fp.plot_fov_density(tbl, mode=mode, save_dir=figdir)
        figs += fp.plot_theta_joint_density(tbl, mode=mode, save_dir=figdir)
        # joint P(courting inner, courting outer) over all frames (2×2 states)
        figs += fp.plot_courting_joint_prob(df, mode=mode, save_dir=figdir)
        figs += fp.plot_theta_fovmotion_maps(tbl, mode=mode, save_dir=figdir)
        figs += fp.plot_theta_angvel_map(tbl, mode=mode, save_dir=figdir)
        for f in figs:
            plt.close(f)

    print('\nDone.')


if __name__ == '__main__':
    main()
