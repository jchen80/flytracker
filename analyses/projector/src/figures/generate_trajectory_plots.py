"""CLI: plot fly + dot trajectories for individual assays.

For each species (and optionally each stimulus direction) writes one figure with
rows = assays/acqs and columns = LED intensity; each cell shows the fly path plus
each dot's path over that assay's frames at that intensity.

Usage:
    python -m analyses.projector.src.figures.generate_trajectory_plots <assay_type_dir>
    python -m analyses.projector.src.figures.generate_trajectory_plots <assay_type_dir> --by-direction
    python -m analyses.projector.src.figures.generate_trajectory_plots --root <projector-male-* dir>

--by-direction makes separate same/opposite figures (needs target_direction baked
in via build_assays --target-direction). --chasing-only additionally writes a
parallel '_chasing' set restricted to frames annotated as chasing either dot.
Figures go to {assay_type_dir}/figures/. Run with the flytracker env.
"""

import os
import argparse

from . import plotting_helpers as ph
from ..putil import plotting as pl


def main():
    ap = ph.base_arg_parser(__doc__)
    ap.add_argument('--by-direction', action='store_true',
                    help='separate figures per stimulus direction (needs target_direction)')
    ap.add_argument('--use-px', action='store_true', help='plot in pixels instead of mm')
    ap.add_argument('--chasing-only', action='store_true',
                    help='also make a parallel set of figures restricted to frames '
                         'annotated as chasing either dot (chasing_innerdot OR '
                         'chasing_outerdot)')
    args = ap.parse_args()

    df, out_dir = ph.load_from_args(args)
    if df is None or df.empty:
        print('No data loaded; build the parquet cache first with build_assays.py')
        return

    # which directions to split on
    directions = [None]
    if args.by_direction:
        if 'target_direction' not in df.columns or df['target_direction'].dropna().empty:
            print('[warn] --by-direction requested but no target_direction in the cache '
                  '(rebuild with build_assays --target-direction); making combined plots')
        else:
            directions = sorted(df['target_direction'].dropna().unique())

    species = sorted(df['species'].dropna().unique())
    print(f"species: {species} | directions: {directions}")

    # always make the full-trajectory figures; with --chasing-only also make a
    # parallel set restricted to chasing frames (suffix '_chasing')
    variants = [False, True] if args.chasing_only else [False]
    for sp in species:
        for d in directions:
            tag = sp + (f'_{d}' if d is not None else '')
            for chasing_only in variants:
                suffix = '_chasing' if chasing_only else ''
                pl.plot_assay_trajectories(df, species=sp, direction=d,
                                           use_mm=not args.use_px,
                                           chasing_only=chasing_only,
                                           assay_type_dir=out_dir,
                                           save_name=f'trajectories_{tag}{suffix}.png')


if __name__ == '__main__':
    main()