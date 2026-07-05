"""CLI: plot fraction of time chasing / orienting / singing vs LED intensity,
split by species (Dmel vs Dyak).

Loads the per-assay parquet cache (built by build_assays.py), aggregates the
fraction of frames each behavior fires per (species, LED intensity) -- averaged
across assays so each assay counts equally -- and writes a multi-panel figure
(one panel per behavior, one line per species) plus a tidy summary CSV.

Usage:
    python -m analyses.projector.src.figures.generate_behavior_fraction_plots <assay_type_dir>
    python -m analyses.projector.src.figures.generate_behavior_fraction_plots --root <projector-male-2dot dir>
    python -m analyses.projector.src.figures.generate_behavior_fraction_plots <assay_type_dir> --behaviors chasing_outerdot singing

Outputs land in {assay_type_dir}/figures/ (or the --root's first level when --root).
Run with the flytracker env:
    ~/miniconda3/envs/flytracker/bin/python -m analyses.projector.src.figures.generate_behavior_fraction_plots <dir>
"""

import os
import argparse

from . import plotting_helpers as ph
from .. import analyze_outerdot as ao
from ..putil import plotting as pl

DEFAULT_BEHAVIORS = ['chasing_innerdot', 'chasing_outerdot', 'orienting_outerdot', 'singing']


def main():
    ap = ph.base_arg_parser(__doc__)
    ap.add_argument('--behaviors', nargs='+', default=DEFAULT_BEHAVIORS,
                    help=f'behavior columns to plot (default: {DEFAULT_BEHAVIORS})')
    hue_grp = ap.add_mutually_exclusive_group()
    hue_grp.add_argument('--by-speed', action='store_true',
                         help='color each panel by stimulus speed (needs stim_speed '
                              'from build_assays --stim-speed)')
    hue_grp.add_argument('--by-direction', action='store_true',
                         help='color each panel by stimulus direction same/opposite '
                              '(needs target_direction from build_assays --target-direction)')
    ap.add_argument('--union-chasing', action='store_true',
                    help='also plot chasing_anydot = chasing_innerdot OR chasing_outerdot')
    ap.add_argument('--save-name', default='fraction_vs_intensity.png')
    args = ap.parse_args()

    # where to load from + where to write
    df, out_dir = ph.load_from_args(args)
    if df is None or df.empty:
        print('No data loaded; build the parquet cache first with build_assays.py')
        return

    if args.union_chasing:
        df = ao.add_chasing_union(df)

    behaviors = [b for b in args.behaviors if b in df.columns]
    if args.union_chasing and 'chasing_anydot' in df.columns and 'chasing_anydot' not in behaviors:
        behaviors.append('chasing_anydot')
    missing = set(args.behaviors) - set(behaviors)
    if missing:
        print(f'[warn] behaviors not in data, skipping: {sorted(missing)}')

    # pick the optional hue column from the flags
    hue = 'stim_speed' if args.by_speed else 'target_direction' if args.by_direction else None
    if hue and (hue not in df.columns or df[hue].notna().sum() == 0):
        flag = '--stim-speed' if hue == 'stim_speed' else '--target-direction'
        print(f"[warn] split requested but no '{hue}' in the cache "
              f"(rebuild with build_assays {flag}); falling back to no split")
        hue = None

    print(f"species present: {sorted(df['species'].dropna().unique())}")
    print(f"LED intensities: {sorted(df['led_intensity'].dropna().unique())}")
    if hue:
        print(f"{hue}: {sorted(df[hue].dropna().unique())}")
    print(f"behaviors: {behaviors}")

    summary = ao.fraction_chasing_orienting(df, behaviors, hue=hue)
    print('\n' + summary.to_string())

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(fig_dir, os.path.splitext(args.save_name)[0] + '.csv')
    summary.to_csv(csv_path, index=False)
    print(f'\nsaved {csv_path}')

    pl.plot_fraction_vs_intensity(df, behaviors=behaviors, assay_type_dir=out_dir,
                                  save_name=args.save_name, hue=hue)


if __name__ == '__main__':
    main()
