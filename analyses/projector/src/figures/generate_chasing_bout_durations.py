"""CLI: chasing-either-dot bout-duration distributions, by species × LED intensity.

Runs of chasing either dot, with gaps shorter than --bridge-sec linked into one bout
(the fly may orient briefly mid-chase), then:
  - prints a per (species, LED intensity) summary: n_bouts, n_assays, median/mean
    duration, frac_short (fraction of bouts < --short-sec), total chasing seconds.
    Useful to check whether a block's "chasing" is mostly a short false-positive tail
    (a worry at 0%, where there is no stimulus to chase).
  - saves chasing_bout_durations.png: duration histograms overlaid by LED intensity,
    one panel per species (log-x by default).

Output: {assay_type_dir}/figures/chasing_bout_durations.png

Usage:
    python -m analyses.projector.src.figures.generate_chasing_bout_durations <assay_type_dir>
    python -m analyses.projector.src.figures.generate_chasing_bout_durations <dir> --bridge-sec 0.5 --short-sec 0.25
    python -m analyses.projector.src.figures.generate_chasing_bout_durations --root <projector-male-2dot dir>

Run with the flytracker env.
"""

import os
import argparse

import pandas as pd

from . import plotting_helpers as ph
from .. import analyze_outerdot as ao
from ..putil import plotting as pl


def main():
    ap = ph.base_arg_parser(__doc__)
    ap.add_argument('--bridge-sec', type=float, default=1.0,
                    help='link chasing bouts separated by gaps shorter than this (default: 1.0)')
    ap.add_argument('--short-sec', type=float, default=0.25,
                    help='bouts shorter than this count toward frac_short (default: 0.25)')
    ap.add_argument('--bins', type=int, default=40, help='histogram bins (default: 40)')
    ap.add_argument('--linear', action='store_true', help='linear duration axis (default: log)')
    args = ap.parse_args()

    df, out_dir = ph.load_from_args(args)
    if df is None or df.empty:
        print('No data loaded; build the parquet cache first with build_assays.py')
        return

    bouts = ao.chasing_bouts(df, bridge_sec=args.bridge_sec)
    if bouts.empty:
        print('No chasing bouts found (need chasing_innerdot/chasing_outerdot columns).')
        return

    summary = ao.chasing_bout_summary(bouts, short_sec=args.short_sec)
    pd.set_option('display.width', 200)
    print(f"\nChasing bout summary (gaps <{args.bridge_sec:g}s linked, "
          f"short = <{args.short_sec:g}s):\n")
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    pl.plot_chasing_bout_durations(
        df, bridge_sec=args.bridge_sec, bins=args.bins, log_x=not args.linear,
        short_sec=args.short_sec, assay_type_dir=out_dir)


if __name__ == '__main__':
    main()