"""CLI: distributions of pursuit parameters during chasing, by species × arousal.

For each species the data is split by LED intensity (the arousal level). Two figure
styles are produced per metric by default:
    panels  -- one panel per LED intensity, overlaying the *chasing*-frame
               distribution on the *all*-frames distribution (a sanity check)
    overlay -- all LED intensities' chasing-frame histograms overlaid in one axes,
               colored by intensity, to read the shift with arousal at a glance
(`--styles violin` adds grouped violins; x=LED intensity, hue=chasing/all.)

Per-dot metrics are read relative to the dot being chased -- inner-dot value on
frames chasing the inner dot, outer-dot value on frames chasing the outer dot,
pooled across both dots. The speed/ang_vel metrics are single fly values (not
dot-specific); for those, chasing = frames chasing either dot.

Metrics:
    theta_error -- |theta error to chased dot| in degrees (absangle_to_{dot})
    dist        -- distance to chased dot (headdist_to_{dot}_mm if present, else dist_to_{dot}_mm)
    speed       -- fly speed (mm/s) from lin_speed (px/frame -> mm/s via ×fps ÷px_per_mm)
    ang_vel     -- fly |angular velocity| (deg/s) from ang_vel (rad/frame -> deg/s via ×fps ×180/π)

Usage:
    python -m analyses.projector.src.figures.generate_pursuit_distributions <assay_type_dir>
    python -m analyses.projector.src.figures.generate_pursuit_distributions <dir> --metrics dist speed
    python -m analyses.projector.src.figures.generate_pursuit_distributions <dir> --styles overlay
    python -m analyses.projector.src.figures.generate_pursuit_distributions --root <projector-male-2dot dir>

Figures land in {assay_type_dir}/figures/. Run with the flytracker env, e.g.:
    ~/miniconda3/envs/flytracker/bin/python -m analyses.projector.src.figures.generate_pursuit_distributions <dir>
"""

import os
import argparse

from . import plotting_helpers as ph
from ..putil import plotting as pl


def main():
    ap = ph.base_arg_parser(__doc__)
    ap.add_argument('--metrics', nargs='+', default=list(pl.PURSUIT_METRICS),
                    choices=list(pl.PURSUIT_METRICS),
                    help=f'which metrics to plot (default: {list(pl.PURSUIT_METRICS)})')
    ap.add_argument('--styles', nargs='+',
                    choices=['panels', 'overlay', 'cdf', 'violin', 'peracq', 'summary'],
                    default=['panels', 'overlay', 'cdf', 'violin', 'peracq', 'summary'],
                    help="which figures to make (default: all). "
                         "panels: chasing-vs-all histograms, one panel per LED intensity; "
                         "overlay: chasing-frame histograms across intensities overlaid, "
                         "colored by intensity; cdf: same overlay as ECDF outlines "
                         "(cumulative density, no fill); violin: x=LED intensity, "
                         "hue=chasing/all; peracq: panel per species, x=LED intensity, "
                         "per-acquisition mean as points with a horizontal mean line "
                         "(one figure per metric); summary: panel per species, x=LED "
                         "intensity, pooled frame center +/- spread (see --center/--spread) "
                         "as one line (one figure per metric)")
    ap.add_argument('--center', choices=['median', 'mean'], default='median',
                    help="center stat for the 'summary' style (default: median, robust "
                         "to the skewed tails of these metrics)")
    ap.add_argument('--spread', choices=['iqr', 'mad', 'sd', 'sem', 'ci95'], default='iqr',
                    help="error band for the 'summary' style: iqr (default, Q1..Q3, pairs "
                         "with median), mad, sd (frame spread), sem (std/sqrt n_frames), "
                         "or ci95 (1.96*sem)")
    ap.add_argument('--bins', type=int, default=120, help='histogram bins (default: 120)')
    ap.add_argument('--kde', action='store_true',
                    help='in the overlay, draw smooth KDE density lines per intensity '
                         'instead of step histograms (cleaner when low-n is noisy)')
    ap.add_argument('--xlim-percentile', type=float, default=99,
                    help='clip histogram x-range to this percentile to tame tails '
                         '(default: 99; pass 0 for full range)')
    args = ap.parse_args()

    df, out_dir = ph.load_from_args(args)
    if df is None or df.empty:
        print('No data loaded; build the parquet cache first with build_assays.py')
        return

    species = sorted(df['species'].dropna().unique())
    print(f"species: {species}")
    print(f"LED intensities: {sorted(df['led_intensity'].dropna().unique())}")

    pct = None if args.xlim_percentile == 0 else args.xlim_percentile
    for metric in args.metrics:
        # per-acquisition figure is one per metric (species are panels), not per species
        if 'peracq' in args.styles:
            pl.plot_pursuit_per_acquisition(
                df, metric=metric, assay_type_dir=out_dir, agg='mean',
                save_name=f'pursuit_{metric}_per_acquisition.png')
        # pooled frame-level mean +/- error summary, also one figure per metric
        if 'summary' in args.styles:
            pl.plot_pursuit_summary(
                df, metric=metric, center=args.center, spread=args.spread,
                assay_type_dir=out_dir, save_name=f'pursuit_{metric}_summary.png')
        for sp in species:
            for style in args.styles:
                if style in ('peracq', 'summary'):   # one figure per metric, handled above
                    continue
                save_name = f'pursuit_{metric}_{style}_{sp}.png'
                if style == 'cdf':
                    pl.plot_pursuit_by_intensity(
                        df, metric=metric, species=sp, cdf=True,
                        xlim_percentile=pct, assay_type_dir=out_dir, save_name=save_name)
                elif style == 'overlay':
                    if args.kde:
                        save_name = f'pursuit_{metric}_overlay_kde_{sp}.png'
                    pl.plot_pursuit_by_intensity(
                        df, metric=metric, species=sp, bins=args.bins, kde=args.kde,
                        xlim_percentile=pct, assay_type_dir=out_dir, save_name=save_name)
                else:
                    pl.plot_pursuit_distributions(
                        df, metric=metric, species=sp, bins=args.bins,
                        violin=(style == 'violin'), xlim_percentile=pct,
                        assay_type_dir=out_dir, save_name=save_name)


if __name__ == '__main__':
    main()