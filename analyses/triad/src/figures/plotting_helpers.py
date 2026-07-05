#!/usr/bin/env python3
"""Shared scaffolding for the triad ``generate_*_plots`` CLIs.

Holds the constants, small helpers and the load/prepare sequence that every
plotting CLI used to re-declare. CLI scripts import what they need from here so
their ``main()`` shrinks to a single :func:`load_and_prepare` call plus their own
plot sections. This is a CLI-level module — it does not touch the ``putil/``
plotting backend.
"""

import os
import sys
from collections import namedtuple

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from analyses.triad.src import data_io
from analyses.triad.src import util as tutil
import libs.plotting as putil

# ── Shared constants ──────────────────────────────────────────────────────────

ACTION_COL = 'courtship'

FOCAL_FLIES = {
    'MMF':  [0, 1],
    'MFF':  [0],
    'MMMF': [0, 1, 2],
}

SEX_MAP = {
    'MMF':  {0: 'M', 1: 'M', 2: 'F'},
    'MFF':  {0: 'M', 1: 'F', 2: 'F'},
    'MMMF': {0: 'M', 1: 'M', 2: 'M', 3: 'F'},
}

PPM_VARIATION_THRESHOLD = 0.05

BODY_ADJ_DIST = 'dist_to_other_body_adj'

# Velocity range bins shared by the pursuit-metric and position-density CLIs:
# (min_mm_s, max_mm_s), None = open bound.
VEL_RANGES = [
    (None, 5),
    (5,    None),
]
VEL_SOURCES = ['focal', 'target']

# Fixed bin width (mm) for the distance axis of the dist-vs-θ relationship plots.
DIST_BIN_MM = 1.0

PLOT_STYLE = 'courtship'
MIN_FONT_SIZE = 12
putil.set_sns_style(style=PLOT_STYLE, min_fontsize=MIN_FONT_SIZE)


def resolve_rootdir_and_theme(argv, usage):
    """Parse the shared triad-CLI args: a required ``<rootdir>`` plus an optional
    ``--light`` flag (white background / black text instead of the default dark
    courtship theme). Applies the theme and returns rootdir; prints ``usage`` and
    exits if no rootdir was given.
    """
    if '--light' in argv[1:]:
        putil.set_sns_style(style='courtship_light', min_fontsize=MIN_FONT_SIZE)
    rest = [a for a in argv[1:] if a != '--light']
    if not rest:
        print(usage)
        sys.exit(1)
    return rest[0]


# ── Small helpers ─────────────────────────────────────────────────────────────

def _save(fig, figdir, name):
    savepath = os.path.join(figdir, name)
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    print(f"  Saved {name}")
    plt.close(fig)


def _dist_metrics(assay_dfs):
    """Distance metrics present in at least one assay df."""
    metrics = ['dist_to_other']
    if any(BODY_ADJ_DIST in df.columns for df in assay_dfs.values()):
        metrics.append(BODY_ADJ_DIST)
    return metrics


def _range_tag(lo, hi):
    if lo is None:
        return f'lt{hi}'
    if hi is None:
        return f'gt{lo}'
    return f'{lo}to{hi}'


def _bin_label(lo, hi):
    """Human-readable velocity bin label for axis ticks."""
    if lo is None:
        return f'< {hi}'
    if hi is None:
        return f'> {lo}'
    return f'{lo}–{hi}'


def _vel_tag(lo, hi):
    return f'lt{hi:g}' if lo is None else f'gt{lo:g}' if hi is None else f'{lo:g}to{hi:g}'


def _vel_label(lo, hi):
    return f'< {hi:g}' if lo is None else f'> {lo:g}' if hi is None else f'{lo:g}–{hi:g}'


def build_velocity_bins(assay_dfs, vel_source, label_fn):
    """Filter each assay to pursuit frames within each VEL_RANGES speed bin.

    Returns a list of ``(label, bin_assay_dfs)`` for the non-empty bins, where
    ``label`` is ``label_fn(lo, hi)``. Collapses the per-section loop that the
    velocity-binned plot sections used to repeat.
    """
    binned = []
    for lo, hi in VEL_RANGES:
        bin_assay_dfs = {
            k: tutil.filter_pursuit_frames(
                v, action_col=ACTION_COL,
                min_vel_mm_s=lo, max_vel_mm_s=hi,
                vel_source=vel_source)
            for k, v in assay_dfs.items()
        }
        bin_assay_dfs = {k: v for k, v in bin_assay_dfs.items() if len(v) > 0}
        if bin_assay_dfs:
            binned.append((label_fn(lo, hi), bin_assay_dfs))
    return binned


def add_female_target_subcases(split_assay_dfs, split_ppm):
    """Add ``*_MMF_2M_F`` subcases (2M frames restricted to female-target courtship).

    A subcase of MMF_2M ("courting either target"): demote male-target courtship
    to non-action so the per-focal courtship selection keeps female-target frames
    only. Returns augmented copies; the inputs are left unmodified.
    """
    out_dfs = dict(split_assay_dfs)
    out_ppm = dict(split_ppm)
    for k in [k for k in list(split_assay_dfs) if k.endswith('_MMF_2M')]:
        fk = f'{k}_F'
        sub = tutil.restrict_action_to_target_sex(
            split_assay_dfs[k], SEX_MAP, focal_flies_map=FOCAL_FLIES,
            keep_sex='F', action_col=ACTION_COL)
        if 'assay_type' in sub.columns:
            sub['assay_type'] = fk
        out_dfs[fk] = sub
        out_ppm[fk] = split_ppm.get(k)
    return out_dfs, out_ppm


# ── Load + prepare ────────────────────────────────────────────────────────────

PreparedData = namedtuple('PreparedData', [
    'combined_df', 'assay_dfs', 'ppm_dict', 'fps', 'figdir',
    'split_assay_dfs', 'split_ppm', 'assay_colors', 'assay_colors_split',
])


def load_and_prepare(rootdir, triad_filter=None):
    """Load processed parquet, build assay dfs, the 2M/1M MMF split and colors.

    Returns a :class:`PreparedData`, or ``None`` if no data (or, with
    ``triad_filter``, no matching acquisitions) is found.

    ``assay_colors`` is the base Ruta-lab courtship scheme; ``assay_colors_split``
    expands it for the 2M/1M split (2M = base color, 1M = lightened). The
    ``split_*`` fields break each MMF assay into 2M (both focal males courting)
    and 1M (one male) frame subsets; MFF/MMMF pass through unchanged.
    """
    processedmat_dir = data_io.resolve_data_dir(rootdir)
    figdir = os.path.join(rootdir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    print("Loading data...")
    combined_df = data_io.load_all_processed_dfs(processedmat_dir, verbose=False)
    if len(combined_df) == 0:
        print("No data found — nothing to plot.")
        return None
    combined_df['abs_theta_error_deg'] = combined_df['theta_error_deg'].abs()

    if triad_filter is not None:
        combined_df = combined_df[combined_df['assay_type'].str.contains(triad_filter)]
        if len(combined_df) == 0:
            print(f"No {triad_filter} acquisitions found — nothing to plot.")
            return None

    assay_dfs = data_io.get_assay_dfs(combined_df)
    fps = int(combined_df['FPS'].iloc[0])
    print(f"Assay types: {sorted(assay_dfs.keys())}")
    print(f"FPS: {fps}")

    ppm_dict = {}
    for assay_type, assay_df in assay_dfs.items():
        ppm = tutil.get_assay_ppm(assay_df, assay_type, threshold=PPM_VARIATION_THRESHOLD)
        if ppm is not None:
            ppm_dict[assay_type] = ppm

    # Split each MMF assay into 2M (both males courting) and 1M (one male) frame
    # subsets so the comparisons run on the 6-way set (Dmel/Dyak × {MMF-2M,
    # MMF-1M, MFF}). MFF passes through unchanged.
    split_assay_dfs = tutil.split_assay_dfs_by_courting_count(
        assay_dfs, FOCAL_FLIES, split_triad='MMF', action_col=ACTION_COL)
    split_ppm = tutil.expand_keyed_dict_for_split(ppm_dict, assay_dfs, split_triad='MMF')

    assay_colors = putil.ASSAY_TYPE_COLORS   # shared Ruta-lab courtship scheme
    assay_colors_split = tutil.expand_keyed_dict_for_split(   # 2M = base, 1M = lighter
        assay_colors, assay_dfs, split_triad='MMF',
        transform=lambda c, sub: c if sub == '2M' else putil.lighten(c, 0.45))

    return PreparedData(
        combined_df=combined_df, assay_dfs=assay_dfs, ppm_dict=ppm_dict,
        fps=fps, figdir=figdir,
        split_assay_dfs=split_assay_dfs, split_ppm=split_ppm,
        assay_colors=assay_colors, assay_colors_split=assay_colors_split)
