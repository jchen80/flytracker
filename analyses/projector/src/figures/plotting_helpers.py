"""Shared scaffolding for the projector ``generate_*`` plotting CLIs.

Holds the argparse parent (the ``assay_type_dir`` / ``--root`` pair and the
``--light`` theme flag every CLI takes) and the load branch they all repeat. A
CLI builds its parser from :func:`base_arg_parser`, adds its own options, then
loads with :func:`load_from_args`. This is a CLI-level module — the only
plotting it touches is selecting the theme (:func:`apply_theme`); it does not
reach into the ``putil/`` plotting backend.
"""

import argparse

import libs.plotting as putil

from .. import data_io as dio

# Theme every projector figure CLI draws in: the Ruta-lab courtship scheme (dark
# #262626 background). ``--light`` swaps in its white-background variant. The
# putil/ modules also set the dark style at import; whichever set_sns_style()
# call runs last wins, and apply_theme() runs after all imports.
PLOT_STYLE = 'courtship'
MIN_FONT_SIZE = 11


def base_arg_parser(description):
    """An ArgumentParser preloaded with the mutually-exclusive assay_type_dir/--root
    inputs and the --light theme flag every projector plotting CLI accepts. Add
    CLI-specific options to the returned parser, then call ``parse_args()``."""
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('assay_type_dir', nargs='?', help='a single {assay_type} folder')
    g.add_argument('--root', help='pool every assay_type under this folder')
    ap.add_argument('--light', action='store_true',
                    help='white background / black text (courtship_light) instead of '
                         'the default dark courtship theme')
    return ap


def apply_theme(args):
    """Apply the theme selected by ``--light``. Must run before any figure is
    created; :func:`load_from_args` calls it, so a CLI using that needs no extra
    call. Only the background/foreground flip — the assay and species colors are
    the same on both themes.
    """
    style = 'courtship_light' if getattr(args, 'light', False) else PLOT_STYLE
    putil.set_sns_style(style=style, min_fontsize=MIN_FONT_SIZE)


def load_from_args(args):
    """Apply the --light theme, then load the per-assay parquet cache per the
    assay_type_dir / --root args.

    Returns ``(df, out_dir)``; ``df`` may be None/empty (the caller does its own
    empty-check so it can print a build-step-specific message).
    """
    apply_theme(args)
    if args.root:
        return dio.load_all_assay_types(args.root), args.root
    return dio.load_all_assays(args.assay_type_dir), args.assay_type_dir
