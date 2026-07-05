"""Shared scaffolding for the projector ``generate_*`` plotting CLIs.

Holds the argparse parent (the ``assay_type_dir`` / ``--root`` pair every CLI
takes) and the load branch they all repeat. A CLI builds its parser from
:func:`base_arg_parser`, adds its own options, then loads with
:func:`load_from_args`. This is a CLI-level module — it does not touch the
``putil/`` plotting backend.
"""

import argparse

from .. import data_io as dio


def base_arg_parser(description):
    """An ArgumentParser preloaded with the mutually-exclusive assay_type_dir/--root
    inputs every projector plotting CLI accepts. Add CLI-specific options to the
    returned parser, then call ``parse_args()``."""
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('assay_type_dir', nargs='?', help='a single {assay_type} folder')
    g.add_argument('--root', help='pool every assay_type under this folder')
    return ap


def load_from_args(args):
    """Load the per-assay parquet cache per the assay_type_dir / --root args.

    Returns ``(df, out_dir)``; ``df`` may be None/empty (the caller does its own
    empty-check so it can print a build-step-specific message).
    """
    if args.root:
        return dio.load_all_assay_types(args.root), args.root
    return dio.load_all_assays(args.assay_type_dir), args.assay_type_dir
