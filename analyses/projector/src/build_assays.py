"""CLI: build the per-assay parquet cache for the projector dot-assay.

It resolves a px->mm calibration once per assay type (launching the scale-bar GUI
on the first assay's video if the type isn't calibrated yet, or whenever
--recalibrate is given), then bakes each assay's JAABA mats + run json into one
tidy parquet under {assay_type}/processed/.

Usage:
    python -m analyses.projector.src.build_assays <assay_type_dir> [--assay A] [--recalibrate]
    python -m analyses.projector.src.build_assays --root <projector-male-2dot dir>  [--recalibrate]

Run with the flytracker env, e.g.:
    ~/miniconda3/envs/flytracker/bin/python -m analyses.projector.src.build_assays <dir>
"""

import os
import argparse

from . import data_io as dio
from . import calibration as cal


def build_one_assay_type(assay_type_dir, only_assay=None, recalibrate=False,
                         add_stim_speed=False, add_target_direction=False,
                         add_switches=False):
    assays = dio.list_assays(assay_type_dir)
    if only_assay:
        assays = [a for a in assays if a == only_assay]
    if not assays:
        print(f"No JAABA assays found under {assay_type_dir}")
        return
    print(f"{os.path.basename(os.path.normpath(assay_type_dir))}: {len(assays)} assays")

    # one px->mm calibration for the whole assay type (projector geometry is fixed);
    # calibrate on the first assay's video if needed, then reuse for every assay
    if recalibrate or not cal.has_calibration(assay_type_dir):
        px_per_mm = cal.calibrate_assay_type(assay_type_dir, assays[0])
        if px_per_mm is None:
            print(f"  [skip] {assay_type_dir}: not calibrated")
            return
    else:
        px_per_mm = cal.load_calibration(assay_type_dir)

    for assay in assays:
        df = dio.build_assay_df(assay_type_dir, assay, px_per_mm,
                                add_stim_speed=add_stim_speed,
                                add_target_direction=add_target_direction,
                                add_switches=add_switches)
        path = dio.save_assay_df(df, assay_type_dir, assay)
        extra = f", speeds={sorted(df['stim_speed'].dropna().unique())}" if 'stim_speed' in df else ''
        if 'target_direction' in df:
            extra += f", dirs={sorted(df['target_direction'].dropna().unique())}"
        if 'switching' in df:
            extra += f", switches={int(df['switching'].sum())}"
        print(f"  [{assay}] {len(df)} frames, "
              f"{df['led_block'].nunique()} LED blocks{extra}  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('assay_type_dir', nargs='?', help='a single {assay_type} folder')
    g.add_argument('--root', help='loop over every assay_type under this folder')
    ap.add_argument('--assay', default=None, help='build only this assay')
    ap.add_argument('--recalibrate', action='store_true',
                    help='re-run the calibration GUI even if already calibrated')
    ap.add_argument('--stim-speed', action='store_true',
                    help='also bake per-frame stimulus speed parsed from the trajectory filename')
    ap.add_argument('--target-direction', action='store_true',
                    help='also bake same/opposite dot direction (TEMP heuristic from a 2x45 trajectory name)')
    ap.add_argument('--switches', action='store_true',
                    help='also merge switch annotations from the FlyTracker actions.mat')
    args = ap.parse_args()

    if args.root:
        for name in sorted(os.listdir(args.root)):
            atd = os.path.join(args.root, name)
            if os.path.isdir(os.path.join(atd, 'JAABA')):
                build_one_assay_type(atd, args.assay, args.recalibrate,
                                     args.stim_speed, args.target_direction, args.switches)
    else:
        build_one_assay_type(args.assay_type_dir, args.assay, args.recalibrate,
                             args.stim_speed, args.target_direction, args.switches)


if __name__ == '__main__':
    main()
