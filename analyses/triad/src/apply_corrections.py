#!/usr/bin/env python3
"""
Replay the curation manifests onto fresh Stage-1 output.

The pipeline keeps machine output and human curation in separate layers:

    processed_mats/{acq}.parquet      Stage 1 output (disposable, regenerable)
    corrections/{acq}.switches.json   confirmed FP switches      (durable)
    corrections/{acq}.targets.json    post-switch target fixes   (durable)
    reviewed_mats/{acq}.parquet       = processed + manifests    (what plots read)

This script rebuilds reviewed_mats from processed_mats + corrections, so a
Stage-1 rerun never costs manual re-review.

Usage
-----
    # Rebuild reviewed_mats from processed_mats + corrections/ (the normal step)
    python -m analyses.triad.src.apply_corrections <rootdir>

    # One-time: backfill corrections/*.json from already-curated parquets
    python -m analyses.triad.src.apply_corrections <rootdir> --migrate

    --dry-run   report what would change without writing
    --force     (with --migrate) overwrite manifests that already exist
"""
import os
import sys
import glob
import argparse
import pandas as pd

from analyses.triad.src import corrections as C


def _acqs(processed_dir):
    return [(os.path.splitext(os.path.basename(fp))[0], fp)
            for fp in sorted(glob.glob(os.path.join(processed_dir, '*.parquet')))]


def migrate(rootdir, dry_run=False, force=False):
    """Reconstruct manifests from curated parquets in processed_mats/."""
    processed = C.processed_dir(rootdir)
    cdir      = C.corrections_dir(rootdir)
    acqs = _acqs(processed)
    if not acqs:
        print(f"No parquet files in {processed}")
        return

    print(f"Migrating manifests from {len(acqs)} parquet(s) in {processed}")
    print(f"  → writing to {cdir}\n")
    n_sw, n_tg = 0, 0
    for acq, fp in acqs:
        df = pd.read_parquet(fp)
        events  = C.reconstruct_switch_events(df)
        targets = C.reconstruct_target_corrections(df, action_col='courtship')
        if not events and not targets:
            continue

        msg = [f"  {acq[-46:]:46s}"]
        if events:
            sp = C.switches_path(cdir, acq)
            if os.path.exists(sp) and not force:
                msg.append(f"switches: SKIP (exists)")
            elif dry_run:
                msg.append(f"switches: would write {len(events)}")
                n_sw += 1
            else:
                C.write_switches_manifest(cdir, acq, events, merge=False)
                msg.append(f"switches: wrote {len(events)}")
                n_sw += 1
        if targets:
            tp = C.targets_path(cdir, acq)
            n_corr = sum(len(v) for v in targets.values())
            if os.path.exists(tp) and not force:
                msg.append(f"targets: SKIP (exists)")
            elif dry_run:
                msg.append(f"targets: would write {n_corr} corr / {len(targets)} focal")
                n_tg += 1
            else:
                C.write_targets_manifest(cdir, acq, targets, action_col='courtship')
                msg.append(f"targets: wrote {n_corr} corr / {len(targets)} focal")
                n_tg += 1
        print("  ".join(msg))

    verb = "would write" if dry_run else "wrote"
    print(f"\nDone. {verb} {n_sw} switch manifest(s), {n_tg} target manifest(s).")
    if not dry_run:
        print("Inspect them under corrections/ before running the build step.")


def build(rootdir, dry_run=False):
    """Rebuild reviewed_mats/ = processed_mats/ + corrections/."""
    processed = C.processed_dir(rootdir)
    cdir      = C.corrections_dir(rootdir)
    rdir      = C.reviewed_dir(rootdir)
    acqs = _acqs(processed)
    if not acqs:
        print(f"No parquet files in {processed}")
        return

    if not dry_run:
        os.makedirs(rdir, exist_ok=True)
    print(f"Building {rdir} from {len(acqs)} processed parquet(s) + {cdir}\n")
    n_sw, n_tg, n_plain = 0, 0, 0
    for acq, fp in acqs:
        has_sw = C.read_switches_manifest(cdir, acq) is not None
        has_tg = C.read_targets_manifest(cdir, acq) is not None
        tag = ('+switches' if has_sw else '') + ('+targets' if has_tg else '')
        tag = tag or '(copy-through, no manifest)'
        if has_sw: n_sw += 1
        if has_tg: n_tg += 1
        if not (has_sw or has_tg): n_plain += 1

        if dry_run:
            print(f"  would build {acq[-46:]:46s} {tag}")
            continue

        df = pd.read_parquet(fp)
        reviewed = C.build_reviewed_df(df, cdir, acq, action_col='courtship')
        reviewed.to_parquet(os.path.join(rdir, f'{acq}.parquet'), index=False)
        print(f"  built {acq[-46:]:46s} {tag}")

    verb = "would build" if dry_run else "built"
    print(f"\nDone. {verb} {len(acqs)} reviewed parquet(s): "
          f"{n_sw} with switch manifest, {n_tg} with target manifest, "
          f"{n_plain} copy-through.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rootdir', help='root dir containing processed_mats/')
    parser.add_argument('--migrate', action='store_true',
                        help='reconstruct corrections/*.json from curated parquets (one-time)')
    parser.add_argument('--force', action='store_true',
                        help='with --migrate, overwrite manifests that already exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='report what would change without writing')
    args = parser.parse_args()

    if not os.path.isdir(C.processed_dir(args.rootdir)):
        print(f"No processed_mats/ under {args.rootdir}")
        sys.exit(1)

    if args.migrate:
        migrate(args.rootdir, dry_run=args.dry_run, force=args.force)
    else:
        build(args.rootdir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()