#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch metadata in already-processed parquet files.

For acquisitions processed before parse_acquisition_metadata was fixed to
handle multi-chamber filenames, the 'species' column may be wrong (e.g. ch1
gets the first species instead of the second), and the focal fly filtering
during processing may have used the wrong triad_type.

This script:
  1. Compares old vs. new metadata for every parquet in processed_mats/.
  2. Patches 'species' in-place when only that column changed.
  3. Warns about acquisitions where triad_type changed — those had wrong focal
     fly selection and should be reprocessed (or deleted so the processing
     script recreates them).
  4. With --delete-bad, removes parquets whose triad_type was wrong so that
     pairwise_transformation_metrics.py will regenerate them on the next run.

Usage:
    python -m analyses.triad.src.patch_parquet_metadata <rootdir>
    python -m analyses.triad.src.patch_parquet_metadata <rootdir> --delete-bad
"""

import sys
import os
import glob
import pandas as pd
from analyses.triad.src import data_io

FOCAL_FLIES = {
    'MMF':  [0, 1],
    'MFF':  [0],
    'MMMF': [0, 1, 2],
}


def _old_species(acq_key):
    """Reproduce the old (broken) species logic for comparison."""
    # Old code: 'mel' if 'mel' in acq else 'yak'  (case-insensitive substring)
    return 'mel' if 'mel' in acq_key.lower() else 'yak'


def _old_triad_type(acq_key):
    """Reproduce the old (broken) triad_type logic for comparison."""
    # Old code: always used the first triad token regardless of chamber
    import re
    base = re.sub(r'_ch\d+$', '', acq_key)
    fields = [f for f in base.split('_') if f]
    known = {'MMF', 'MFF', 'MMMF'}
    for f in fields:
        candidate = f.split('-')[0]
        if candidate in known:
            return candidate
    return None


def main():
    delete_bad = '--delete-bad' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if not args:
        print(__doc__)
        sys.exit(1)

    rootdir = args[0]
    processedmat_dir = os.path.join(rootdir, 'processed_mats')
    parquet_files = sorted(glob.glob(os.path.join(processedmat_dir, '*.parquet')))

    if not parquet_files:
        print(f"No parquet files found in {processedmat_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet file(s) in {processedmat_dir}\n")

    species_patched = []
    needs_reprocess = []
    ok = []

    for fp in parquet_files:
        acq = os.path.splitext(os.path.basename(fp))[0]

        try:
            new_meta = data_io.parse_acquisition_metadata(acq)
        except ValueError as e:
            print(f"  SKIP {acq}: could not parse — {e}")
            continue

        new_species = new_meta['species']
        new_triad = new_meta['triad_type']
        old_species = _old_species(acq)
        old_triad = _old_triad_type(acq)

        triad_changed = (old_triad is not None) and (old_triad != new_triad)
        species_changed = old_species != new_species

        if triad_changed:
            old_focal = FOCAL_FLIES.get(old_triad, '?')
            new_focal = FOCAL_FLIES.get(new_triad, '?')
            print(f"  [REPROCESS] {acq}")
            print(f"    triad_type: {old_triad!r} -> {new_triad!r}  "
                  f"(focal flies: {old_focal} -> {new_focal})")
            if species_changed:
                print(f"    species:    {old_species!r} -> {new_species!r}")
            needs_reprocess.append(fp)

        elif species_changed:
            print(f"  [PATCH]     {acq}")
            print(f"    species: {old_species!r} -> {new_species!r}")

            df = pd.read_parquet(fp)
            df['species'] = new_species
            # Also fix assay_type / triad_type if already in the parquet
            if 'triad_type' in df.columns:
                df['triad_type'] = new_triad
            if 'assay_type' in df.columns:
                df['assay_type'] = f"{new_species}_{new_triad}"
            df.to_parquet(fp, index=False)
            species_patched.append(acq)

        else:
            ok.append(acq)

    print(f"\n{'='*60}")
    print(f"  OK (no change needed):  {len(ok)}")
    print(f"  Species patched:        {len(species_patched)}")
    print(f"  Need reprocessing:      {len(needs_reprocess)}")

    if needs_reprocess:
        print(f"\nAcquisitions needing reprocessing (wrong focal fly pairs were used):")
        for fp in needs_reprocess:
            print(f"  {os.path.basename(fp)}")

        if delete_bad:
            print("\n--delete-bad: removing bad parquets so they will be regenerated.")
            for fp in needs_reprocess:
                os.remove(fp)
                print(f"  Deleted {os.path.basename(fp)}")
        else:
            print("\nRun with --delete-bad to remove them so pairwise_transformation_metrics.py")
            print("will regenerate them with the correct triad type and focal fly selection.")


if __name__ == '__main__':
    main()
