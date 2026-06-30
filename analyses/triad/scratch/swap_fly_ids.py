#!/usr/bin/env python3
"""
Swap two fly IDs in FlyTracker .mat files (track, feat, actions) for all
multi-chamber acquisitions under a rootdir.

Originals are preserved as <filename>.bak before any modification.
If a .bak file already exists the acquisition is skipped (already processed).

Usage:
    python analyses/triad/src/swap_fly_ids.py <rootdir> [id1] [id2]

    id1, id2  global fly IDs to swap (default: 1 and 2)

Example:
    python analyses/triad/src/swap_fly_ids.py /Volumes/Julie/fb_MMF_MFF_triad_20mm/ 1 2
"""

import os
import sys
import glob
import shutil

import numpy as np
import scipy.io

try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mat(fpath):
    """Load a .mat file; return (mat_dict, loader) where loader is 'scipy'|'mat73'."""
    try:
        return scipy.io.loadmat(fpath), 'scipy'
    except NotImplementedError:
        if HAS_MAT73:
            return mat73.loadmat(fpath), 'mat73'
        raise


def _is_multichamber(acq_dir):
    """Return True if calibration.mat reports n_chambers > 1."""
    calib_path = os.path.join(acq_dir, 'calibration.mat')
    if not os.path.exists(calib_path):
        return False
    try:
        mat, loader = _load_mat(calib_path)
    except Exception:
        return False

    try:
        if loader == 'scipy':
            struct_name = next(k for k in mat if not k.startswith('__'))
            mdata = mat[struct_name]
            n_chambers = int(mdata['n_chambers'][0][0].flat[0])
        else:
            struct_name = next(k for k in mat if not k.startswith('__'))
            n_chambers = int(mat[struct_name].get('n_chambers', 1))
        return n_chambers > 1
    except Exception:
        return False


def _backup(fpath):
    """
    Copy fpath -> fpath.bak if the backup does not already exist.
    Returns True if backup was created, False if it already existed (skip flag).
    """
    bak = fpath + '.bak'
    if os.path.exists(bak):
        return False   # already processed
    shutil.copy2(fpath, bak)
    return True


# ---------------------------------------------------------------------------
# Per-file swap functions
# ---------------------------------------------------------------------------

def _swap_track_feat_v73(fpath, id1, id2):
    """
    Swap fly axis in a v7.3 (HDF5) track/feat file using h5py.

    MATLAB stores (n_flies, n_frames, n_fields) with dimensions reversed in HDF5,
    so the fly axis is last: HDF5 shape is (n_fields, n_frames, n_flies).
    We modify the dataset in-place without loading everything into memory.
    """
    if not HAS_H5PY:
        print(f"    {os.path.basename(fpath)}: h5py not installed — cannot handle v7.3, skipping.")
        return

    with h5py.File(fpath, 'r+') as f:
        # Find the struct group (exclude HDF5 internal groups like #refs#)
        struct_names = [k for k in f.keys() if not k.startswith('#')]
        if not struct_names:
            print(f"    {os.path.basename(fpath)}: no struct group found, skipping.")
            return
        struct_name = struct_names[0]

        if 'data' not in f[struct_name]:
            print(f"    {os.path.basename(fpath)}: no 'data' dataset in '{struct_name}', skipping.")
            return

        ds = f[struct_name]['data']
        # Fly axis is last in HDF5 (MATLAB reverses dimensions for column-major storage)
        n_flies = ds.shape[-1]
        if n_flies <= max(id1, id2):
            print(f"    {os.path.basename(fpath)}: only {n_flies} flies — "
                  f"cannot swap {id1}<->{id2}, skipping.")
            return

        fly1 = ds[..., id1][()]   # read into memory
        fly2 = ds[..., id2][()]
        ds[..., id1] = fly2
        ds[..., id2] = fly1

    print(f"    {os.path.basename(fpath)}: swapped fly {id1} <-> {id2} (v7.3 HDF5, "
          f"{n_flies} flies)")


def _swap_actions_v73(fpath, id1, id2):
    """
    Swap fly axis in a v7.3 (HDF5) actions file using h5py.

    MATLAB stores bouts as (n_flies, n_behaviors) cell array.
    In HDF5 the dimensions are reversed: shape is (n_behaviors, n_flies).
    Each element is an HDF5 object reference pointing to data in '#refs#'.
    We swap the reference values — the underlying data in '#refs#' is unchanged.
    """
    if not HAS_H5PY:
        print(f"    {os.path.basename(fpath)}: h5py not installed — cannot handle v7.3, skipping.")
        return

    with h5py.File(fpath, 'r+') as f:
        if 'bouts' not in f:
            print(f"    {os.path.basename(fpath)}: no 'bouts' dataset, skipping.")
            return

        ds = f['bouts']
        # Fly axis is last in HDF5 (n_behaviors, n_flies)
        n_flies = ds.shape[-1]
        if n_flies <= max(id1, id2):
            print(f"    {os.path.basename(fpath)}: only {n_flies} flies in bouts — "
                  f"cannot swap {id1}<->{id2}, skipping.")
            return

        refs = ds[()]  # read full reference array into memory
        tmp = refs[..., id1].copy()
        refs[..., id1] = refs[..., id2]
        refs[..., id2] = tmp
        ds[()] = refs   # write back swapped references

    print(f"    {os.path.basename(fpath)}: swapped fly {id1} <-> {id2} (v7.3 HDF5, "
          f"{n_flies} flies)")


def swap_track_or_feat(fpath, id1, id2):
    """
    Swap fly rows id1 <-> id2 in the 'data' field of a -track.mat or -feat.mat.

    The MATLAB struct contains a 'data' field of shape (n_flies, n_frames, n_fields).
    Accessed via scipy as mat[struct_name]['data'][0][0].
    """
    if not _backup(fpath):
        print(f"    {os.path.basename(fpath)}: .bak exists — already processed, skipping.")
        return

    mat, loader = _load_mat(fpath)

    if loader == 'mat73':
        _swap_track_feat_v73(fpath, id1, id2)
        return

    struct_name = next((k for k in mat if not k.startswith('__')), None)
    if struct_name is None:
        print(f"    {os.path.basename(fpath)}: no struct found, skipping.")
        return

    try:
        data = mat[struct_name]['data'][0][0]   # (n_flies, n_frames, n_fields)
    except Exception as e:
        print(f"    {os.path.basename(fpath)}: could not read 'data': {e}, skipping.")
        return

    if data.shape[0] <= max(id1, id2):
        print(f"    {os.path.basename(fpath)}: only {data.shape[0]} flies — "
              f"cannot swap {id1}<->{id2}, skipping.")
        return

    # Swap in place (data is a view into mat's structured array)
    tmp = data[id1].copy()
    data[id1] = data[id2]
    data[id2] = tmp

    scipy.io.savemat(fpath, mat)
    print(f"    {os.path.basename(fpath)}: swapped fly {id1} <-> {id2}  "
          f"({data.shape[0]} flies, {data.shape[1]} frames)")


def swap_actions(fpath, id1, id2):
    """
    Swap fly rows id1 <-> id2 in the 'bouts' object array of a -actions.mat.

    The actions file has:
        mat['bouts']  shape (n_flies, n_behaviors) object array
        mat['behs']   cell array of behavior name strings
    """
    if not _backup(fpath):
        print(f"    {os.path.basename(fpath)}: .bak exists — already processed, skipping.")
        return

    mat, loader = _load_mat(fpath)

    if loader == 'mat73':
        _swap_actions_v73(fpath, id1, id2)
        return

    if 'bouts' not in mat:
        print(f"    {os.path.basename(fpath)}: no 'bouts' field, skipping.")
        return

    bouts = mat['bouts']   # (n_flies, n_behaviors) object array
    if bouts.shape[0] <= max(id1, id2):
        print(f"    {os.path.basename(fpath)}: only {bouts.shape[0]} flies in bouts — "
              f"cannot swap {id1}<->{id2}, skipping.")
        return

    # Swap entire fly rows in the object array
    row1 = bouts[id1].copy()
    row2 = bouts[id2].copy()
    bouts[id1] = row2
    bouts[id2] = row1

    # Rebuild bouts with explicit float64 arrays so scipy writes clean MATLAB matrices.
    # Empty cells → (0,0) to match MATLAB's native empty [] (not (0,3), which
    # scipy may mangle on roundtrip). Non-empty cells → explicit float64 (N,3).
    n_flies, n_behs = bouts.shape
    bouts_out = np.empty((n_flies, n_behs), dtype=object)
    for i in range(n_flies):
        for j in range(n_behs):
            elem = bouts[i, j]
            if isinstance(elem, np.ndarray) and elem.ndim == 2 and elem.shape[1] == 3 and elem.shape[0] > 0:
                bouts_out[i, j] = np.asfortranarray(elem, dtype=np.float64)
            else:
                bouts_out[i, j] = np.zeros((0, 0), dtype=np.float64)

    # Preserve the original behs structure exactly — do NOT rebuild it.
    # ft_actions_to_bout_df uses [i[0][0] for i in mat['behs']] which depends on
    # the original MATLAB cell layout (typically (n_behs, 1) column vector).
    # Re-encoding behs changes that layout and breaks the parser.
    save_dict = {
        'bouts': bouts_out,
        'behs': mat['behs'],  # pass-through unchanged
    }

    scipy.io.savemat(fpath, save_dict, format='5')

    # Roundtrip check: verify scipy can read back the shapes it just wrote.
    verify = scipy.io.loadmat(fpath)
    bad_cells = []
    good_cells = 0
    for i in range(n_flies):
        for j in range(n_behs):
            cell = verify['bouts'][i, j]
            if cell.size > 0:  # non-empty bout cell
                if cell.ndim == 2 and cell.shape[1] == 3:
                    good_cells += 1
                else:
                    bad_cells.append((i, j, cell.shape))
    if bad_cells:
        print(f"    WARNING: scipy roundtrip produced unexpected bout shapes: {bad_cells[:3]}")
        print(f"    The file may not be parseable by ft_actions_to_bout_df. "
              f"Restoring from .bak — you will need to apply the swap a different way.")
        # Restore from backup so the file isn't left in a bad state
        bak = fpath + '.bak'
        if os.path.exists(bak):
            shutil.copy2(bak, fpath)
            print(f"    Restored {os.path.basename(fpath)} from .bak.")
    else:
        print(f"    {os.path.basename(fpath)}: swapped fly {id1} <-> {id2}  "
              f"({n_flies} flies, {n_behs} behaviors, {good_cells} non-empty bout cells)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def restore_actions_from_bak(rootdir):
    """
    Restore any *-actions.mat files from their .bak backups.
    scipy.io.savemat cannot reliably write MATLAB cell arrays (bouts/behs),
    so this undoes any corrupted actions swaps. The ID swap for actions is
    applied in memory by pairwise_transformation_metrics.py instead.
    """
    acquisition_parentdir = os.path.join(rootdir, 'raw_videos')
    acqs = sorted(f for f in os.listdir(acquisition_parentdir) if not f.startswith('.'))
    restored = 0
    for acq in acqs:
        ft_subdir = os.path.join(acquisition_parentdir, acq, acq)
        for bak in sorted(glob.glob(os.path.join(ft_subdir, '*-actions.mat.bak'))):
            original = bak[:-4]  # strip .bak
            shutil.copy2(bak, original)
            print(f"  Restored {os.path.basename(original)} from .bak")
            restored += 1
    print(f"Restored {restored} actions file(s).")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    rootdir = sys.argv[1]
    id1 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    id2 = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    acquisition_parentdir = os.path.join(rootdir, 'raw_videos')
    acqs = sorted(f for f in os.listdir(acquisition_parentdir) if not f.startswith('.'))
    print(f"Found {len(acqs)} acquisitions under {acquisition_parentdir}")
    print(f"Swapping fly IDs {id1} <-> {id2} in multi-chamber acquisitions\n")

    for acq in acqs:
        acq_dir = os.path.join(acquisition_parentdir, acq)

        if not _is_multichamber(acq_dir):
            print(f"  {acq}: single-chamber, skipping.")
            continue

        print(f"  {acq}: multi-chamber")
        ft_subdir = os.path.join(acq_dir, acq)

        for fp in sorted(glob.glob(os.path.join(ft_subdir, '*-track.mat'))):
            swap_track_or_feat(fp, id1, id2)

        for fp in sorted(glob.glob(os.path.join(ft_subdir, '*-feat.mat'))):
            swap_track_or_feat(fp, id1, id2)

        for fp in sorted(glob.glob(os.path.join(ft_subdir, '*-actions.mat'))):
            # Restore from .bak first if the previous write was corrupted,
            # then re-attempt the swap with the fixed reconstruction logic.
            bak = fp + '.bak'
            if os.path.exists(bak):
                shutil.copy2(bak, fp)
                print(f"    {os.path.basename(fp)}: restored from .bak before re-swapping.")
                # Remove .bak so _backup() will create a fresh one
                os.remove(bak)
            swap_actions(fp, id1, id2)

    print("\nDone.")


if __name__ == '__main__':
    main()
