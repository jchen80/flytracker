"""Read switch annotations back from a FlyTracker `-actions.mat` (annotated in the
FlyTracker visualizer on the export produced by flytracker_export.py) and merge
them into a tidy assay df.

The visualizer stores actions as `behs` (cell of behavior names) + `bouts`
(n_flies x n_behaviors cell; each cell is a kx3 array of [start, end, value]). For
switch annotation Julie labels single-frame bouts of `switching_inner` /
`switching_outer` on the male (fly 0). Each bout's start frame is one switch; the
post-switch target comes from the behavior name. FlyTracker frames are 1-based, so
they are converted to the parquet's 0-based `frame`.
"""

import os
import numpy as np
import scipy.io as sio

ACTIONS_DIRNAME = 'flytracker'


def actions_path(assay_type_dir, assay):
    """Path to an assay's FlyTracker actions.mat (nested export layout), or None."""
    p = os.path.join(assay_type_dir, ACTIONS_DIRNAME, assay, assay, f'{assay}-actions.mat')
    return p if os.path.exists(p) else None


def _target_of(beh_name):
    """'inner'/'outer' from a switching behavior name, else None."""
    n = beh_name.lower()
    if 'switch' not in n:
        return None
    if 'inner' in n:
        return 'inner'
    if 'outer' in n:
        return 'outer'
    return None


def load_switches(assay_type_dir, assay, fly_index=0, one_based=True):
    """Parse switch events from the actions.mat -> [{frame, target}], sorted.

    Reads single-frame `switching_inner`/`switching_outer` bouts for `fly_index`
    (the male). FlyTracker 1-based frames are shifted to 0-based when one_based.
    Returns [] if there is no actions.mat.
    """
    path = actions_path(assay_type_dir, assay)
    if path is None:
        return []
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    behs = [str(b).strip() for b in np.atleast_1d(m['behs'])]
    bouts = m['bouts']                                  # (n_flies, n_behaviors) object array
    shift = 1 if one_based else 0

    switches = {}
    for bi, name in enumerate(behs):
        target = _target_of(name)
        if target is None:
            continue
        cell = np.atleast_1d(bouts[fly_index, bi]) if np.ndim(bouts) == 2 else bouts[bi]
        cell = np.atleast_2d(cell)
        if cell.size == 0:
            continue
        for start in cell[:, 0]:
            switches[int(start) - shift] = target          # last label wins per frame
    return [{'frame': f, 'target': switches[f]} for f in sorted(switches)]


def merge_switches(df, switches):
    """Add `switching` (1 at a switch frame, else 0) and `switch_target`
    ('inner'/'outer' at the switch frame, else None) to a frame-indexed df."""
    df = df.copy()
    df['switching'] = 0
    df['switch_target'] = None
    by_frame = {int(s['frame']): s['target'] for s in switches}
    if by_frame:
        hit = df['frame'].isin(by_frame)
        df.loc[hit, 'switching'] = 1
        df.loc[hit, 'switch_target'] = df.loc[hit, 'frame'].map(by_frame)
    return df
