"""Loaders for the JAABA outputs of the projector dot-assay.

Each assay (one recording) has a JAABA experiment folder:
    JAABA/{assay}/
        trx.mat                 1xN struct array: trx(i) = target i
        perframe/{feat}.mat     data = 1xN cell, data{i} = target i's 1xnframes vector
        scores_{behavior}.mat   allScores.postprocessed[i] (binary), .scores[i] (continuous)

Target ordering follows the preprocessing convention (jaaba_dots/dot.py::DOT_LAYOUTS):
    id 1 = fly, id 2 = innerdot, id 3 = outerdot   (2dot model)
    id 1 = fly, id 2 = outerdot                     (1dot model: lone dot is the outer dot)
The fly is the only target with a non-dot id; everything else is a projected dot.
The mapping is read FROM the data (trx ids) rather than hardcoded so the 1dot and
future 2-fly cases extend with minimal change.
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio

# trx per-frame vector fields (length nframes) vs scalar metadata fields.
TRX_PERFRAME = ['x', 'y', 'theta', 'a', 'b',
                'x_mm', 'y_mm', 'theta_mm', 'a_mm', 'b_mm', 'timestamps']
TRX_SCALAR = ['nframes', 'firstframe', 'endframe', 'off', 'id', 'fps']

# id -> target name. Dots are ids >= 2; id 2 is the inner dot when an outer
# (id 3) is present, otherwise the lone dot is the outer dot.
_DOT_NAME_2DOT = {2: 'innerdot', 3: 'outerdot'}


def _loadmat(path):
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)


def target_names(ids):
    """Map a list of trx ids to target names.

    Fly = id 1. Dots get inner/outer names: with two dots present they are
    id2=innerdot, id3=outerdot; a single dot (id 2 only) is the outer dot.
    """
    ids = [int(i) for i in ids]
    names = {}
    dot_ids = sorted(i for i in ids if i != 1)
    for i in ids:
        if i == 1:
            names[i] = 'fly'
        elif len(dot_ids) == 1:
            names[i] = 'outerdot'
        else:
            names[i] = _DOT_NAME_2DOT.get(i, f'dot{i}')
    return names


def load_trx(trx_path):
    """Load trx.mat -> dict {target_name: DataFrame}.

    Each DataFrame is frame-indexed with the per-frame trx fields as columns,
    plus scalar fields (id, nframes, firstframe, fps) broadcast as constants.
    """
    trx = _loadmat(trx_path)['trx']
    trx = np.atleast_1d(trx)
    ids = [int(getattr(t, 'id')) for t in trx]
    names = target_names(ids)

    out = {}
    for t in trx:
        rec = {f: np.asarray(getattr(t, f), dtype=float).ravel() for f in TRX_PERFRAME}
        df = pd.DataFrame(rec)
        for f in TRX_SCALAR:
            df[f] = float(getattr(t, f))
        df['sex'] = str(getattr(t, 'sex'))
        out[names[int(getattr(t, 'id'))]] = df.reset_index(drop=True)
    return out


def load_perframe(perframe_dir, target_index=0):
    """Load every perframe/{feat}.mat for one target -> dict {feature: 1d array}.

    `data` is a 1xN cell (one vector per target); index the requested target
    (default 0 = fly). Returns an empty dict if the folder is missing.
    """
    feats = {}
    for mat in sorted(glob.glob(os.path.join(perframe_dir, '*.mat'))):
        feat = os.path.splitext(os.path.basename(mat))[0]
        data = np.atleast_1d(_loadmat(mat)['data'])
        feats[feat] = np.asarray(data[target_index], dtype=float).ravel()
    return feats


def load_scores(scores_mat_path, target_index=0):
    """Load one scores_{behavior}.mat -> (behavior, binary, continuous) for a target.

    binary     = allScores.postprocessed[i] (0/1 per frame)
    continuous = allScores.scores[i] (raw classifier margin per frame)
    behavior is parsed from the filename (scores_{behavior}.mat).
    """
    behavior = os.path.splitext(os.path.basename(scores_mat_path))[0]
    if behavior.startswith('scores_'):
        behavior = behavior[len('scores_'):]
    al = _loadmat(scores_mat_path)['allScores']
    binary = np.asarray(np.atleast_1d(al.postprocessed)[target_index], dtype=float).ravel()
    cont = np.asarray(np.atleast_1d(al.scores)[target_index], dtype=float).ravel()
    return behavior, binary, cont


def find_score_mats(jaaba_assay_dir):
    """List scores_*.mat paths in an assay's JAABA folder."""
    return sorted(glob.glob(os.path.join(jaaba_assay_dir, 'scores_*.mat')))
