"""
Manifest-based curation layer for the triad pipeline.

Separates machine output from human curation so that re-running Stage 1
(pairwise_transformation_metrics.py) never destroys hand review:

    processed_mats/{acq}.parquet    -- Stage 1 output ONLY (disposable, regenerable)
    corrections/{acq}.switches.json -- confirmed FP switches (from concordance review)
    corrections/{acq}.targets.json  -- post-switch target reassignments (from target review)
    reviewed_mats/{acq}.parquet     -- derived = apply(processed + manifests); Stage 3 reads this

The two manifests are the durable record of human decisions. reviewed_mats is
rebuilt from processed_mats + manifests by apply_corrections.py, so a Stage-1
rerun costs no manual re-review. The manifests are the SOLE source of truth for
confirmed switches and target fixes -- actions.mat is left untouched (it holds
only the original manual switch annotations), so processed_mats.switching stays
pure-manual.

Manifest formats
----------------
{acq}.switches.json
    {"acq": <str>,
     "events": [{"fly_id": <int>, "frame": <int>, "after_target": <int|null>}, ...]}

{acq}.targets.json
    {"acq": <str>, "action_col": "courtship",
     "corrections": {"<focal_id>": [[switch_frame, before_tgt, after_tgt], ...], ...}}
    (before_tgt / after_tgt may be null)
"""
import os
import json
import numpy as np
import pandas as pd

from analyses.triad.src import util as tutil

PROCESSED_DIRNAME   = 'processed_mats'
CORRECTIONS_DIRNAME = 'corrections'
REVIEWED_DIRNAME    = 'reviewed_mats'


# ── Directory / path helpers ────────────────────────────────────────────────
def processed_dir(rootdir):   return os.path.join(rootdir, PROCESSED_DIRNAME)
def corrections_dir(rootdir): return os.path.join(rootdir, CORRECTIONS_DIRNAME)
def reviewed_dir(rootdir):    return os.path.join(rootdir, REVIEWED_DIRNAME)

def switches_path(cdir, acq): return os.path.join(cdir, f'{acq}.switches.json')
def targets_path(cdir, acq):  return os.path.join(cdir, f'{acq}.targets.json')


# ── Switches manifest I/O ───────────────────────────────────────────────────
def read_switches_manifest(cdir, acq):
    """Return list of event dicts [{fly_id, frame, after_target}], or None."""
    fp = switches_path(cdir, acq)
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f).get('events', [])


def write_switches_manifest(cdir, acq, events, merge=True):
    """
    Write the confirmed-switch manifest. events: iterable of dicts with keys
    fly_id, frame, after_target. When merge=True, union with any existing
    manifest (dedup on (fly_id, frame), new after_target wins).
    """
    os.makedirs(cdir, exist_ok=True)
    norm = {(int(e['fly_id']), int(e['frame'])): e.get('after_target') for e in events}
    if merge:
        existing = read_switches_manifest(cdir, acq) or []
        merged = {(int(e['fly_id']), int(e['frame'])): e.get('after_target') for e in existing}
        merged.update(norm)
        norm = merged
    out = [{'fly_id': fid, 'frame': fr,
            'after_target': (None if at is None else int(at))}
           for (fid, fr), at in sorted(norm.items())]
    with open(switches_path(cdir, acq), 'w') as f:
        json.dump({'acq': acq, 'events': out}, f, indent=2)
    return out


def write_switches_manifest_from_confirmed(cdir, confirmed):
    """
    Convenience: take the concordance `confirmed` dict
    {(acq, fly_id, frame): after_target} and write per-acq switch manifests.
    Returns {acq: n_events_written}.
    """
    by_acq = {}
    for (acq, fly_id, frame), after in confirmed.items():
        by_acq.setdefault(acq, []).append(
            {'fly_id': int(fly_id), 'frame': int(frame), 'after_target': after})
    written = {}
    for acq, events in by_acq.items():
        out = write_switches_manifest(cdir, acq, events, merge=True)
        written[acq] = len(out)
    return written


# ── Targets manifest I/O ────────────────────────────────────────────────────
def read_targets_manifest(cdir, acq):
    """Return {'action_col': str, 'corrections': {focal_id(str): [[f,b,a],...]}}, or None."""
    fp = targets_path(cdir, acq)
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        m = json.load(f)
    return {'action_col': m.get('action_col', 'courtship'),
            'corrections': m.get('corrections', {})}


def write_targets_manifest(cdir, acq, corrections_by_focal, action_col='courtship'):
    """
    corrections_by_focal: {focal_id: [(switch_frame, before_tgt, after_tgt), ...]}.
    Keys are coerced to str, values to JSON-safe ints/None. Overwrites any
    existing targets manifest for this acq (review is a full re-decision).
    """
    os.makedirs(cdir, exist_ok=True)

    def _i(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)

    corr = {str(int(focal)): [[int(sw), _i(b), _i(a)] for (sw, b, a) in corrs]
            for focal, corrs in corrections_by_focal.items() if corrs}
    with open(targets_path(cdir, acq), 'w') as f:
        json.dump({'acq': acq, 'action_col': action_col, 'corrections': corr},
                  f, indent=2)
    return corr


# ── Pure apply (manifest → df) ──────────────────────────────────────────────
def apply_switch_events(df, events):
    """
    Set switching = fly_id at each confirmed event (frame, fly), and snapshot the
    pre-overlay switching column into 'switching_manual_only' (once).

    The snapshot is a plain copy of the loaded switching column: in this pipeline
    reviewed_mats is always rebuilt from a fresh processed_mats parquet whose
    switching is pure-manual (confirmations are NOT written back to actions.mat),
    so the snapshot is genuinely manual-only. A frame that is both a manual switch
    and a confirmed switch keeps its manual record in switching_manual_only.

    Modifies df in place; returns it.
    """
    if not events:
        return df
    if 'switching' not in df.columns:
        df['switching'] = -1

    if 'switching_manual_only' not in df.columns:
        df['switching_manual_only'] = df['switching'].copy()

    for e in events:
        fid, fr = int(e['fly_id']), int(e['frame'])
        df.loc[(df['id'] == fid) & (df['frame'] == fr), 'switching'] = fid
    return df


def apply_target_corrections(df, corrections_by_focal, action_col='courtship'):
    """
    Apply post-switch target reassignments. Preserves the pre-correction target
    in '{action_col}_target_pre_correction' (set once, before applying).
    Modifies df in place; returns it.
    """
    if not corrections_by_focal:
        return df
    target_col = f'{action_col}_target'
    pre_col    = f'{target_col}_pre_correction'
    if target_col in df.columns and pre_col not in df.columns:
        df[pre_col] = df[target_col].copy()

    for focal, corrs in corrections_by_focal.items():
        tuples = [tuple(c) for c in corrs]
        tutil.apply_switch_corrections_to_target(df, int(focal), tuples,
                                                 action_col=action_col)
    return df


def build_reviewed_df(df, cdir, acq, action_col='courtship'):
    """Return a curated copy of df with switch + target manifests applied."""
    df = df.copy()
    events = read_switches_manifest(cdir, acq)
    if events:
        apply_switch_events(df, events)
    tm = read_targets_manifest(cdir, acq)
    if tm:
        apply_target_corrections(df, tm['corrections'],
                                 action_col=tm.get('action_col', action_col))
    return df


# ── Reconstruction (curated parquet → manifests) for one-time migration ─────
def reconstruct_switch_events(df):
    """
    Recover confirmed FP switches from a curated parquet as the frames where
    switching differs from switching_manual_only (i.e. confirmations added on
    top of the manual annotation). Returns list of {fly_id, frame, after_target}.
    If switching_manual_only is absent, returns [] (no confirmations to recover).
    """
    if 'switching' not in df.columns or 'switching_manual_only' not in df.columns:
        return []
    added = df[(df['switching'] != df['switching_manual_only']) &
               (df['switching'] != -1)].drop_duplicates(['frame', 'id'])
    events = []
    tgt_col = 'courtship_target'
    for _, row in added.iterrows():
        fid = int(row['switching'])
        fr  = int(row['frame'])
        after = None
        if tgt_col in df.columns:
            sel = df[(df['id'] == fid) & (df['frame'] == fr)][tgt_col]
            if len(sel) and sel.iloc[0] != -1 and not pd.isna(sel.iloc[0]):
                after = int(sel.iloc[0])
        events.append({'fly_id': fid, 'frame': fr, 'after_target': after})
    return events


def reconstruct_target_corrections(df, action_col='courtship'):
    """
    Recover (switch_frame, before_tgt, after_tgt) corrections per focal fly from
    a curated parquet, such that re-applying them to the pre-correction target
    reproduces the curated target. Reads segment target values straight from the
    corrected '{action_col}_target' at bout/switch boundaries.

    Requires '{action_col}_target_pre_correction' to be present (the marker that
    this parquet was target-reviewed); returns {} otherwise.
    """
    target_col  = f'{action_col}_target'
    pre_col     = f'{target_col}_pre_correction'
    boutnum_col = f'{action_col}_boutnum'
    if pre_col not in df.columns or 'switching' not in df.columns:
        return {}

    out = {}
    sw_all = df[(df['switching'] != -1) & df['switching'].notna()]
    for focal in sorted(sw_all['switching'].unique().astype(int)):
        cmask = (df['id'] == focal) & (df[action_col] == focal)
        cdf = df[cmask].drop_duplicates('frame').sort_values('frame')
        if cdf.empty:
            continue
        sw_frames = sorted(df[df['switching'] == focal]
                           .drop_duplicates('frame')['frame'].astype(int).tolist())
        if not sw_frames:
            continue

        tgt = dict(zip(cdf['frame'].astype(int), cdf[target_col]))
        cframes = sorted(tgt)

        def _tgt_at_or_after(f):
            later = [c for c in cframes if c >= f]
            return tgt[later[0]] if later else None

        def _tgt_before(f):
            earlier = [c for c in cframes if c < f]
            return tgt[earlier[-1]] if earlier else None

        groups = (cdf.groupby(boutnum_col) if boutnum_col in cdf.columns
                  else [(0, cdf)])
        corr = []
        for _, b in groups:
            bf = sorted(b['frame'].astype(int).tolist())
            bsw = [f for f in sw_frames if bf[0] <= f <= bf[-1]]
            if not bsw:
                continue
            before0 = _tgt_before(bsw[0])
            for i, swf in enumerate(bsw):
                after = _tgt_at_or_after(swf)
                before = before0 if i == 0 else _tgt_at_or_after(bsw[i - 1])
                corr.append((swf,
                             None if before is None or pd.isna(before) else int(before),
                             None if after is None or pd.isna(after) else int(after)))
        if corr:
            out[focal] = corr
    return out