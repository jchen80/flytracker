"""Headline analysis: fraction of time the fly chases / orients toward the outer
dot, as a function of species and LED intensity.

The JAABA classifiers are already outer-dot-specific (chasing_outerdot,
orienting_outerdot), so the "fraction of time" is just the mean of the binary
behavior column over the frames in each (species, LED-intensity) group.
"""

import numpy as np
import pandas as pd

from . import led_metadata as lm

DEFAULT_BEHAVIORS = ('chasing_outerdot', 'orienting_outerdot')


def add_chasing_union(df, members=('chasing_innerdot', 'chasing_outerdot'),
                      name='chasing_anydot'):
    """Add a union behavior column = OR of the member binary columns present.

    `chasing_anydot` is 1 on any frame the fly is chasing *either* dot. Only the
    members actually present are OR'd (so it still works before the innerdot
    classifier exists -- it just equals chasing_outerdot then). A frame is NaN
    only if every present member is NaN. Returns a copy; no-op if no members exist.
    """
    present = [c for c in members if c in df.columns]
    if not present:
        print(f"[warn] none of {members} present; '{name}' not added")
        return df
    df = df.copy()
    union = df[present].astype(float).fillna(0).max(axis=1)
    union[df[present].isna().all(axis=1)] = np.nan
    df[name] = union
    print(f"added '{name}' = union of {present}")
    return df


def _bridge_bouts(mask, max_gap_frames):
    """Fill False gaps shorter than max_gap_frames between True runs (on a copy).

    Links behavior bouts separated by a brief gap into one bout -- e.g. the fly
    orienting for a moment between two chasing segments is still effectively one
    chase. max_gap_frames < 1 leaves the mask unchanged.
    """
    m = np.asarray(mask, dtype=bool).copy()
    on = np.flatnonzero(m)
    if on.size < 2 or max_gap_frames < 1:
        return m
    for a, b in zip(on[:-1], on[1:]):
        if 0 < (b - a - 1) < max_gap_frames:        # gap shorter than the threshold
            m[a + 1:b] = True
    return m


def _true_runs(mask):
    """(start_idx, end_idx) inclusive ranges of each True run in a boolean array."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        ends = ends + [len(m) - 1]
    return list(zip(starts, ends))


def _contiguous_index_runs(frames):
    """(lo, hi) array-index slices of maximal consecutive (step-1) runs in sorted frames."""
    if len(frames) == 0:
        return []
    brk = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(frames)]])
    return list(zip(starts, ends))


def chasing_bouts(df, bridge_sec=1.0, members=('chasing_innerdot', 'chasing_outerdot')):
    """One row per (bridged) chasing-either-dot bout, for bout-duration analysis.

    Within each (species, assay, LED block) the chasing-either-dot mask has gaps
    shorter than `bridge_sec` linked into a single bout (the fly may orient briefly
    between chasing segments). Bouts are found inside contiguous frame runs only, so
    they never span an LED-block boundary or a frame gap. Only valid-LED frames are
    used; bridge_sec=0 disables linking.

    Returns a long DataFrame:
        species, assay, led_intensity, start_frame, end_frame, n_frames, dur_sec,
        frac_chasing  (chasing frames / bout length; < 1 when a gap was bridged).
    """
    df = df[lm.valid_led(df)]
    present = [c for c in members if c in df.columns]
    if not present:
        print(f"[warn] none of {members} present; no chasing bouts")
        return pd.DataFrame()
    rows = []
    for (sp, assay, inten), g in df.groupby(['species', 'assay', 'led_intensity']):
        g = g.sort_values('frame')
        frames = g['frame'].to_numpy()
        fps = float(g['fps'].iloc[0])
        chasing = (g[present].astype(float).fillna(0).to_numpy() > 0).any(axis=1)
        bridge_frames = int(round(bridge_sec * fps))
        for lo, hi in _contiguous_index_runs(frames):       # split at frame gaps
            seg_frames, seg_chase = frames[lo:hi], chasing[lo:hi]
            for bs, be in _true_runs(_bridge_bouts(seg_chase, bridge_frames)):
                n = be - bs + 1
                rows.append({
                    'species': sp, 'assay': assay, 'led_intensity': inten,
                    'start_frame': int(seg_frames[bs]), 'end_frame': int(seg_frames[be]),
                    'n_frames': int(n), 'dur_sec': n / fps,
                    'frac_chasing': float(seg_chase[bs:be + 1].mean()),
                })
    return pd.DataFrame(rows)


def chasing_bout_summary(bouts, short_sec=0.25):
    """Per (species, LED intensity) summary of a chasing_bouts() table.

    Columns: n_bouts, n_assays, median_dur_sec, mean_dur_sec, frac_short
    (fraction of *bouts* shorter than `short_sec`), frac_short_frames (fraction of
    *chasing frames* in bouts shorter than short_sec -- the time-weighted version,
    what actually matters for contamination), total_chasing_sec. Useful to spot
    blocks (e.g. 0%) dominated by very short bouts.
    """
    if bouts.empty:
        return bouts
    def _agg(g):
        cframes = g['n_frames'] * g['frac_chasing']        # chasing frames per bout
        short = g['dur_sec'] < short_sec
        tot = cframes.sum()
        return pd.Series({
            'n_bouts': len(g),
            'n_assays': g['assay'].nunique(),
            'median_dur_sec': g['dur_sec'].median(),
            'mean_dur_sec': g['dur_sec'].mean(),
            'frac_short': float(short.mean()),
            'frac_short_frames': float(cframes[short].sum() / tot) if tot > 0 else 0.0,
            # chasing-only seconds (exclude bridged gaps): dur_sec weighted by frac_chasing
            'total_chasing_sec': float((g['dur_sec'] * g['frac_chasing']).sum()),
        })
    return (bouts.groupby(['species', 'led_intensity']).apply(_agg).reset_index())


def _valid(df, hue=None):
    """Drop frames with no assigned LED block (e.g. pre-trigger); also no hue if hue given.

    `hue` is an optional extra grouping column (e.g. 'stim_speed' or
    'target_direction'); frames where it is NaN/None are dropped.
    """
    out = df[lm.valid_led(df)]
    if hue:
        if hue not in out.columns:
            raise KeyError(f"hue '{hue}' is not a column; rebuild the cache with the "
                           f"matching build_assays flag (--stim-speed / --target-direction)")
        out = out[out[hue].notna()]
    return out


def fraction_per_assay(df, behaviors=DEFAULT_BEHAVIORS, hue=None):
    """Per-assay fraction of frames engaged, by (species, assay, led_intensity[, hue]).

    Returns a long DataFrame: species, assay, led_intensity[, hue], behavior,
    fraction, n_frames. Per-assay fractions are the unit for error bars / scatter.
    `hue` adds an extra grouping column (e.g. 'stim_speed', 'target_direction').
    """
    df = _valid(df, hue)
    behaviors = [b for b in behaviors if b in df.columns]
    keys = ['species', 'assay', 'led_intensity'] + ([hue] if hue else [])
    rows = []
    for key_vals, g in df.groupby(keys):
        base = dict(zip(keys, key_vals))
        for b in behaviors:
            rows.append({**base, 'behavior': b,
                         'fraction': float(g[b].mean()), 'n_frames': len(g)})
    return pd.DataFrame(rows)


def fraction_chasing_orienting(df, behaviors=DEFAULT_BEHAVIORS, hue=None):
    """Group-level fraction of time engaged, by (species, led_intensity[, hue], behavior).

    Aggregates the per-assay fractions (mean across assays, with sem and n) so
    each assay counts equally regardless of length. `hue` adds an extra grouping
    column (e.g. 'stim_speed', 'target_direction').
    """
    per = fraction_per_assay(df, behaviors, hue)
    if per.empty:
        return per
    keys = ['species', 'led_intensity'] + ([hue] if hue else []) + ['behavior']
    out = (per.groupby(keys)['fraction']
              .agg(mean='mean', sem='sem', n_assays='count')
              .reset_index())
    return out
