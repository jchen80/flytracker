"""Analyses of annotated target switches (from the FlyTracker actions.mat, merged
into the parquet via build_assays --switches as `switching` + `switch_target`).

Analyses:
  - switch rate normalized to courtship time (of either dot), by species x LED
    intensity (switch_rate_*).
  - old/new target egocentric vectors at the switch (switch_vectors).
  - per-switch windows of new/old target ego trajectories + retinal FOV motion
    and peri-switch metrics (switch_target_trajectories).
"""

import numpy as np
import pandas as pd

from . import led_metadata as lm

# behaviors that count as courting *either* dot (whichever columns are present)
COURTSHIP_BEHAVIORS = ('chasing_innerdot', 'chasing_outerdot',
                       'orienting_innerdot', 'orienting_outerdot')


def courtship_mask(df, behaviors=None):
    """Boolean Series: courting either dot on each frame (OR of present behaviors)."""
    behaviors = behaviors or [c for c in COURTSHIP_BEHAVIORS if c in df.columns]
    if not behaviors:
        raise KeyError("no courtship behavior columns present (chasing_/orienting_*dot); "
                       "rebuild the cache with the JAABA scores")
    return (df[behaviors].astype(float).fillna(0) > 0).any(axis=1)


def _require_switches(df):
    if 'switching' not in df.columns:
        raise KeyError("no 'switching' column; rebuild the cache with build_assays --switches")


def old_abs_theta_deg(row):
    """|angle to the OLD (un-switched-to) target| in degrees at a switch row, or nan.

    The old target is the dot the fly was NOT switching to (the opposite of
    `switch_target`). Used to bin switches by how well-oriented the fly was to the
    target it abandoned.
    """
    new = row.get('switch_target')
    if new not in ('inner', 'outer'):
        return np.nan
    old = 'outer' if new == 'inner' else 'inner'
    oa = row.get(f'absangle_to_{old}dot')
    if oa is None or not np.isfinite(oa):
        oa = row.get(f'angle_to_{old}dot')
        oa = abs(oa) if oa is not None and np.isfinite(oa) else np.nan
    return np.degrees(float(oa)) if np.isfinite(oa) else np.nan


def old_target_motion(row):
    """Signed retinal FOV bearing-rate (deg/s) of the OLD target at a switch row:
    + progressive / - regressive, or nan. Needs the target_ang_vel_fov_*_signed_deg
    columns (baked by build_assay_df)."""
    new = row.get('switch_target')
    if new not in ('inner', 'outer'):
        return np.nan
    old = 'outer' if new == 'inner' else 'inner'
    v = row.get(f'target_ang_vel_fov_{old}dot_signed_deg')
    return float(v) if v is not None and np.isfinite(v) else np.nan


def _switch_event_passes(row, old_theta_range=None, old_motion=None):
    """True if a switch row passes the old-target filters (each None = no filter):

    old_theta_range -- (lo, hi): old target's |theta error| in [lo, hi) degrees.
    old_motion      -- 'progressive' (old-target FOV bearing-rate > 0) or
                       'regressive' (< 0).
    """
    if old_theta_range is not None:
        oad = old_abs_theta_deg(row)
        if not (np.isfinite(oad) and old_theta_range[0] <= oad < old_theta_range[1]):
            return False
    if old_motion is not None:
        m = old_target_motion(row)
        if not np.isfinite(m):
            return False
        if old_motion == 'progressive' and not m > 0:
            return False
        if old_motion == 'regressive' and not m < 0:
            return False
    return True


def switch_rate_per_assay(df, hue=None, courtship_behaviors=None, per_minute=True):
    """Per-assay switch rate = switches / courtship-time, by (species, assay,
    led_intensity[, hue]).

    Returns a long DataFrame with n_switches, courtship_sec, switch_rate (per
    minute of courtship by default, else per second). Rows with zero courtship
    time get switch_rate = NaN.
    """
    _require_switches(df)
    d = df[lm.valid_led(df)].copy()
    if hue:
        d = d[d[hue].notna()]
    d['_court'] = courtship_mask(d, courtship_behaviors).to_numpy().astype(int)
    keys = ['species', 'assay', 'led_intensity'] + ([hue] if hue else [])
    scale = 60.0 if per_minute else 1.0
    rows = []
    for kv, g in d.groupby(keys):
        base = dict(zip(keys, kv))
        nsw = int(g['switching'].sum())
        court_sec = g['_court'].sum() / float(g['fps'].iloc[0])
        rate = nsw / court_sec * scale if court_sec > 0 else np.nan
        rows.append({**base, 'n_switches': nsw, 'courtship_sec': court_sec,
                     'switch_rate': rate})
    return pd.DataFrame(rows)


def switch_rate(df, hue=None, courtship_behaviors=None, per_minute=True):
    """Group-level switch rate by (species, led_intensity[, hue]).

    Aggregates the per-assay rates (mean +/- sem across assays). Returns
    (summary, per_assay).
    """
    per = switch_rate_per_assay(df, hue, courtship_behaviors, per_minute)
    keys = ['species', 'led_intensity'] + ([hue] if hue else [])
    summary = (per.groupby(keys)['switch_rate']
                  .agg(mean='mean', sem='sem', n_assays='count').reset_index())
    return summary, per


def switch_vectors(df, groupby=('species',), old_theta_range=None, old_motion=None):
    """Old + new target egocentric vectors at each switch, in the format triad's
    `putil.switch_plots.plot_switch_vectors_across_assays` expects.

    For each switch frame the NEW target is the post-switch dot (switch_target) and
    the OLD target is the other dot. Positions are the fly-egocentric coordinates
    (focal fly at origin facing +x), in PIXELS (the triad plotter divides by ppm):
        x =  dist_to_{dot} * cos(angle_to_{dot})    (forward, +x = ahead)
        y = -dist_to_{dot} * sin(angle_to_{dot})    (lateral, +y = focal's left)
    The lateral sign is NEGATED because the prep computes `angle_to` in image
    coords (y down) while triad's egocentric frame is y-up (it negates ori; see
    relative_metrics.do_transformations_on_df). Without the flip new/old target
    left/right would be mirrored vs the triad figures.

    Returns (vector_dfs, ppm_dict): dicts keyed by the `groupby` label (e.g.
    'Dyak', or 'Dyak_40' when grouping by species+led_intensity). Each vector df
    has columns acquisition/id/frame/new_target/old_target/new_x/new_y/old_x/old_y.
    """
    _require_switches(df)
    sw = df[(df['switching'] == 1) & lm.valid_led(df)]   # ignore switches in >99% (invalid) blocks
    rows = []
    for _, r in sw.iterrows():
        new = r['switch_target']
        if new not in ('inner', 'outer'):
            continue
        old = 'outer' if new == 'inner' else 'inner'
        nd, na = r.get(f'dist_to_{new}dot'), r.get(f'angle_to_{new}dot')
        od, oa = r.get(f'dist_to_{old}dot'), r.get(f'angle_to_{old}dot')
        vals = [nd, na, od, oa]
        if any(v is None or not np.isfinite(v) for v in vals):
            continue
        if not _switch_event_passes(r, old_theta_range, old_motion):
            continue
        rows.append({'acquisition': r['assay'], 'id': 1, 'frame': int(r['frame']),
                     'species': r['species'], 'led_intensity': r['led_intensity'],
                     'new_target': new, 'old_target': old,
                     'new_x': nd * np.cos(na), 'new_y': -nd * np.sin(na),
                     'old_x': od * np.cos(oa), 'old_y': -od * np.sin(oa),
                     'px_per_mm': float(r['px_per_mm'])})
    vdf = pd.DataFrame(rows)
    if vdf.empty:
        return {}, {}

    keys = list(groupby)
    # zero-pad numeric key columns (e.g. led_intensity) so the triad plotter's
    # string-sort of the column labels follows magnitude: species first, then
    # ascending intensity (otherwise 'Dmel_10' sorts before 'Dmel_2').
    pad = {}
    for k in keys:
        if np.issubdtype(vdf[k].dtype, np.number):
            mx = int(np.nanmax(vdf[k].to_numpy())) if len(vdf) else 0
            pad[k] = max(2, len(str(mx)))

    def _lbl(k, v):
        return f'{int(round(float(v))):0{pad[k]}d}' if k in pad else str(v)

    vector_dfs, ppm_dict = {}, {}
    for kv, g in vdf.groupby(keys):
        vals = kv if isinstance(kv, tuple) else (kv,)
        label = '_'.join(_lbl(k, v) for k, v in zip(keys, vals))
        vector_dfs[label] = g.copy()
        ppm_dict[label] = float(g['px_per_mm'].iloc[0])
    return vector_dfs, ppm_dict


def _labeled_groups(df, keys):
    """Split df into ({label: subdf}, {label: ppm}); numeric keys zero-padded so a
    string sort follows magnitude (species first, then ascending intensity)."""
    pad = {}
    for k in keys:
        if np.issubdtype(df[k].dtype, np.number):
            mx = int(np.nanmax(df[k].to_numpy())) if len(df) else 0
            pad[k] = max(2, len(str(mx)))

    def _lbl(k, v):
        return f'{int(round(float(v))):0{pad[k]}d}' if k in pad else str(v)

    groups, ppm = {}, {}
    for kv, g in df.groupby(keys):
        vals = kv if isinstance(kv, tuple) else (kv,)
        label = '_'.join(_lbl(k, v) for k, v in zip(keys, vals))
        groups[label] = g.copy()
        ppm[label] = float(g['px_per_mm'].iloc[0])
    return groups, ppm


def switch_target_trajectories(df, window_sec=1.0, groupby=('species',),
                               old_theta_range=None, old_motion=None):
    """Per-switch windows of new/old target ego trajectories + retinal FOV motion,
    in the format triad's switch_plots tail / positions-colored plotters expect.

    For each switch (valid LED) expands a +/- window_sec/2 window of frames and
    emits, per frame: t_rel (s from switch; <0 before), the ego positions of the
    new and old dot (focal at origin facing +x, +y = focal's left -- same y-up
    convention as switch_vectors), and the signed retinal FOV bearing-rate (deg/s,
    +progressive / -regressive) for each. Returns (traj_assay_dfs, ppm_dict) keyed
    by the `groupby` label. Needs the target_ang_vel_fov_* columns baked by
    build_assay_df (rebuild the cache if missing).
    """
    _require_switches(df)
    need = []
    for d in ('innerdot', 'outerdot'):
        need += [f'dist_to_{d}', f'dist_to_{d}_mm', f'angle_to_{d}', f'absangle_to_{d}',
                 f'target_ang_vel_fov_{d}_signed_deg', f'target_ang_vel_fov_{d}_deg']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing[:3]}...; rebuild the cache with "
                       f"build_assays so the FOV metric is baked")

    parts = []
    for assay, a in df.groupby('assay'):
        a = a.sort_values('frame')
        fps = float(a['fps'].iloc[0])
        half = int(round(window_sec * fps / 2))
        frames = a['frame'].to_numpy()
        sw = a[(a['switching'] == 1) & lm.valid_led(a)]
        for _, ev in sw.iterrows():
            new = ev['switch_target']
            if new not in ('inner', 'outer'):
                continue
            old = 'outer' if new == 'inner' else 'inner'
            if not _switch_event_passes(ev, old_theta_range, old_motion):
                continue
            nd_, od_ = f'{new}dot', f'{old}dot'
            f0 = int(ev['frame'])
            w = a[(frames >= f0 - half) & (frames <= f0 + half)]
            nd, na = w[f'dist_to_{nd_}'].to_numpy(), w[f'angle_to_{nd_}'].to_numpy()
            od, oa = w[f'dist_to_{od_}'].to_numpy(), w[f'angle_to_{od_}'].to_numpy()
            # lab-frame (allo) columns for the sampled-trajectory plotter; y is
            # negated (and theta with it) so the lab frame shares the y-up / +y=left
            # convention of the ego columns above. Positions stay in px (the plotter
            # divides by ppm). Dots carry no orientation, so only the fly gets *_ori.
            parts.append(pd.DataFrame({
                'acquisition': assay, 'id': 1, 'switch_frame': f0,
                'new_target': new, 'old_target': old,
                'frame': w['frame'].to_numpy(),
                'frame_offset': (w['frame'].to_numpy() - f0).astype(int),
                't_rel': (w['frame'].to_numpy() - f0) / fps,
                # peri-switch time-course metrics for new vs old dot
                'new_absangle_deg': np.degrees(w[f'absangle_to_{nd_}'].to_numpy()),
                'old_absangle_deg': np.degrees(w[f'absangle_to_{od_}'].to_numpy()),
                'new_dist_mm': w[f'dist_to_{nd_}_mm'].to_numpy(),
                'old_dist_mm': w[f'dist_to_{od_}_mm'].to_numpy(),
                # head-to-dot distance (falls back to centroid dist if not baked)
                'new_headdist_mm': w[f'headdist_to_{nd_}_mm'].to_numpy()
                if f'headdist_to_{nd_}_mm' in w.columns else w[f'dist_to_{nd_}_mm'].to_numpy(),
                'old_headdist_mm': w[f'headdist_to_{od_}_mm'].to_numpy()
                if f'headdist_to_{od_}_mm' in w.columns else w[f'dist_to_{od_}_mm'].to_numpy(),
                'x_ego': nd * np.cos(na), 'y_ego': -nd * np.sin(na),
                'old_x_ego': od * np.cos(oa), 'old_y_ego': -od * np.sin(oa),
                'focal_x_allo': w['fly_x'].to_numpy(), 'focal_y_allo': -w['fly_y'].to_numpy(),
                'focal_ori_allo': -w['fly_theta'].to_numpy(),
                'new_x_allo': w[f'{nd_}_x'].to_numpy(), 'new_y_allo': -w[f'{nd_}_y'].to_numpy(),
                'old_x_allo': w[f'{od_}_x'].to_numpy(), 'old_y_allo': -w[f'{od_}_y'].to_numpy(),
                'new_target_ang_vel_fov_signed_deg': w[f'target_ang_vel_fov_{nd_}_signed_deg'].to_numpy(),
                'old_target_ang_vel_fov_signed_deg': w[f'target_ang_vel_fov_{od_}_signed_deg'].to_numpy(),
                'new_target_ang_vel_fov_deg': w[f'target_ang_vel_fov_{nd_}_deg'].to_numpy(),
                'old_target_ang_vel_fov_deg': w[f'target_ang_vel_fov_{od_}_deg'].to_numpy(),
                'species': ev['species'], 'led_intensity': ev['led_intensity'],
                'px_per_mm': float(ev['px_per_mm']),
            }))
    if not parts:
        return {}, {}
    return _labeled_groups(pd.concat(parts, ignore_index=True), list(groupby))


def switch_fly_poses(df, use_mm=True):
    """One row per switch event: the fly's lab-frame (allocentric) pose at the switch.

    Columns: species, assay, frame, led_intensity, switch_target ('inner'/'outer',
    the dot switched TO), fly_x, fly_y (lab coords; mm when available, else px),
    fly_theta (rad, image convention), and -- for BOTH the new (switched-to) and old
    (abandoned) target -- the |theta error| (deg) and signed FOV motion (deg/s), so
    events can be binned/colored by either target. Only valid-LED switches with a
    defined target are returned.
    """
    _require_switches(df)
    sw = df[(df['switching'] == 1) & lm.valid_led(df)]
    suf = '_mm' if (use_mm and 'fly_x_mm' in df.columns) else ''
    rows = []
    for _, r in sw.iterrows():
        new = r['switch_target']
        if new not in ('inner', 'outer'):
            continue
        nd = f'{new}dot'
        na = r.get(f'absangle_to_{nd}')
        if na is None or not np.isfinite(na):
            a = r.get(f'angle_to_{nd}')
            na = abs(a) if a is not None and np.isfinite(a) else np.nan
        nm = r.get(f'target_ang_vel_fov_{nd}_signed_deg')
        rows.append({
            'species': r['species'], 'assay': r['assay'], 'frame': int(r['frame']),
            'led_intensity': r['led_intensity'], 'switch_target': new,
            'fly_x': float(r[f'fly_x{suf}']), 'fly_y': float(r[f'fly_y{suf}']),
            'fly_theta': float(r['fly_theta']),
            'old_abs_theta_deg': old_abs_theta_deg(r),
            'old_motion_signed': old_target_motion(r),
            'new_abs_theta_deg': np.degrees(na) if np.isfinite(na) else np.nan,
            'new_motion_signed': float(nm) if nm is not None and np.isfinite(nm) else np.nan,
        })
    return pd.DataFrame(rows)
