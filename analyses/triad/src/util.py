import os
import re
import glob
from collections import defaultdict

import os
import re
import glob
from collections import defaultdict

import pandas as pd
import numpy as np

def _enforce_min_bout_duration(targets, min_bout_frames):
    '''
    Enforce minimum bout duration on a sequence of target assignments within
    a single courtship bout. Short target runs are backward-filled with the
    previous stable value. Short runs at the very start are flagged as -2 and
    forward-filled with the first stable target found later in the sequence
    (or -1 if no stable target exists). Call this per-boutnum so that
    forward/backward fill cannot cross bout boundaries.

    NaN values (frames where orientation was unavailable) are left untouched —
    they pass through as NaN and are not subject to min-bout enforcement.

    Arguments:
        targets         -- np.array of float target ids (may contain NaN)
        min_bout_frames -- minimum number of frames a target run must span

    Returns:
        enforced -- np.array of float target ids with short runs removed.
    '''
    enforced = targets.copy()   # keep as float; NaN-safe
    n = len(enforced)
    i = 0

    while i < n:
        current = enforced[i]

        if np.isnan(current):
            # NaN frames pass through unchanged — skip the whole NaN run
            j = i
            while j < n and np.isnan(enforced[j]):
                j += 1
            i = j
            continue

        j = i
        while j < n and not np.isnan(enforced[j]) and enforced[j] == current:
            j += 1

        bout_len = j - i
        if bout_len < min_bout_frames:
            if i > 0:
                enforced[i:j] = enforced[i - 1]
            else:
                enforced[i:j] = -2

        i = j

    # forward fill any flagged frames at start with first stable target
    neg2_mask = (enforced == -2)
    if neg2_mask.any():
        first_stable = next(
            (v for v in enforced if not np.isnan(v) and v not in (-1, -2)), None)
        enforced[neg2_mask] = first_stable if first_stable is not None else -1

    return enforced

def assign_target_orientation(df, action_col='courtship',
                               fps=60, delta_theta_deg=15.0,
                               min_bout_sec=1.0):
    '''
    Assign the most likely target fly for each frame of a courtship bout.
    Uses abs_theta_error_deg (orientation) as the primary signal, with
    dist_to_other as a tiebreaker when the better-oriented fly is farther away.
    Enforces a minimum target bout duration to prevent transient misassignments
    during reorientation lags.

    Assumes the following columns are pre-computed in df:
        - abs_theta_error_deg: absolute theta error in degrees
        - dist_to_other: distance to the other fly in pixels
        - pair: string of format 'flyid1_flyid2'
        - frame: frame index
        - id: fly id

    Arguments:
        df         -- single-acquisition transformed tracks df with pair-wise
                      abs_theta_error_deg, dist_to_other, and action columns

    Keyword Arguments:
        action_col      -- name of action column (default: 'courtship')
        fps             -- frames per second (default: 60)
        delta_theta_deg -- orientation advantage threshold in degrees;
                           if the better-oriented fly is farther away but the
                           orientation difference is less than this, prefer the
                           closer fly instead (default: 15.0)
        min_bout_sec    -- minimum duration in seconds a target assignment must
                           persist before a switch is accepted (default: 1.0)

    Returns:
        df -- with new columns:
              '{action_col}_target' (-1 if not courting, else target fly id)
              '{action_col}_switch' (-1 if not courting, 0 if courting with no switch,
                                     1 at the first frame of each new target run within a bout)
    '''
    min_bout_frames = int(min_bout_sec * fps)
    fly_ids = sorted(df['id'].unique().tolist())
    target_col = f'{action_col}_target'
    switch_col = f'{action_col}_auto_switch'
    df[target_col] = -1

    for acting_fly_id in fly_ids:
        acting_mask = (df['id'] == acting_fly_id) & (df[action_col] == acting_fly_id)
        if acting_mask.sum() == 0:
            continue

        acting_df = df[acting_mask].copy()

        acting_df['_cand_id'] = (acting_df['pair']
                                  .str.split('_')
                                  .apply(lambda x: [int(i) for i in x])
                                  .apply(lambda x: [i for i in x if i != acting_fly_id][0]))

        # per frame: apply orientation + distance tiebreaker logic
        best_targets = []
        for frame_idx, frame_group in acting_df.groupby('frame'):
            if len(frame_group) < 2:
                # Only one candidate — require valid theta error; NaN ori gives
                # no reliable orientation signal so leave this frame unassigned (-1).
                if pd.isna(frame_group.iloc[0]['abs_theta_error_deg']):
                    best_targets.append((frame_idx, np.nan))
                else:
                    best_targets.append((frame_idx, frame_group.iloc[0]['_cand_id']))
                continue

            frame_group = frame_group.sort_values('abs_theta_error_deg')
            primary = frame_group.iloc[0]
            other = frame_group.iloc[1]

            primary_id = primary['_cand_id']
            primary_theta = primary['abs_theta_error_deg']
            primary_dist = primary['dist_to_other']

            other_id = other['_cand_id']
            other_theta = other['abs_theta_error_deg']
            other_dist = other['dist_to_other']

            # If the best candidate has NaN theta error, orientation is unavailable
            # for this frame — leave unassigned rather than picking arbitrarily.
            if pd.isna(primary_theta):
                best_targets.append((frame_idx, np.nan))
                continue

            # distance tiebreaker: if better-oriented fly is farther,
            # only keep it if orientation advantage exceeds delta_theta
            if primary_dist > other_dist:
                if (other_theta - primary_theta) < delta_theta_deg:
                    best_targets.append((frame_idx, other_id))
                else:
                    best_targets.append((frame_idx, primary_id))
            else:
                best_targets.append((frame_idx, primary_id))

        if len(best_targets) == 0:
            continue

        frame_to_raw = dict(best_targets)
        boutnum_col_local = f'{action_col}_boutnum'
        frame_to_target = {}

        if boutnum_col_local in acting_df.columns:
            # enforce per boutnum so forward/backward fill cannot cross bout boundaries
            for _, bout_group in acting_df.drop_duplicates('frame').groupby(boutnum_col_local):
                bout_frames = sorted(bout_group['frame'].tolist())
                bout_raw = np.array([frame_to_raw[f] for f in bout_frames], dtype=float)
                enforced = _enforce_min_bout_duration(bout_raw, min_bout_frames)
                frame_to_target.update(zip(bout_frames,
                                           [int(v) if not np.isnan(v) else -1 for v in enforced]))
        else:
            frame_indices, raw_targets = zip(*best_targets)
            enforced = _enforce_min_bout_duration(np.array(raw_targets, dtype=float), min_bout_frames)
            frame_to_target = dict(zip(frame_indices,
                                       [int(v) if not np.isnan(v) else -1 for v in enforced]))

        df.loc[acting_mask, target_col] = df.loc[acting_mask, 'frame'].map(frame_to_target)

    # ── Switch detection ────────────────────────────────────────────────────
    # A switch is the first frame of a new target run within the same bout.
    # Requires a boutnum column to avoid flagging the start of a new bout.
    boutnum_col = f'{action_col}_boutnum'
    df[switch_col] = -1

    if boutnum_col not in df.columns:
        print(f"  '{boutnum_col}' not found — skipping switch detection.")
        return df

    for acting_fly_id in fly_ids:
        acting_mask = (df['id'] == acting_fly_id) & (df[action_col] == acting_fly_id)
        if acting_mask.sum() == 0:
            continue

        # Default all acting frames for this fly to 0 (no switch)
        df.loc[acting_mask, switch_col] = 0

        acting_df = df[acting_mask].sort_values('frame')
        # bout numbers are assigned per acting fly
        for _, bout_df in acting_df.groupby(boutnum_col):
            bout_df = bout_df.sort_values('frame')
            targets = bout_df[target_col].values
            indices = bout_df.index.values

            for i in range(1, len(targets)):
                prev, curr = targets[i - 1], targets[i]
                # valid switch: both endpoints are assigned and target changed
                if curr != -1 and prev != -1 and curr != prev:
                    df.loc[indices[i], switch_col] = 1

    n_switches = df[df[switch_col] == 1].drop_duplicates(subset=['frame', 'id']).shape[0]
    print(f"  Detected {n_switches} switch events for '{action_col}'.")
    return df

def assign_target_nearest(df, action_col):
    '''
    For each frame of an action bout, assign the target fly as whichever
    candidate fly is closest (minimum dist_to_other).
    Vectorized implementation.

    Arguments:
        df         -- single-acquisition transformed tracks df with pair-wise
                      dist_to_other and action columns
        action_col -- name of action column to assign targets for

    Returns:
        df -- with new column '{action_col}_target' (-1 if not acting, else target fly id)
    '''
    fly_ids = sorted(df['id'].unique().tolist())
    target_col = f'{action_col}_target'
    df[target_col] = -1

    for acting_fly_id in fly_ids:
        acting_mask = (df['id'] == acting_fly_id) & (df[action_col] == acting_fly_id)
        if acting_mask.sum() == 0:
            continue

        acting_df = df[acting_mask].copy()

        # Extract candidate id from pair string vectorized
        acting_df['_cand_id'] = (acting_df['pair']
                                  .str.split('_')
                                  .apply(lambda x: [int(i) for i in x])
                                  .apply(lambda x: [i for i in x if i != acting_fly_id][0]))

        # For each frame pick candidate with minimum dist_to_other
        best = (acting_df
                .sort_values('dist_to_other')
                .groupby('frame')['_cand_id']
                .first()
                .reset_index()
                .rename(columns={'_cand_id': 'best_cand'}))

        frame_to_target = dict(zip(best['frame'], best['best_cand']))
        df.loc[acting_mask, target_col] = df.loc[acting_mask, 'frame'].map(frame_to_target)

    return df


def _resolve_switch_source(df, action_col, switch_source):
    '''
    Resolve switch_source ('auto', 'manual', or 'prefer_manual') to 'manual' or
    'auto' based on what columns are actually available in df.

    Returns 'manual', 'auto', or None if the resolved source is unavailable.
    '''
    switch_col = f'{action_col}_auto_switch'
    has_manual = ('switching' in df.columns and
                  (df['switching'].notna() & (df['switching'] != -1)).any())
    has_auto = switch_col in df.columns

    if switch_source == 'prefer_manual':
        resolved = 'manual' if has_manual else 'auto'
    elif switch_source in ('manual', 'auto'):
        resolved = switch_source
    else:
        raise ValueError(f"switch_source must be 'auto', 'manual', or 'prefer_manual'; "
                         f"got {switch_source!r}")

    if resolved == 'manual' and not has_manual:
        return None
    if resolved == 'auto' and not has_auto:
        return None
    return resolved


def filter_to_switch_frames(df, action_col='courtship', switch_source='prefer_manual'):
    '''
    Filter df to only the first frame of each target switch event, keeping only
    the pair row where the other fly is the new target.

    Arguments:
        df         -- processed tracks dataframe with switch and target columns
        action_col -- action column prefix (default: 'courtship')

    Keyword Arguments:
        switch_source -- which switch annotation to use: 'auto' ({action_col}_auto_switch),
                         'manual' (switching column), or 'prefer_manual' (manual if present,
                         else auto). Default: 'prefer_manual'.

    Returns:
        filtered df (empty if required columns are absent)
    '''
    target_col = f'{action_col}_target'

    if target_col not in df.columns:
        print(f"  filter_to_switch_frames: '{target_col}' not found.")
        return df.iloc[0:0].copy()

    resolved = _resolve_switch_source(df, action_col, switch_source)
    if resolved is None:
        print(f"  filter_to_switch_frames: no switch column available for "
              f"switch_source={switch_source!r}.")
        return df.iloc[0:0].copy()

    if resolved == 'manual':
        switch_mask = (df['switching'] == df['id']) & (df['switching'] != -1)
    else:
        switch_mask = df[f'{action_col}_auto_switch'] == 1

    # Rows where a switch occurred; extract new target and pair partner
    acting = df[switch_mask][['acquisition', 'frame', 'id', target_col, 'pair']].copy()

    # Vectorized: split pair string and pick the element that isn't the acting fly
    pair_split = acting['pair'].str.split('_')
    fly0 = pair_split.str[0].astype(int)
    fly1 = pair_split.str[1].astype(int)
    acting['_other'] = np.where(fly0 == acting['id'].values, fly1, fly0)

    # Keep only the pair row where the partner IS the new target, then merge
    # on (acquisition, frame, id, pair) so we return exactly that one pair row
    # rather than all pair rows for the acting fly at the switch frame.
    switch_rows = acting[acting['_other'] == acting[target_col]][
        ['acquisition', 'frame', 'id', 'pair']
    ].drop_duplicates()

    result = df.merge(switch_rows, on=['acquisition', 'frame', 'id', 'pair'], how='inner')

    # TEMP: drop switch rows where ego-centric positions are NaN (NaN orientation
    # at the switch frame in already-processed data). Remove once all data is
    # reprocessed with the fixed assign_target_orientation that skips NaN-ori frames.
    if 'targ_rel_pos_x' in result.columns:
        n_before = len(result)
        result = result.dropna(subset=['targ_rel_pos_x', 'targ_rel_pos_y'])
        n_dropped = n_before - len(result)
        if n_dropped > 0:
            print(f"  filter_to_switch_frames: dropped {n_dropped} switch rows with NaN "
                  f"ego-centric positions (NaN orientation at switch frame).")
    return result


def get_switch_frame_vectors(df, action_col='courtship', switch_source='prefer_manual'):
    '''
    For each courtship switch event return the ego-centric positions of both
    the old target (the fly being courted immediately before the switch) and
    the new target (the fly being courted at the switch frame).

    The focal fly sits at the origin in ego-centric coordinates, facing +x.
    targ_rel_pos_x/y from the pair row gives the partner's position in that frame.

    Arguments:
        df         -- processed tracks dataframe (output of load_all_processed_dfs)
        action_col -- action column prefix (default: 'courtship')

    Keyword Arguments:
        switch_source -- which switch annotation to use: 'auto', 'manual',
                         or 'prefer_manual'. Default: 'prefer_manual'.

    Returns:
        DataFrame with columns:
            acquisition, id, frame,
            new_target, new_x, new_y,
            old_target, old_x, old_y,
            plus any metadata columns present (triad_type, assay_type, species)
    '''
    target_col = f'{action_col}_target'

    if target_col not in df.columns:
        print(f"  get_switch_frame_vectors: '{target_col}' not found.")
        return pd.DataFrame()

    resolved = _resolve_switch_source(df, action_col, switch_source)
    if resolved is None:
        print(f"  get_switch_frame_vectors: no switch column available for "
              f"switch_source={switch_source!r}.")
        return pd.DataFrame()

    # One row per (acquisition, frame, id) for acting frames
    acting = (df[df[action_col] == df['id']]
              .drop_duplicates(['acquisition', 'frame', 'id'])
              .sort_values(['acquisition', 'id', 'frame'])
              .copy())

    # Previous frame's target within each (acquisition, fly) group
    acting['_prev_target'] = acting.groupby(['acquisition', 'id'])[target_col].shift(1)

    # Switch events: rows where switch just happened and previous target is known/valid
    if resolved == 'manual':
        switch_mask = (acting['switching'] == acting['id']) & (acting['switching'] != -1)
    else:
        switch_mask = acting[f'{action_col}_auto_switch'] == 1
    switch_events = acting[switch_mask].copy()
    switch_events = switch_events[
        switch_events['_prev_target'].notna() &
        (switch_events['_prev_target'] != -1)
    ][['acquisition', 'id', 'frame', target_col, '_prev_target']].rename(
        columns={target_col: 'new_target', '_prev_target': 'old_target'}
    )

    switch_events['new_target'] = switch_events['new_target'].astype(int)
    switch_events['old_target'] = switch_events['old_target'].astype(int)

    if len(switch_events) == 0:
        return pd.DataFrame()

    # Build a lookup of (acquisition, frame, id, partner) -> targ_rel_pos_x/y + dist_to_other
    pos_cols = ['acquisition', 'frame', 'id', 'pair', 'targ_rel_pos_x', 'targ_rel_pos_y',
                'dist_to_other']
    available = [c for c in pos_cols if c in df.columns]
    df_pos = df[available].copy()
    pair_split = df_pos['pair'].str.split('_')
    fly0 = pair_split.str[0].astype(int)
    fly1 = pair_split.str[1].astype(int)
    df_pos['_partner'] = np.where(fly0 == df_pos['id'].values, fly1, fly0)
    df_pos = df_pos.drop(columns='pair')

    has_dist = 'dist_to_other' in df_pos.columns

    # Join new target positions
    new_rename = {'_partner': '_nt', 'targ_rel_pos_x': 'new_x', 'targ_rel_pos_y': 'new_y'}
    if has_dist:
        new_rename['dist_to_other'] = 'new_dist'
    switch_events = switch_events.merge(
        df_pos.rename(columns=new_rename),
        left_on=['acquisition', 'frame', 'id', 'new_target'],
        right_on=['acquisition', 'frame', 'id', '_nt'],
        how='inner'
    ).drop(columns='_nt')

    # Join old target positions
    old_rename = {'_partner': '_ot', 'targ_rel_pos_x': 'old_x', 'targ_rel_pos_y': 'old_y'}
    if has_dist:
        old_rename['dist_to_other'] = 'old_dist'
    switch_events = switch_events.merge(
        df_pos.rename(columns=old_rename),
        left_on=['acquisition', 'frame', 'id', 'old_target'],
        right_on=['acquisition', 'frame', 'id', '_ot'],
        how='inner'
    ).drop(columns='_ot')

    # Carry over metadata columns if present
    meta_cols = [c for c in ['triad_type', 'assay_type', 'species'] if c in df.columns]
    if meta_cols:
        meta = (df[['acquisition', 'id'] + meta_cols]
                .drop_duplicates(['acquisition', 'id']))
        switch_events = switch_events.merge(meta, on=['acquisition', 'id'], how='left')

    dist_cols = [c for c in ['new_dist', 'old_dist'] if c in switch_events.columns]
    keep = (['acquisition', 'id', 'frame',
             'new_target', 'new_x', 'new_y',
             'old_target', 'old_x', 'old_y'] + dist_cols + meta_cols)
    result = switch_events[keep].reset_index(drop=True)

    # TEMP: drop events where position lookup returned NaN (NaN orientation at
    # switch frame in already-processed data). Remove once all data is reprocessed
    # with the fixed assign_target_orientation that skips NaN-ori frames.
    nan_check = ['new_x', 'new_y', 'old_x', 'old_y']
    n_before = len(result)
    result = result.dropna(subset=nan_check)
    n_dropped = n_before - len(result)
    if n_dropped > 0:
        print(f"  get_switch_frame_vectors: dropped {n_dropped} events with NaN "
              f"positions (NaN orientation at switch frame).")
    return result


def get_courtship_target_fov(df, action_col='courtship'):
    '''
    For every courtship frame (focal fly acting, with an assigned target), return the
    focal's egocentric view of BOTH of its possible targets: the *pursued* target
    (action_col_target, the auto-assigned best-oriented fly) and the *other* (the
    remaining non-focal fly). One row per (acquisition, frame, focal id).

    The focal fly sits at the origin facing +x; targ_rel_pos_x/y from each pair row is
    that partner's egocentric position (pixels, y-up), and theta_error_deg is the signed
    bearing error to that partner. Mirrors get_switch_frame_vectors but for all
    courtship frames and pursued/other instead of new/old.

    Arguments:
        df         -- processed tracks dataframe (load_all_processed_dfs output)
        action_col -- action column prefix (default: 'courtship')

    Returns:
        DataFrame with columns:
            acquisition, id, frame,
            pursued_target, pursued_x, pursued_y, pursued_theta_deg,
            other_target,   other_x,   other_y,   other_theta_deg,
            plus metadata columns present (triad_type, assay_type, species)
        Empty DataFrame if the target column is missing or no courtship frames exist.
    '''
    target_col = f'{action_col}_target'
    if target_col not in df.columns:
        print(f"  get_courtship_target_fov: '{target_col}' not found.")
        return pd.DataFrame()

    # one row per (acquisition, frame, id) acting frame with a valid assigned target
    acting = df[df[action_col] == df['id']].drop_duplicates(['acquisition', 'frame', 'id'])
    ev = acting[['acquisition', 'frame', 'id', target_col]].copy()
    ev['pursued_target'] = pd.to_numeric(ev[target_col], errors='coerce').fillna(-1).astype(int)
    ev = ev[ev['pursued_target'] != -1].drop(columns=target_col)
    if ev.empty:
        return pd.DataFrame()

    # lookup of (acquisition, frame, id, partner) -> ego position + signed theta error
    pos_cols = ['acquisition', 'frame', 'id', 'pair',
                'targ_rel_pos_x', 'targ_rel_pos_y', 'theta_error_deg']
    df_pos = df[[c for c in pos_cols if c in df.columns]].copy()
    pair_split = df_pos['pair'].str.split('_')
    fly0 = pair_split.str[0].astype(int)
    fly1 = pair_split.str[1].astype(int)
    df_pos['_partner'] = np.where(fly0 == df_pos['id'].values, fly1, fly0)
    df_pos = df_pos.drop(columns='pair')

    # pursued target: the partner row whose partner == assigned target
    pur = df_pos.rename(columns={'_partner': '_pt', 'targ_rel_pos_x': 'pursued_x',
                                 'targ_rel_pos_y': 'pursued_y',
                                 'theta_error_deg': 'pursued_theta_deg'})
    ev = ev.merge(pur, left_on=['acquisition', 'frame', 'id', 'pursued_target'],
                  right_on=['acquisition', 'frame', 'id', '_pt'], how='inner').drop(columns='_pt')

    # other target: the remaining partner (!= pursued target) for that focal-frame
    oth = df_pos.rename(columns={'_partner': 'other_target', 'targ_rel_pos_x': 'other_x',
                                 'targ_rel_pos_y': 'other_y',
                                 'theta_error_deg': 'other_theta_deg'})
    ev = ev.merge(oth, on=['acquisition', 'frame', 'id'], how='inner')
    ev = ev[ev['other_target'].astype(int) != ev['pursued_target']]
    ev['other_target'] = ev['other_target'].astype(int)

    meta_cols = [c for c in ['triad_type', 'assay_type', 'species'] if c in df.columns]
    if meta_cols:
        meta = df[['acquisition', 'id'] + meta_cols].drop_duplicates(['acquisition', 'id'])
        ev = ev.merge(meta, on=['acquisition', 'id'], how='left')

    keep = (['acquisition', 'id', 'frame',
             'pursued_target', 'pursued_x', 'pursued_y', 'pursued_theta_deg',
             'other_target', 'other_x', 'other_y', 'other_theta_deg'] + meta_cols)
    result = ev[keep].reset_index(drop=True)
    return result.dropna(subset=['pursued_x', 'pursued_y', 'other_x', 'other_y'])


def get_target_pair_fov(df, sex_map, focal_flies_map=None, action_col='courtship'):
    '''
    For every frame (focal flies only), the focal's signed bearing (theta error, deg) to
    each of its two possible targets, with the targets labeled by SEX -- courtship-
    independent, unlike get_courtship_target_fov's pursued/other. Used to map, over the
    joint (theta_x, theta_y) occupancy space, where courtship/switching of each specific
    target happens.

    Labeling (per focal fly, from sex_map[triad_type]): if its two partners differ in sex
    (e.g. MMF: one M, one F) -> x = female, y = male; if same sex (e.g. MFF: two females)
    -> x = lower-id, y = higher-id.

    Arguments:
        df              -- processed tracks df (one triad_type/assay)
        sex_map         -- {triad_type: {id: 'M'|'F'}}
        focal_flies_map -- {triad_type: [focal ids]}; defaults to all ids if None
        action_col      -- action column prefix (default: 'courtship')

    Returns one row per (acquisition, frame, focal id) with columns:
        acquisition, id, frame,
        theta_x_deg, theta_y_deg   (signed bearing to target x / y),
        court_x, court_y           (1 if focal is courting that target this frame),
        dtheta_x_degs, dtheta_y_degs (signed d(θ-error)/dt to target x / y, deg/s;
                                    present only if theta_error_dt is in df),
        ang_vel_fly_degs           (fly's own angular velocity, deg/s, signed by
                                    rotation direction; present only if ang_vel_fly in df),
        vel_mm_s                   (focal fly's own speed, mm/s; present only if vel in df),
        x_id, y_id, x_label, y_label,
        plus metadata present (triad_type, assay_type, species).
    '''
    if df.empty or 'theta_error_deg' not in df.columns:
        return pd.DataFrame()
    triad_type = df['triad_type'].iloc[0]
    sexes = sex_map.get(triad_type, {})
    all_ids = sorted(int(i) for i in df['id'].unique())
    focal = (focal_flies_map.get(triad_type) if focal_flies_map else None) or all_ids
    target_col = f'{action_col}_target'

    # x/y partner assignment is constant per focal fly (its two partners never change)
    def _assign_xy(focal_id):
        partners = sorted(p for p in all_ids if p != focal_id)
        if len(partners) != 2:
            return None
        fem = [p for p in partners if sexes.get(p) == 'F']
        male = [p for p in partners if sexes.get(p) == 'M']
        if len(fem) == 1 and len(male) == 1:           # mixed-sex pair (MMF)
            return fem[0], male[0], 'female', 'male'
        a, b = partners                                 # same-sex pair (MFF): by id
        base = {'F': 'female', 'M': 'male'}.get(sexes.get(a), 'target')
        return a, b, f'{base} {a}', f'{base} {b}'
    assign = {f: _assign_xy(f) for f in focal if _assign_xy(f) is not None}
    if not assign:
        return pd.DataFrame()

    foc = df[df['id'].isin(assign.keys())].copy()
    pair_split = foc['pair'].str.split('_')
    f0 = pair_split.str[0].astype(int)
    f1 = pair_split.str[1].astype(int)
    foc['_partner'] = np.where(f0 == foc['id'].values, f1, f0)
    if target_col in foc.columns:
        tgt = pd.to_numeric(foc[target_col], errors='coerce').fillna(-1).astype(int)
        foc['_court'] = ((foc[action_col] == foc['id']) & (tgt == foc['_partner'])).astype(int)
    else:
        foc['_court'] = 0
    foc['_x_id'] = foc['id'].map(lambda i: assign[i][0])
    foc['_y_id'] = foc['id'].map(lambda i: assign[i][1])
    has_dt = 'theta_error_dt' in foc.columns       # signed d(θ-error)/dt (rad/s)
    has_avf = 'ang_vel_fly' in foc.columns         # fly's own angular velocity (rad/s)
    has_vel = 'vel' in foc.columns                 # focal fly's own speed (mm/s)
    keep_vals = (['theta_error_deg', '_court'] + (['theta_error_dt'] if has_dt else [])
                 + (['ang_vel_fly'] if has_avf else []) + (['vel'] if has_vel else []))
    longt = foc[['acquisition', 'frame', 'id', '_partner', '_x_id', '_y_id'] + keep_vals
                ].dropna(subset=['theta_error_deg'])

    base = ['acquisition', 'frame', 'id']
    xren = {'theta_error_deg': 'theta_x_deg', '_court': 'court_x'}
    yren = {'theta_error_deg': 'theta_y_deg', '_court': 'court_y'}
    if has_dt:
        xren['theta_error_dt'] = 'dtheta_x'
        yren['theta_error_dt'] = 'dtheta_y'
    if has_avf:                                     # same per focal-frame; carry via x only
        xren['ang_vel_fly'] = '_avf'
    if has_vel:                                     # focal speed; carry via x only
        xren['vel'] = '_vel'
    xcols = (base + ['theta_x_deg', 'court_x'] + (['dtheta_x'] if has_dt else [])
             + (['_avf'] if has_avf else []) + (['_vel'] if has_vel else []))
    ycols = base + ['theta_y_deg', 'court_y'] + (['dtheta_y'] if has_dt else [])
    xt = longt[longt['_partner'] == longt['_x_id']].rename(columns=xren)[xcols]
    yt = longt[longt['_partner'] == longt['_y_id']].rename(columns=yren)[ycols]
    ev = xt.merge(yt, on=base, how='inner')
    if ev.empty:
        return pd.DataFrame()
    if has_dt:                                      # signed bearing rate in deg/s
        ev['dtheta_x_degs'] = np.degrees(ev['dtheta_x'])
        ev['dtheta_y_degs'] = np.degrees(ev['dtheta_y'])
        ev = ev.drop(columns=['dtheta_x', 'dtheta_y'])
    if has_avf:                                    # fly's own angular velocity, deg/s,
        ev['ang_vel_fly_degs'] = np.degrees(ev['_avf'])   # signed by rotation direction
        ev = ev.drop(columns='_avf')
    if has_vel:                                    # focal fly speed (mm/s)
        ev['vel_mm_s'] = ev['_vel']
        ev = ev.drop(columns='_vel')

    ev['x_id'] = ev['id'].map(lambda i: assign[i][0])
    ev['y_id'] = ev['id'].map(lambda i: assign[i][1])
    ev['x_label'] = ev['id'].map(lambda i: assign[i][2])
    ev['y_label'] = ev['id'].map(lambda i: assign[i][3])
    for m in ('triad_type', 'assay_type', 'species'):
        if m in df.columns:
            ev[m] = df[m].iloc[0]
    return ev.reset_index(drop=True)


def find_copulation_frame(df, copulation_col='copulation', boutnum_col='copulation_boutnum'):
    '''
    Return the first frame of the first copulation bout, or None if not found.

    Arguments:
        df -- dataframe with copulation action column and frame column
    '''
    if copulation_col not in df.columns or boutnum_col not in df.columns:
        return None
    cop_rows = df[(df[copulation_col] != -1) & (df[boutnum_col].notna())]
    return int(cop_rows['frame'].min()) if len(cop_rows) > 0 else None


def trim_at_copulation(df, copulation_col='copulation', boutnum_col='copulation_boutnum'):
    '''
    Trim df to exclude all frames after the start of the first copulation bout.
    Returns df unchanged if no copulation is annotated.

    Arguments:
        df -- dataframe with copulation action column and frame column
    '''
    cop_frame = find_copulation_frame(df, copulation_col=copulation_col,
                                      boutnum_col=boutnum_col)
    if cop_frame is None:
        return df
    print(f"First copulation frame: {cop_frame} — trimming all frames after it.")
    return df[df['frame'] <= cop_frame].copy()

def filter_to_target_pairs(df, action_col='courtship'):
    '''
    Keep only rows where the pair partners the focal fly with its assigned
    courtship target.  In those rows target_vel is the target's speed (mm/s).

    Equivalent to putil._filter_to_target_pairs but exposed for notebook use.

    Arguments:
        df         -- processed tracks df with {action_col}_target column
        action_col -- action column prefix (default: 'courtship')

    Returns:
        filtered df (copy); empty if target column is missing or no target assigned
    '''
    target_col = f'{action_col}_target'
    if target_col not in df.columns:
        print(f"  filter_to_target_pairs: '{target_col}' not found.")
        return df.iloc[0:0].copy()

    pair_split = df['pair'].str.split('_')
    pair_fly0 = pair_split.str[0].astype(int)
    pair_fly1 = pair_split.str[1].astype(int)
    fly_id = df['id'].astype(int)
    target_id = pd.to_numeric(df[target_col], errors='coerce').fillna(-1).astype(int)

    has_target = target_id != -1
    pair_matches = (
        ((pair_fly0 == fly_id) & (pair_fly1 == target_id)) |
        ((pair_fly1 == fly_id) & (pair_fly0 == target_id))
    )
    return df[has_target & pair_matches].copy()


def filter_pursuit_frames(df, action_col='courtship',
                          min_vel_mm_s=None, max_vel_mm_s=None,
                          vel_source='target'):
    '''
    Filter to courtship frames where the focal fly is pursuing a moving target.

    Calls filter_to_target_pairs first, then optionally restricts to frames
    where a velocity falls within [min_vel_mm_s, max_vel_mm_s].

    Arguments:
        df         -- processed tracks df with courtship columns
        action_col -- action column prefix (default: 'courtship')

    Keyword Arguments:
        min_vel_mm_s -- minimum speed in mm/s, inclusive (default: None)
        max_vel_mm_s -- maximum speed in mm/s, inclusive (default: None)
        vel_source   -- which fly's speed to filter on: 'target' uses target_vel
                        (assigned target's speed); 'focal' uses vel (focal fly's
                        own speed from FlyTracker features). Default: 'target'.

    Returns:
        filtered df (copy)
    '''
    if vel_source not in ('target', 'focal'):
        raise ValueError(f"vel_source must be 'target' or 'focal', got {vel_source!r}")

    target_df = filter_to_target_pairs(df, action_col=action_col)

    if min_vel_mm_s is None and max_vel_mm_s is None:
        return target_df

    vel_col = 'target_vel' if vel_source == 'target' else 'vel'
    if vel_col not in target_df.columns:
        return target_df

    mask = pd.Series(True, index=target_df.index)
    if min_vel_mm_s is not None:
        mask &= target_df[vel_col] >= min_vel_mm_s
    if max_vel_mm_s is not None:
        mask &= target_df[vel_col] <= max_vel_mm_s
    return target_df[mask].copy()


def get_action_relative_timing(df, action_col='courtship', target_col='mounting attempt',
                                focal_flies_map=None):
    '''
    For each focal fly, compute the relative timing of a target behavior's bout
    onsets within the cumulative sequence of action frames (pseudotime).

    Relative timing = (# action frames at or before the first frame of the target
    bout) / (total action frames for that fly), expressed as a fraction 0–1.

    Arguments:
        df         -- processed tracks dataframe
        action_col -- base action column whose frame sequence defines pseudotime
                      (default: 'courtship')
        target_col -- target behavior column whose bout onsets are timed
                      (default: 'mounting attempt')

    Keyword Arguments:
        focal_flies_map -- dict mapping triad_type to list of focal fly ids;
                           if None all fly ids are used (default: None)

    Returns:
        DataFrame with columns:
            acquisition, id, boutnum,
            first_frame         (first frame of each target bout),
            total_action_frames (total action frames for this fly × acq),
            relative_timing     (fraction 0–1 through action pseudotime)
        plus metadata columns (triad_type, assay_type, species) if present
    '''
    boutnum_col = f'{target_col}_boutnum'
    for col in [action_col, target_col, boutnum_col]:
        if col not in df.columns:
            print(f"  get_action_relative_timing: '{col}' not found — returning empty.")
            return pd.DataFrame()

    meta_cols = [c for c in ['triad_type', 'assay_type', 'species'] if c in df.columns]
    records = []

    for acq, acq_df in df.groupby('acquisition'):
        if focal_flies_map is not None and 'triad_type' in acq_df.columns:
            triad_type = acq_df['triad_type'].iloc[0]
            fly_ids = focal_flies_map.get(triad_type) or sorted(acq_df['id'].unique())
        else:
            fly_ids = sorted(acq_df['id'].unique())

        for fly_id in fly_ids:
            fly_rows = acq_df[acq_df['id'] == fly_id]

            # Sorted unique action frames — defines the pseudotime axis
            court_frames = (fly_rows[fly_rows[action_col] == fly_id]['frame']
                            .drop_duplicates().sort_values().values)
            if len(court_frames) == 0:
                continue

            # First frame of each target bout (deduplicated to frame level first)
            target_rows = (fly_rows[fly_rows[target_col] == fly_id]
                           [['frame', boutnum_col]]
                           .drop_duplicates())
            if len(target_rows) == 0:
                continue

            bout_onsets = (target_rows.groupby(boutnum_col)['frame']
                           .min().reset_index()
                           .rename(columns={'frame': 'first_frame'}))

            meta = {c: fly_rows[c].iloc[0] for c in meta_cols}
            total = len(court_frames)

            for _, row in bout_onsets.iterrows():
                first_frame = int(row['first_frame'])
                # searchsorted with side='right' counts frames <= first_frame
                n_before = int(np.searchsorted(court_frames, first_frame, side='right'))
                rec = dict(acquisition=acq, id=fly_id,
                           boutnum=row[boutnum_col],
                           first_frame=first_frame,
                           total_action_frames=total,
                           relative_timing=n_before / total)
                rec.update(meta)
                records.append(rec)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).reset_index(drop=True)


def get_assay_ppm(assay_df, assay_type, ppm_colname="PPM", threshold=0.05):
    '''
    Check PPM variation across acquisitions for an assay type and return
    mean PPM if within threshold, else warn and return None.

    Arguments:
        assay_df   -- df for a single assay type
        assay_type -- assay type string for reporting

    Keyword Arguments:
        threshold -- max allowed coefficient of variation (default: 0.05)

    Returns:
        ppm -- mean PPM if variation is acceptable, else None
    '''
    ppm_per_acq = assay_df.groupby('acquisition')[ppm_colname].first()
    ppm_mean = ppm_per_acq.mean()
    ppm_cv = ppm_per_acq.std() / ppm_mean  # coefficient of variation

    print(f"\n  PPM summary for {assay_type}:")
    for acq, ppm in ppm_per_acq.items():
        pct_diff = abs(ppm - ppm_mean) / ppm_mean * 100
        print(f"    {acq}: {ppm:.3f} px/mm  ({pct_diff:.1f}% from mean)")
    print(f"    mean: {ppm_mean:.3f}, CV: {ppm_cv*100:.1f}%")

    if ppm_cv > threshold:
        print(f"  WARNING: PPM variation ({ppm_cv*100:.1f}%) exceeds threshold "
              f"({threshold*100:.1f}%) for {assay_type}. "
              f"Skipping relative position density plot.")
        return None

    print(f"  PPM variation within threshold — using mean PPM: {ppm_mean:.3f}")
    return ppm_mean


# ── Switch correction utilities ───────────────────────────────────────────────

def apply_switch_corrections_to_target(df, focal_id, corrections,
                                       action_col='courtship'):
    '''
    Apply (switch_frame, before_tgt, after_tgt) corrections to the target column
    for focal_id, operating independently within each courtship bout.
    Modifies df in place.

    Uses {action_col}_boutnum to identify bout boundaries (consistent with the
    rest of the pipeline).  Falls back to treating all courtship as one bout if
    the boutnum column is absent.

    corrections -- [(switch_frame, before_tgt, after_tgt), ...].
                   before_tgt / after_tgt may be None (skip that segment).

    Within each bout:
        [bout_start … first_switch)  → before_tgt of first correction in bout
        [switch[i]  … switch[i+1])  → after_tgt of correction[i]
        [last_switch … bout_end]    → after_tgt of last correction in bout
    '''
    if not corrections:
        return

    target_col  = f'{action_col}_target'
    boutnum_col = f'{action_col}_boutnum'
    corrections = sorted(corrections, key=lambda c: c[0])

    court_mask = (df['id'] == focal_id) & (df[action_col] == focal_id)
    if not court_mask.any():
        return

    def _assign(frame_list, target_id):
        if target_id is None or not frame_list:
            return
        mask = (df['id'] == focal_id) & (df['frame'].isin(set(frame_list)))
        df.loc[mask, target_col] = int(target_id)

    def _apply_to_bout(bout_frames):
        bout_start = bout_frames[0]
        bout_end   = bout_frames[-1]
        bout_corr  = [(sw, b, a) for sw, b, a in corrections
                      if bout_start <= sw <= bout_end]
        if not bout_corr:
            return
        sw_frames = [c[0] for c in bout_corr]
        _assign([f for f in bout_frames if f < sw_frames[0]], bout_corr[0][1])
        for i, (sw_frame, _, after_tgt) in enumerate(bout_corr):
            next_sw = bout_corr[i + 1][0] if i + 1 < len(bout_corr) else None
            seg = ([f for f in bout_frames if sw_frame <= f < next_sw]
                   if next_sw is not None
                   else [f for f in bout_frames if f >= sw_frame])
            _assign(seg, after_tgt)

    court_df = df[court_mask]
    if boutnum_col in df.columns:
        for _, bout_group in court_df.groupby(boutnum_col):
            _apply_to_bout(sorted(bout_group['frame'].unique()))
    else:
        _apply_to_bout(sorted(court_df['frame'].unique()))


def get_switch_new_target_trajectories(df, action_col='courtship', fps=60,
                                        window_sec=1.0,
                                        switch_source='prefer_manual'):
    '''
    For each switch event return the ego-centric trajectory of the new target
    over a symmetric window centred on the switch frame.

    Positions are targ_rel_pos_x/y from the pair row for (focal_fly, new_target)
    at each frame — i.e. new target position in the focal fly's instantaneous
    reference frame (focal facing +x, origin at focal fly centroid).

    Calls get_switch_frame_vectors to identify new_target per event, then expands
    each event to the full frame range and merges with pair rows.

    Arguments:
        df         -- processed tracks dataframe (output of load_all_processed_dfs)
        action_col -- action column prefix (default: 'courtship')
        fps        -- frames per second (default: 60)
        window_sec -- full window duration in seconds; half on each side (default: 1.0)

    Keyword Arguments:
        switch_source -- 'auto', 'manual', or 'prefer_manual' (default: 'prefer_manual')

    Returns:
        DataFrame with columns:
            acquisition, id (focal fly), switch_frame, new_target, old_target,
            frame, t_rel  (seconds from switch; negative = before),
            x_ego, y_ego  (ego-centric pixels — divide by ppm for mm),
            new_targ_rel_ori, old_targ_rel_ori  (target body orientation in the
                focal frame, radians, if 'targ_rel_ori' present),
            focal_x_allo, focal_y_allo, focal_ori_allo,
            new_x_allo, new_y_allo, new_ori_allo,
            old_x_allo, old_y_allo, old_ori_allo  (lab-frame position in pixels
                and orientation in radians, if 'pos_x'/'pos_y' present),
            new_rel_vel, new_abs_theta_error_dt  (new target pair row, if columns present),
            old_rel_vel, old_abs_theta_error_dt  (old target pair row, if columns present),
            plus metadata columns (triad_type, assay_type, species) if present
    '''
    if 'targ_rel_pos_x' not in df.columns or 'targ_rel_pos_y' not in df.columns:
        print("  get_switch_new_target_trajectories: 'targ_rel_pos_x/y' not found.")
        return pd.DataFrame()

    switch_events = get_switch_frame_vectors(df, action_col=action_col,
                                              switch_source=switch_source)
    if len(switch_events) == 0:
        return pd.DataFrame()

    half_win  = int(round(window_sec * fps / 2))
    meta_cols = [c for c in ['triad_type', 'assay_type', 'species']
                 if c in switch_events.columns]

    # Build per-pair lookup — include scalar metrics if available
    scalar_cols = [c for c in ['rel_vel', 'abs_theta_error_dt',
                                'theta_error_deg',
                                'target_ang_vel_fov', 'target_ang_vel_fov_signed',
                                'target_ang_vel_fov_deg', 'target_ang_vel_fov_signed_deg']
                   if c in df.columns]
    ori_cols    = ['targ_rel_ori'] if 'targ_rel_ori' in df.columns else []
    pair_cols   = (['acquisition', 'frame', 'id', 'pair',
                    'targ_rel_pos_x', 'targ_rel_pos_y'] + ori_cols + scalar_cols)
    pair_df     = df[pair_cols].copy()
    pair_split  = pair_df['pair'].str.split('_')
    fly0        = pair_split.str[0].astype(int)
    fly1        = pair_split.str[1].astype(int)
    pair_df['_partner'] = np.where(fly0 == pair_df['id'].values, fly1, fly0)
    pair_df = (pair_df.drop(columns='pair')
               .drop_duplicates(subset=['acquisition', 'frame', 'id', '_partner']))

    # Expand each switch event to its window of frames (carry both new and old target IDs)
    event_rows = []
    for _, ev in switch_events.iterrows():
        acq          = ev['acquisition']
        focal_id     = int(ev['id'])
        switch_frame = int(ev['frame'])
        new_target   = int(ev['new_target'])
        old_target   = int(ev['old_target']) if 'old_target' in ev.index else -1
        meta         = {c: ev[c] for c in meta_cols}
        for frame in range(switch_frame - half_win, switch_frame + half_win + 1):
            row = dict(acquisition=acq, id=focal_id,
                       switch_frame=switch_frame,
                       new_target=new_target, old_target=old_target,
                       frame=frame, t_rel=(frame - switch_frame) / fps)
            row.update(meta)
            event_rows.append(row)

    if not event_rows:
        return pd.DataFrame()

    events_expanded = pd.DataFrame(event_rows)

    # Merge 1: new target — ego-centric position + scalar metrics
    new_rename = {'_partner': 'new_target',
                  'targ_rel_pos_x': 'x_ego', 'targ_rel_pos_y': 'y_ego'}
    if ori_cols:
        new_rename['targ_rel_ori'] = 'new_targ_rel_ori'
    for sc in scalar_cols:
        new_rename[sc] = f'new_{sc}'
    result = events_expanded.merge(
        pair_df.rename(columns=new_rename),
        on=['acquisition', 'frame', 'id', 'new_target'],
        how='inner'
    ).dropna(subset=['x_ego', 'y_ego'])

    # Merge 2: old target — position + scalar metrics (left join; NaN if row absent)
    if (result['old_target'] != -1).any():
        old_pos_scalar = ['targ_rel_pos_x', 'targ_rel_pos_y'] + ori_cols + scalar_cols
        old_pair = pair_df[['acquisition', 'frame', 'id', '_partner'] + old_pos_scalar].copy()
        old_rename = {'_partner': 'old_target',
                      'targ_rel_pos_x': 'old_x_ego', 'targ_rel_pos_y': 'old_y_ego'}
        if ori_cols:
            old_rename['targ_rel_ori'] = 'old_targ_rel_ori'
        for sc in scalar_cols:
            old_rename[sc] = f'old_{sc}'
        result = result.merge(
            old_pair.rename(columns=old_rename),
            on=['acquisition', 'frame', 'id', 'old_target'],
            how='left'
        )

    # Vel lookup: per-fly absolute velocity (one value per fly per frame, not a pair row)
    if 'vel' in df.columns:
        fly_vel = (df[['acquisition', 'frame', 'id', 'vel']]
                   .drop_duplicates(['acquisition', 'frame', 'id']))
        result = result.merge(
            fly_vel.rename(columns={'id': 'new_target', 'vel': 'new_vel'}),
            on=['acquisition', 'frame', 'new_target'], how='left')
        result = result.merge(
            fly_vel.rename(columns={'id': 'old_target', 'vel': 'old_vel'}),
            on=['acquisition', 'frame', 'old_target'], how='left')

    # Allocentric (lab-frame) per-fly position + orientation for focal/new/old.
    # Lets callers plot the same window in world coordinates and tell apart
    # focal-fly turning from target-fly motion.
    allo_cols = [c for c in ['pos_x', 'pos_y', 'ori'] if c in df.columns]
    if {'pos_x', 'pos_y'}.issubset(df.columns):
        fly_allo = (df[['acquisition', 'frame', 'id'] + allo_cols]
                    .drop_duplicates(['acquisition', 'frame', 'id']))
        result = result.merge(
            fly_allo.rename(columns={'pos_x': 'focal_x_allo', 'pos_y': 'focal_y_allo',
                                     'ori': 'focal_ori_allo'}),
            on=['acquisition', 'frame', 'id'], how='left')
        result = result.merge(
            fly_allo.rename(columns={'id': 'new_target',
                                     'pos_x': 'new_x_allo', 'pos_y': 'new_y_allo',
                                     'ori': 'new_ori_allo'}),
            on=['acquisition', 'frame', 'new_target'], how='left')
        result = result.merge(
            fly_allo.rename(columns={'id': 'old_target',
                                     'pos_x': 'old_x_allo', 'pos_y': 'old_y_allo',
                                     'ori': 'old_ori_allo'}),
            on=['acquisition', 'frame', 'old_target'], how='left')

    return result.reset_index(drop=True)


def get_switch_theta_error_comparison(df, action_col='courtship',
                                       switch_source='prefer_manual'):
    '''
    For each switch event return abs_theta_error_deg to both the old and new target
    at the switch frame, and the delta (new - old).

    Builds on get_switch_frame_vectors to identify old/new target pairs, then joins
    theta_error_deg from the pair rows (one row per focal+partner per frame), so
    both values are available at the switch frame without lookahead/lookbehind.

    Note: old_target and new_target are not stored columns in the processed parquet;
    they are computed here from the courtship_target shift within each bout.

    Arguments:
        df         -- processed tracks dataframe (output of load_all_processed_dfs)
        action_col -- action column prefix (default: 'courtship')

    Keyword Arguments:
        switch_source -- 'auto', 'manual', or 'prefer_manual' (default: 'prefer_manual')

    Returns:
        DataFrame with columns:
            acquisition, id, frame,
            new_target, old_target,
            new_theta_error_deg, old_theta_error_deg,
            new_abs_theta_error_deg, old_abs_theta_error_deg,
            delta_abs_theta_error_deg  (= new - old; negative means switch toward
                                         better-oriented target),
            plus metadata columns (triad_type, assay_type, species) if present
    '''
    if 'theta_error_deg' not in df.columns:
        print("  get_switch_theta_error_comparison: 'theta_error_deg' not found — returning empty.")
        return pd.DataFrame()

    # Reuse get_switch_frame_vectors to correctly identify new_target / old_target
    switch_events = get_switch_frame_vectors(df, action_col=action_col,
                                             switch_source=switch_source)
    if len(switch_events) == 0:
        return pd.DataFrame()

    # theta_error per (acquisition, frame, focal_id, partner_id)
    pair_df = df[['acquisition', 'frame', 'id', 'pair', 'theta_error_deg']].copy()
    pair_split = pair_df['pair'].str.split('_')
    fly0 = pair_split.str[0].astype(int)
    fly1 = pair_split.str[1].astype(int)
    pair_df['_partner'] = np.where(fly0 == pair_df['id'].values, fly1, fly0)
    pair_df = pair_df.drop(columns='pair')

    # First merge: bring in theta_error to the new target at the switch frame
    switch_events = switch_events.merge(
        pair_df.rename(columns={'_partner': '_nt',
                                'theta_error_deg': 'new_theta_error_deg'}),
        left_on=['acquisition', 'frame', 'id', 'new_target'],
        right_on=['acquisition', 'frame', 'id', '_nt'],
        how='inner'
    ).drop(columns='_nt')

    # Second merge: bring in theta_error to the old target at the same frame
    switch_events = switch_events.merge(
        pair_df.rename(columns={'_partner': '_ot',
                                'theta_error_deg': 'old_theta_error_deg'}),
        left_on=['acquisition', 'frame', 'id', 'old_target'],
        right_on=['acquisition', 'frame', 'id', '_ot'],
        how='inner'
    ).drop(columns='_ot')

    switch_events['new_abs_theta_error_deg'] = switch_events['new_theta_error_deg'].abs()
    switch_events['old_abs_theta_error_deg'] = switch_events['old_theta_error_deg'].abs()
    # negative delta = switched toward better-oriented (lower error) target
    switch_events['delta_abs_theta_error_deg'] = (
        switch_events['new_abs_theta_error_deg'] - switch_events['old_abs_theta_error_deg']
    )

    return switch_events.reset_index(drop=True)


def classify_switches_by_theta_error(df, action_col='courtship',
                                     threshold_deg=15.0,
                                     switch_source='prefer_manual'):
    '''
    Classify each switch event by how the new target's |theta error| compares to
    the old target's, at the switch frame.

    With delta = new_abs_theta_error_deg - old_abs_theta_error_deg:
        'similar'    -- |delta| < threshold_deg  (within threshold of each other)
        'new_higher' -- delta >= threshold_deg   (new |theta error| >= old + threshold)
        'new_lower'  -- delta <= -threshold_deg  (new |theta error| <= old - threshold)

    Builds on get_switch_theta_error_comparison (one row per switch event).

    Arguments:
        df         -- processed tracks dataframe (output of load_all_processed_dfs)
        action_col -- action column prefix (default: 'courtship')

    Keyword Arguments:
        threshold_deg -- |theta error| difference cutoff in degrees (default: 15.0)
        switch_source -- 'auto', 'manual', or 'prefer_manual' (default: 'prefer_manual')

    Returns:
        The get_switch_theta_error_comparison DataFrame with two added columns:
            switch_case  -- 'similar' | 'new_higher' | 'new_lower'
            switch_frame -- alias of 'frame' (the switch frame), for joining to
                            get_switch_new_target_trajectories output
        Empty DataFrame if no theta-error switch data.
    '''
    cmp = get_switch_theta_error_comparison(df, action_col=action_col,
                                            switch_source=switch_source)
    if len(cmp) == 0:
        return cmp
    cmp = cmp.dropna(subset=['delta_abs_theta_error_deg']).copy()
    delta = cmp['delta_abs_theta_error_deg']
    cmp['switch_case'] = np.where(
        delta >= threshold_deg, 'new_higher',
        np.where(delta <= -threshold_deg, 'new_lower', 'similar'))
    cmp['switch_frame'] = cmp['frame']
    return cmp.reset_index(drop=True)


def get_switch_target_ang_vel_fov_comparison(df, action_col='courtship',
                                              switch_source='prefer_manual'):
    '''
    For each switch event return target_ang_vel_fov (and signed version) for both
    old and new target at the switch frame, and the delta (new − old).

    target_ang_vel_fov is the bearing rate due to the target fly's own motion only
    (focal fly held fixed) — it isolates the target's contribution to its angular
    position in the focal fly's field of view, excluding the focal fly's own
    translation and rotation.  The signed version is positive for progressive motion
    (target motion reducing |theta_error|) and negative for regressive motion.

    Arguments:
        df         -- processed tracks dataframe (output of load_all_processed_dfs)
        action_col -- action column prefix (default: 'courtship')

    Keyword Arguments:
        switch_source -- 'auto', 'manual', or 'prefer_manual' (default: 'prefer_manual')

    Returns:
        DataFrame with columns:
            acquisition, id, frame, new_target, old_target,
            new_target_ang_vel_fov, old_target_ang_vel_fov, delta_target_ang_vel_fov,
            new_target_ang_vel_fov_signed, old_target_ang_vel_fov_signed,
            delta_target_ang_vel_fov_signed  (if signed column present),
            plus metadata columns (triad_type, assay_type, species) if present
    '''
    if 'target_ang_vel_fov' not in df.columns:
        print("  get_switch_target_ang_vel_fov_comparison: 'target_ang_vel_fov' not found "
              "— returning empty.")
        return pd.DataFrame()

    switch_events = get_switch_frame_vectors(df, action_col=action_col,
                                             switch_source=switch_source)
    if len(switch_events) == 0:
        return pd.DataFrame()

    metric_cols = [c for c in ['target_ang_vel_fov', 'target_ang_vel_fov_signed',
                               'target_ang_vel_fov_deg', 'target_ang_vel_fov_signed_deg']
                   if c in df.columns]

    pair_df = df[['acquisition', 'frame', 'id', 'pair'] + metric_cols].copy()
    pair_split = pair_df['pair'].str.split('_')
    fly0 = pair_split.str[0].astype(int)
    fly1 = pair_split.str[1].astype(int)
    pair_df['_partner'] = np.where(fly0 == pair_df['id'].values, fly1, fly0)
    pair_df = pair_df.drop(columns='pair')

    new_rename = {'_partner': '_nt'}
    for c in metric_cols:
        new_rename[c] = f'new_{c}'
    switch_events = switch_events.merge(
        pair_df.rename(columns=new_rename),
        left_on=['acquisition', 'frame', 'id', 'new_target'],
        right_on=['acquisition', 'frame', 'id', '_nt'],
        how='inner'
    ).drop(columns='_nt')

    old_rename = {'_partner': '_ot'}
    for c in metric_cols:
        old_rename[c] = f'old_{c}'
    switch_events = switch_events.merge(
        pair_df.rename(columns=old_rename),
        left_on=['acquisition', 'frame', 'id', 'old_target'],
        right_on=['acquisition', 'frame', 'id', '_ot'],
        how='inner'
    ).drop(columns='_ot')

    for c in metric_cols:
        switch_events[f'delta_{c}'] = (
            switch_events[f'new_{c}'] - switch_events[f'old_{c}']
        )

    return switch_events.reset_index(drop=True)


def get_switch_metric_comparison(df, metric_cols, action_col='courtship',
                                 switch_source='prefer_manual'):
    '''
    Generic old-vs-new comparison at each switch frame for arbitrary pair-row
    metric column(s) (e.g. 'dist_to_other'). For each switch event returns the
    metric to the old and new target at the switch frame, plus delta (new - old).

    Mirrors get_switch_target_ang_vel_fov_comparison but for any column present in
    df at one-row-per-(focal, partner, frame) granularity.

    Arguments:
        df          -- processed tracks dataframe (output of load_all_processed_dfs)
        metric_cols -- column name or list of names to compare across old/new target

    Keyword Arguments:
        action_col    -- action column prefix (default: 'courtship')
        switch_source -- 'auto', 'manual', or 'prefer_manual' (default: 'prefer_manual')

    Returns:
        DataFrame with acquisition, id, frame, new_target, old_target, and for each
        metric c: new_<c>, old_<c>, delta_<c>, plus metadata columns if present.
        Empty DataFrame if no switches or none of metric_cols present.
    '''
    if isinstance(metric_cols, str):
        metric_cols = [metric_cols]
    have = [c for c in metric_cols if c in df.columns]
    if not have:
        print(f"  get_switch_metric_comparison: none of {metric_cols} found — returning empty.")
        return pd.DataFrame()

    switch_events = get_switch_frame_vectors(df, action_col=action_col,
                                             switch_source=switch_source)
    if len(switch_events) == 0:
        return pd.DataFrame()

    pair_df = df[['acquisition', 'frame', 'id', 'pair'] + have].copy()
    pair_split = pair_df['pair'].str.split('_')
    fly0 = pair_split.str[0].astype(int)
    fly1 = pair_split.str[1].astype(int)
    pair_df['_partner'] = np.where(fly0 == pair_df['id'].values, fly1, fly0)
    pair_df = pair_df.drop(columns='pair')

    new_rename = {'_partner': '_nt', **{c: f'new_{c}' for c in have}}
    switch_events = switch_events.merge(
        pair_df.rename(columns=new_rename),
        left_on=['acquisition', 'frame', 'id', 'new_target'],
        right_on=['acquisition', 'frame', 'id', '_nt'], how='inner').drop(columns='_nt')

    old_rename = {'_partner': '_ot', **{c: f'old_{c}' for c in have}}
    switch_events = switch_events.merge(
        pair_df.rename(columns=old_rename),
        left_on=['acquisition', 'frame', 'id', 'old_target'],
        right_on=['acquisition', 'frame', 'id', '_ot'], how='inner').drop(columns='_ot')

    for c in have:
        switch_events[f'delta_{c}'] = switch_events[f'new_{c}'] - switch_events[f'old_{c}']
    return switch_events.reset_index(drop=True)


# ── 2M / 1M courtship multiplicity (MMF: how many focal males court each frame) ──
#
# In an MMF triad the focal flies are the two males. For each frame we count how
# many focal flies are courting ("any courtship": action_col != -1, regardless of
# target). 2M frames = both males courting; 1M frames = exactly one. This lets MMF
# be split into a 2M and a 1M sub-assay so the pursuit/metric analyses can compare
# them against MFF.

def count_courting_focal_per_frame(df, focal_ids, action_col='courtship'):
    """Series mapping (acquisition, frame) -> number of focal flies courting.

    A focal fly counts as courting on a frame when its `action_col` != -1 on any of
    its pair-rows (regardless of target). Frames with no focal fly present are absent.
    """
    foc = df[df['id'].isin(focal_ids)]
    if foc.empty:
        return pd.Series(dtype='int64')
    courting = (foc.assign(_court=foc[action_col] != -1)
                   .groupby(['acquisition', 'frame', 'id'])['_court'].any())
    return courting.groupby(level=['acquisition', 'frame']).sum().astype('int64')


def _broadcast_frame_counts(df, counts):
    """Map a (acquisition, frame)-indexed Series back onto df's rows (NaN where absent)."""
    idx = pd.MultiIndex.from_arrays([df['acquisition'], df['frame']])
    return counts.reindex(idx).to_numpy()


def split_by_courting_count(df, focal_ids, action_col='courtship'):
    """(df_2M, df_1M): rows whose frame has exactly 2 vs exactly 1 focal flies courting.

    For MMF (2 focal males): 2M = both males courting that frame, 1M = exactly one.
    Frames with 0 courting focal flies belong to neither subset.
    """
    counts = count_courting_focal_per_frame(df, focal_ids, action_col)
    n = _broadcast_frame_counts(df, counts)
    return df[n == 2].copy(), df[n == 1].copy()


def split_assay_dfs_by_courting_count(assay_dfs, focal_flies_map, split_triad='MMF',
                                      action_col='courtship'):
    """Expand an assay_dfs dict so each `split_triad` assay becomes two entries,
    '<key>_2M' and '<key>_1M' (frames with 2 vs 1 focal flies courting). Other assays
    pass through unchanged. Split dfs keep their triad_type (so focal_flies_map still
    resolves) and get their assay_type column set to the new key.
    """
    out = {}
    for key, df in assay_dfs.items():
        triad_type = df['triad_type'].iloc[0]
        focal = focal_flies_map.get(triad_type) if focal_flies_map else None
        if triad_type != split_triad or not focal:
            out[key] = df
            continue
        d2, d1 = split_by_courting_count(df, focal, action_col)
        for sub_key, sub in ((f'{key}_2M', d2), (f'{key}_1M', d1)):
            if 'assay_type' in sub.columns:
                sub['assay_type'] = sub_key
            out[sub_key] = sub
    return out


def expand_keyed_dict_for_split(keyed, assay_dfs, split_triad='MMF', transform=None):
    """Given a dict keyed by assay_type (e.g. ppm or colors) and the ORIGINAL
    assay_dfs, add '<key>_2M' and '<key>_1M' entries for each `split_triad` assay so
    the dict matches split_assay_dfs_by_courting_count's keys. `transform(value, sub)`
    (sub in {'2M','1M'}) maps the base value to the sub-entry value; default copies it.
    """
    out = dict(keyed)
    for key, df in assay_dfs.items():
        if df['triad_type'].iloc[0] == split_triad and key in keyed:
            for sub in ('2M', '1M'):
                out[f'{key}_{sub}'] = transform(keyed[key], sub) if transform else keyed[key]
    return out


def restrict_action_to_target_sex(df, sex_map, focal_flies_map=None, keep_sex='F',
                                  action_col='courtship'):
    '''
    Return a copy of df where focal-fly ACTION frames whose target is NOT of `keep_sex`
    are demoted to non-action (action_col set to -1 on those rows). Non-focal rows and
    non-action rows are left untouched.

    This lets downstream per-focal courtship selection (get_courtship_target_fov /
    get_target_pair_fov) keep only e.g. female-target courtship frames — e.g. to make an
    "MMF 2M, target = female" subcase out of an already 2M-split df.

    Arguments:
        df              -- processed tracks df (one triad_type/assay)
        sex_map         -- {triad_type: {id: 'M'|'F'}}
        focal_flies_map -- {triad_type: [focal ids]}; if None, all flies are eligible
        keep_sex        -- target sex to KEEP as action (default 'F')
        action_col      -- action column prefix (default 'courtship')
    '''
    target_col = f'{action_col}_target'
    if df.empty or action_col not in df.columns or target_col not in df.columns:
        return df.copy()
    triad = df['triad_type'].iloc[0] if 'triad_type' in df.columns else None
    sexes = sex_map.get(triad, {})
    focal = focal_flies_map.get(triad) if focal_flies_map else None
    d = df.copy()
    tgt = pd.to_numeric(d[target_col], errors='coerce')
    tgt_sex = tgt.map(lambda t: sexes.get(int(t)) if pd.notna(t) else None)
    acting = (d[action_col] == d['id'])
    if focal is not None:
        acting &= d['id'].isin(focal)
    demote = acting & (tgt_sex != keep_sex)
    d.loc[demote, action_col] = -1
    return d


def courtship_multiplicity_per_acquisition(df, focal_ids, action_col='courtship'):
    """Per acquisition: fraction of courtship frames that are 2-male vs 1-male.

    A courtship frame = >=1 focal fly courting. Returns columns acquisition,
    n_court_frames, frac_2M, frac_1M (the two fractions sum to 1).
    """
    counts = count_courting_focal_per_frame(df, focal_ids, action_col).reset_index(name='n')
    rows = []
    for acq, g in counts.groupby('acquisition'):
        court = g[g['n'] >= 1]
        if court.empty:
            continue
        tot = len(court)
        n2 = int((court['n'] >= 2).sum())
        rows.append({'acquisition': acq, 'n_court_frames': tot,
                     'frac_2M': n2 / tot, 'frac_1M': (tot - n2) / tot})
    return pd.DataFrame(rows)