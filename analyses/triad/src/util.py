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

    Arguments:
        targets         -- np.array of integer target ids
        min_bout_frames -- minimum number of frames a target run must span

    Returns:
        enforced -- np.array of target ids with short runs removed.
    '''
    enforced = targets.copy().astype(int)
    n = len(enforced)
    i = 0

    while i < n:
        current = enforced[i]

        j = i
        while j < n and enforced[j] == current:
            j += 1

        bout_len = j - i

        if bout_len < min_bout_frames:
            if i > 0:
                enforced[i:j] = enforced[i - 1]
            else:
                enforced[i:j] = -2

        i = j

    # forward fill any flagged frames at start with first stable target
    if (enforced == -2).any():
        first_stable = next((v for v in enforced if v not in (-1, -2)), None)
        if first_stable is not None:
            enforced[enforced == -2] = first_stable
        else:
            enforced[enforced == -2] = -1

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
    switch_col = f'{action_col}_switch'
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

def trim_df_at_copulation(df, copulation_col='copulation', boutnum_col='copulation_boutnum'):
    '''
    Trim df to exclude all frames after the start of the first copulation bout.
    If no copulation is annotated, returns df unchanged.

    Arguments:
        df -- single-acquisition tracks dataframe with copulation action column

    Keyword Arguments:
        copulation_col -- name of copulation action column (default: 'copulation')
        boutnum_col    -- name of copulation boutnum column (default: 'copulation_boutnum')

    Returns:
        df -- trimmed dataframe
    '''
    if copulation_col not in df.columns:
        print("No copulation column found, returning df unchanged.")
        return df

    # find first frame of first copulation bout
    cop_rows = df[(df[copulation_col] != -1) & (df[boutnum_col].notna())]
    if len(cop_rows) == 0:
        print("No copulation bouts found, returning df unchanged.")
        return df

    first_cop_frame = int(cop_rows['frame'].min())
    print(f"First copulation frame: {first_cop_frame} — trimming all frames after it.")

    df = df[df['frame'] <= first_cop_frame].copy()
    return df

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