"""
Orientation-flip detection and correction for FlyTracker data.

Pure numpy/pandas: detects implausible orientation jumps, pairs them into flip
segments, and corrects them by adding pi. Also includes the mounting-frame
orientation override (correct_mounting_nearest_ori). The interactive review GUI
lives in flip.review_gui; matplotlib diagnostics in flip.plots.
"""
import numpy as np
import pandas as pd

from .review_gui import _review_jumps_interactive


def _circ_diff(a, b):
    """Signed circular difference a - b, wrapped to [-π, π]."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _nan_aware_circ_diffs(ori):
    """
    Circular diffs between consecutive non-NaN ori values.
    NaN frames are skipped; the resulting diff is attributed to the later
    non-NaN index so a flip across a NaN gap is still detected.
    """
    diffs = np.zeros(len(ori))
    valid_idxs = np.where(~np.isnan(ori.astype(float)))[0]
    for j in range(1, len(valid_idxs)):
        i_curr = valid_idxs[j]
        i_prev = valid_idxs[j - 1]
        diffs[i_curr] = _circ_diff(ori[i_curr], ori[i_prev])
    return diffs


def print_jump_frames(df, jump_threshold_rad=2.0):
    """
    Print every frame where |d(ori)/dt| exceeds jump_threshold_rad, per fly.
    Useful for debugging before running detect_flip_events.
    """
    for fly_id, fly_df in df.groupby('id'):
        fly_df = (fly_df.drop_duplicates('frame')
                        .sort_values('frame')
                        .reset_index(drop=True))
        ori    = fly_df['ori'].values
        frames = fly_df['frame'].values

        diffs = _nan_aware_circ_diffs(ori)

        jump_idxs = np.where(np.abs(diffs) > jump_threshold_rad)[0]
        if len(jump_idxs) == 0:
            print(f"fly {fly_id}: no jumps above {jump_threshold_rad:.2f} rad")
            continue

        print(f"fly {fly_id}: {len(jump_idxs)} jumps above {jump_threshold_rad:.2f} rad")
        for k in jump_idxs:
            later = jump_idxs[jump_idxs > k]
            next_jump = later[0] if len(later) > 0 else None
            seg_std = (np.nanstd(diffs[k + 1:next_jump])
                       if next_jump is not None and next_jump > k + 1 else np.nan)
            next_str = str(int(frames[next_jump])) if next_jump is not None else 'none'
            std_str  = f"{seg_std:.3f}" if not np.isnan(seg_std) else 'n/a'
            print(f"  frame {int(frames[k]):6d}  diff={diffs[k]:+.3f} rad  "
                  f"next_jump={next_str:>8}  seg_std={std_str}")


def correct_mounting_nearest_ori(df, action_col='mounting_attempt', inplace=False):
    """
    Sloppy heuristic: during mounting attempts, set the actor fly's orientation
    to abs_ang_between toward the nearest other fly.

    No pi-ambiguity resolution — just directly assigns the line-of-sight angle.
    Useful as a quick sanity check before trying the finer pass-1 approach in
    correct_mounting_ori.

    Arguments:
        df  -- tracks df with pairwise metrics computed
               (needs 'dist_to_other', 'abs_ang_between', action_col)

    Keyword Arguments:
        action_col -- column whose non-(-1) value is the actor fly's id
                      (default: 'mounting_attempt')
        inplace    -- modify df in place (default: False)

    Returns:
        df with ori replaced by abs_ang_between for mounting frames
    """
    if not inplace:
        df = df.copy()

    for col in (action_col, 'dist_to_other', 'abs_ang_between'):
        if col not in df.columns:
            print(f"No '{col}' column — skipping.")
            return df

    mount_rows = df[df[action_col] != -1]
    if len(mount_rows) == 0:
        print(f"No mounting frames found in '{action_col}' — skipping.")
        return df

    # for each (frame, id) pick the pair row with the smallest dist_to_other
    nearest = mount_rows.loc[
        mount_rows.groupby(['frame', 'id'])['dist_to_other'].idxmin()
    ][['frame', 'id', 'abs_ang_between']].dropna()

    n = 0
    for _, row in nearest.iterrows():
        mask = (df['id'] == row['id']) & (df['frame'] == row['frame'])
        df.loc[mask, 'ori'] = float(row['abs_ang_between'])
        n += int(mask.sum())

    print(f"correct_mounting_nearest_ori: set ori = abs_ang_between for "
          f"{len(nearest)} frames ({n} rows).")
    return df


def detect_flip_events(df, fps=60, jump_threshold_rad=2.5, stability_threshold_rad=0.2,
                       interactive=False, video_path=None, trk_for_display=None,
                       id_colors=None, arrow_len=20, pad_sec=2.0,
                       interactive_threshold_sec=None):
    """
    Detect orientation flip events per fly.

    A flip event is a consecutive artifact-jump pair where:
      1. Both jumps exceed jump_threshold_rad (large angular step, implausible for
         real behavior).
      2. The orientation is stable between the jumps: nanstd(d(ori)/dt) over the
         flipped segment is below stability_threshold_rad, ruling out noisy motion.

    When interactive=True, each candidate jump is shown in an OpenCV window before
    pairing. You label each jump as:
      a / Enter — artifact (tracking glitch; included in correction pairing)
      t         — true flip (fly physically turned; excluded from correction)
      q / ESC   — skip / uncertain (excluded from correction)
    Only artifact jumps are then paired to form correction intervals.
    Requires video_path; pass trk_for_display=trk_orig to show uncorrected orientations.

    Arguments:
        df  -- single-acquisition tracks df with 'ori', 'id', 'frame' columns

    Keyword Arguments:
        fps                     -- frames per second (default: 60)
        jump_threshold_rad      -- angular jump size (rad) to flag as a flip onset
                                   (default: 2.5 rad)
        stability_threshold_rad -- max std of d(ori)/dt in the flipped segment
                                   (default: 0.2 rad/frame)
        interactive               -- if True, open GUI for per-jump labeling (default: False)
        video_path                -- path to .avi video; required when interactive=True
        trk_for_display           -- tracks df used for arrow overlay; defaults to df
        id_colors                 -- dict of fly_id → BGR color; auto-assigned if None
        arrow_len                 -- orientation arrow length in pixels (default: 20)
        pad_sec                   -- context seconds shown around each jump (default: 2.0)
        interactive_threshold_sec -- REQUIRED when interactive=True: auto-correct events
                                     shorter than this threshold; only interactively review
                                     events that exceed it. Suggested end frames from greedy
                                     pairing are pre-filled in the phase-2 GUI. Ignored when
                                     interactive=False (default: None)

    Returns:
        events -- DataFrame with columns:
                  fly_id, start_frame, end_frame, duration_frames, duration_sec,
                  jump_in_rad, jump_out_rad, segment_ori_std, mean_vel
    """
    if interactive and interactive_threshold_sec is None:
        raise ValueError(
            "detect_flip_events(interactive=True) requires interactive_threshold_sec "
            "(the no-threshold per-jump review mode was removed). Pass e.g. "
            "interactive_threshold_sec=4.0, or use interactive=False for auto-detection.")

    records = []

    for fly_id, fly_df in df.groupby('id'):
        # one row per frame for this fly (pick first pair row — ori is same across pairs)
        fly_df = (fly_df.drop_duplicates('frame')
                        .sort_values('frame')
                        .reset_index(drop=True))

        ori    = fly_df['ori'].values
        frames = fly_df['frame'].values

        # NaN-aware circular diffs: skips NaN frames so a flip across a NaN
        # gap is still attributed to the first non-NaN frame after the gap.
        diffs     = _nan_aware_circ_diffs(ori)
        abs_diffs = np.abs(diffs)

        jump_idxs = np.where(abs_diffs > jump_threshold_rad)[0]
        if len(jump_idxs) == 0:
            continue

        # interactive: user labels each jump and sets its end directly;
        # records are returned immediately — skip the greedy pairing for this fly
        if interactive:
            if video_path is None:
                print("interactive=True requires video_path — falling back to auto-detection.")
            else:
                display_trk = trk_for_display if trk_for_display is not None else df

                new_recs = _review_jumps_interactive(
                    fly_id, jump_idxs, frames, diffs,
                    trk_for_display=display_trk, df=df,
                    video_path=video_path, fps=fps,
                    id_colors=id_colors, arrow_len=arrow_len, pad_sec=pad_sec,
                    stability_threshold_rad=stability_threshold_rad,
                    interactive_threshold_sec=interactive_threshold_sec,
                )
                records.extend(new_recs)
                continue   # greedy pairing not needed

        # greedily pair jumps: consume both on a valid match, advance by 1 on failure
        k = 0
        while k < len(jump_idxs) - 1:
            start_idx = jump_idxs[k]
            end_idx   = jump_idxs[k + 1]

            jump_in_rad  = float(diffs[start_idx])
            jump_out_rad = float(diffs[end_idx])

            # --- stability constraint: d(ori)/dt inside segment must be quiet ---
            inner = diffs[start_idx + 1:end_idx]  # frames strictly between the two jumps
            segment_ori_std = float(np.nanstd(inner)) if len(inner) > 0 else 0.0
            if segment_ori_std >= stability_threshold_rad:
                k += 1
                continue

            # start_frame is the FIRST flipped frame (ori already jumped there).
            # end_frame   is the FIRST correct frame after the flip-out jump.
            # The flipped segment is [start_frame, end_frame).
            start_frame     = int(frames[start_idx])
            end_frame       = int(frames[end_idx])
            duration_frames = end_frame - start_frame
            duration_sec    = duration_frames / fps

            # optional per-fly velocity metric during the flipped segment
            seg_mask = ((df['id'] == fly_id) &
                        (df['frame'] >= start_frame) &
                        (df['frame'] < end_frame))
            seg_vel = df.loc[seg_mask, 'vel'] if 'vel' in df.columns else pd.Series(dtype=float)

            records.append({
                'fly_id':          fly_id,
                'start_frame':     start_frame,
                'end_frame':       end_frame,
                'duration_frames': duration_frames,
                'duration_sec':    round(duration_sec, 2),
                'jump_in_rad':     round(jump_in_rad, 3),
                'jump_out_rad':    round(jump_out_rad, 3),
                'segment_ori_std': round(segment_ori_std, 4),
                'mean_vel':        round(seg_vel.mean(), 3) if len(seg_vel) > 0 else np.nan,
            })
            k += 2  # both jumps consumed; next candidate starts after end_idx

    if not records:
        print("No flip events detected.")
        return pd.DataFrame()

    events = pd.DataFrame(records).sort_values('duration_sec', ascending=False).reset_index(drop=True)
    print(f"Detected {len(events)} flip events across {events['fly_id'].nunique()} flies.")
    print(f"Duration range: {events['duration_sec'].min():.2f}s – {events['duration_sec'].max():.2f}s")
    return events


def correct_flip_events(df, events, inplace=False):
    """
    Correct orientation for detected flip events by adding π (mod 2π).

    FlyTracker ori is always aligned to the major axis — a flip is simply the
    wrong endpoint of that axis being called 'head'. The correction is therefore
    arctan2(-sin(ori), -cos(ori)), which rotates by π and re-wraps to [-π, π]
    without altering the axis alignment.

    Frame semantics:
        start_frame is the FIRST flipped frame (ori has already jumped).
        end_frame   is the FIRST correct frame after the flip-out jump.
        Corrected frames: [start_frame, end_frame).

    The write-back uses (id == fly_id) & (frame == f) so all pair rows for
    that fly/frame are updated together.

    Derived columns that depend on ori (theta_error, ang_vel, facing_angle,
    etc.) will be stale after correction — recompute them from corrected ori.

    Arguments:
        df     -- tracks df (same one passed to detect_flip_events)
        events -- DataFrame returned by detect_flip_events

    Keyword Arguments:
        inplace -- modify df in place instead of returning a copy (default: False)

    Returns:
        df with ori corrected
    """
    if not inplace:
        df = df.copy()

    if len(events) == 0:
        print("No flip events to correct.")
        return df

    n_frames_corrected = 0

    for _, ev in events.iterrows():
        fly_id      = ev['fly_id']
        start_frame = int(ev['start_frame'])
        end_frame   = int(ev['end_frame'])

        mask = ((df['id'] == fly_id) &
                (df['frame'] >= start_frame) &
                (df['frame'] <  end_frame))
        ori_raw = df.loc[mask, 'ori'].values.astype(float)
        df.loc[mask, 'ori'] = np.arctan2(-np.sin(ori_raw), -np.cos(ori_raw))
        n_frames_corrected += int(mask.sum())

    print(f"Corrected {len(events)} flip events ({n_frames_corrected} rows across "
          f"{events['fly_id'].nunique()} flies).")
    return df
