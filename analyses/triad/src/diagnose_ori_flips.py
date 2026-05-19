"""
Diagnostic tools for inspecting orientation flip artifacts in FlyTracker data.

Typical usage (from a notebook or script):
    from analyses.triad.src.diagnose_ori_flips import detect_flip_events, plot_ori_flip_diagnostics

    events = detect_flip_events(df, fps=60, jump_threshold_rad=2.0)
    print(events[['fly_id', 'start_frame', 'duration_sec', 'min_dist_at_flip']].to_string())

    plot_ori_flip_diagnostics(df, fps=60, jump_threshold_rad=2.0, acq='my_acq')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def detect_flip_events(df, fps=60, jump_threshold_rad=2.5, stability_threshold_rad=0.2):
    """
    Detect orientation flip events per fly.

    A flip event is a consecutive jump pair where:
      1. Both jumps exceed jump_threshold_rad (large angular step at flip onset/offset).
         A jump >2 rad (~115°) in a single frame is implausible for real behavior.
      2. The orientation is stable between the jumps: nanstd(d(ori)/dt) over the
         flipped segment is below stability_threshold_rad, ruling out noisy motion.

    Arguments:
        df  -- single-acquisition tracks df with 'ori', 'id', 'frame' columns

    Keyword Arguments:
        fps                     -- frames per second (default: 60)
        jump_threshold_rad      -- angular jump size (rad) to flag as a flip onset
                                   (default: 2.0 rad ≈ 115°)
        stability_threshold_rad -- max std of d(ori)/dt in the flipped segment
                                   (default: 0.3 rad/frame ≈ 17°/frame)

    Returns:
        events -- DataFrame with columns:
                  fly_id, start_frame, end_frame, duration_frames, duration_sec,
                  jump_in_rad, jump_out_rad, segment_ori_std,
                  min_dist_at_flip, mean_dist_at_flip
    """
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


def plot_ori_flip_diagnostics(df, fps=60, jump_threshold_rad=2.0,
                               stability_threshold_rad=0.3,
                               fly_ids=None, acq=None,
                               max_flies=3, figsize_per_fly=(16, 4)):
    """
    For each fly, plot orientation timeseries with flip events highlighted,
    overlaid with minimum distance to any other fly.

    Arguments:
        df  -- single-acquisition tracks df

    Keyword Arguments:
        fps                     -- frames per second (default: 60)
        jump_threshold_rad      -- threshold for jump detection (default: 2.0)
        stability_threshold_rad -- max std of d(ori)/dt in the flipped segment (default: 0.3)
        fly_ids                 -- list of fly ids to plot; if None, plots all (up to max_flies)
        acq                     -- acquisition name for title (default: None)
        max_flies               -- max number of flies to plot (default: 3)
        figsize_per_fly         -- (width, height) per fly panel (default: (16, 4))

    Returns:
        fig
    """
    events = detect_flip_events(df, fps=fps, jump_threshold_rad=jump_threshold_rad,
                                stability_threshold_rad=stability_threshold_rad)

    all_fly_ids = sorted(df['id'].unique())
    if fly_ids is None:
        fly_ids = all_fly_ids[:max_flies]

    n = len(fly_ids)
    fig, axes = plt.subplots(n, 1,
                             figsize=(figsize_per_fly[0], figsize_per_fly[1] * n),
                             sharex=False)
    if n == 1:
        axes = [axes]

    for ax, fly_id in zip(axes, fly_ids):
        fly_df = (df[df['id'] == fly_id]
                    .drop_duplicates('frame')
                    .sort_values('frame'))
        t = fly_df['frame'].values / fps
        ori_deg = np.rad2deg(fly_df['ori'].values)

        # minimum dist_to_other across pairs for this fly, per frame
        if 'dist_to_other' in df.columns:
            min_dist = (df[df['id'] == fly_id]
                          .groupby('frame')['dist_to_other']
                          .min()
                          .reindex(fly_df['frame'])
                          .values)
        else:
            min_dist = None

        ax2 = ax.twinx()

        ax.plot(t, ori_deg, color='white', lw=0.8, alpha=0.9, label='ori (°)')
        ax.axhline(0, color='white', lw=0.3, alpha=0.3)

        if min_dist is not None:
            ax2.plot(t, min_dist, color='cyan', lw=0.8, alpha=0.6, label='min dist (px)')
            ax2.set_ylabel('min dist to other (px)', color='cyan', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='cyan')

        # shade flip event segments for this fly
        fly_events = events[events['fly_id'] == fly_id] if len(events) > 0 else pd.DataFrame()
        for _, ev in fly_events.iterrows():
            t_start = ev['start_frame'] / fps
            t_end   = ev['end_frame'] / fps
            ax.axvspan(t_start, t_end, color='red', alpha=0.2)
            ax.axvline(t_start, color='red', lw=1.0, ls='--', alpha=0.7)
            ax.text(t_start, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 180,
                    f"{ev['duration_sec']:.1f}s", color='red', fontsize=7, va='top')

        ax.set_ylabel('orientation (°)')
        ax.set_xlabel('time (s)')
        title = f'{acq} — ' if acq else ''
        ax.set_title(f'{title}fly {fly_id}  |  {len(fly_events)} flip events detected')
        ax.set_ylim(-190, 190)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels() if min_dist is not None else ([], [])
        flip_patch = mpatches.Patch(color='red', alpha=0.3, label='flip segment')
        ax.legend(handles=lines1 + lines2 + [flip_patch], fontsize=8, loc='upper right')

    plt.tight_layout()
    return fig


def plot_flip_duration_distribution(events, fps=60, figsize=(10, 4)):
    """
    Plot distribution of flip event durations and jump magnitudes.

    Arguments:
        events -- DataFrame from detect_flip_events()
    """
    if len(events) == 0:
        print("No events to plot.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].hist(events['duration_sec'], bins=30, color='tomato', edgecolor='white', lw=0.5)
    axes[0].set_xlabel('duration (s)')
    axes[0].set_ylabel('count')
    axes[0].set_title('Flip segment durations')

    axes[1].hist(np.abs(events['jump_in_rad']), bins=20, color='dodgerblue', edgecolor='white', lw=0.5)
    axes[1].axvline(np.pi, color='yellow', lw=1.5, ls='--', label='π')
    axes[1].set_xlabel('|jump| (rad)')
    axes[1].set_title('Jump magnitude at flip onset')
    axes[1].legend(fontsize=8)

    if 'min_dist_at_flip' in events.columns:
        axes[2].scatter(events['min_dist_at_flip'], events['duration_sec'],
                        color='limegreen', alpha=0.7, s=30, edgecolors='white', lw=0.5)
        axes[2].set_xlabel('min dist to other during flip (px)')
        axes[2].set_ylabel('duration (s)')
        axes[2].set_title('Flip duration vs proximity')

    plt.tight_layout()
    return fig
