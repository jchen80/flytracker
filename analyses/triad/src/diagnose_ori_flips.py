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


def _gui_build_lookup(trk, needed_frames):
    """Build {frame_idx: (pos_dict, ori_dict)} for the given set of frames."""
    lookup = {}
    for fidx, fdf in trk[trk['frame'].isin(needed_frames)].groupby('frame'):
        pos, ori_map = {}, {}
        for _, row in fdf.drop_duplicates('id').iterrows():
            fid = int(row['id'])
            if pd.isna(row['pos_x']) or pd.isna(row['pos_y']):
                continue
            pos[fid] = (int(row['pos_x']), int(row['pos_y']))
            if not pd.isna(row['ori']):
                ori_map[fid] = float(row['ori'])
        lookup[fidx] = (pos, ori_map)
    return lookup


def _gui_draw_arrows(frame_img, current, lookup, subj_fly_id, id_colors, arrow_len, cv2):
    """Draw orientation arrows and fly labels onto frame_img in place."""
    if current not in lookup:
        return
    positions, ori_vals = lookup[current]
    for fid, (px, py) in positions.items():
        color   = id_colors.get(fid, (255, 255, 255))
        is_subj = fid == subj_fly_id
        cv2.circle(frame_img, (px, py),
                   radius=10 if is_subj else 6,
                   color=color, thickness=2 if is_subj else -1)
        if fid in ori_vals:
            o  = ori_vals[fid]
            ex = int(px + arrow_len * np.cos(o))
            ey = int(py + arrow_len * np.sin(o))
            cv2.arrowedLine(frame_img, (px, py), (ex, ey),
                            color=color, thickness=2, tipLength=0.3)
        cv2.putText(frame_img, f'id={fid}', (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _gui_read_key(cv2, delay):
    """Return (key_raw, key_masked, left, right, up, down) for one waitKey call."""
    key_raw = cv2.waitKey(delay)
    key = key_raw & 0xFF
    # Linux:  81/83/82/84 after masking
    # macOS:  raw 63234/63235/63232/63233 → 2/3/0/1 after masking
    left  = key in (81, 2)  or key_raw in (63234, 65361)
    right = key in (83, 3)  or key_raw in (63235, 65363)
    up    = key == 82       or key_raw in (63232, 65362) or (key == 0 and key_raw >= 0)
    down  = key in (84, 1)  or key_raw in (63233, 65364)
    return key_raw, key, left, right, up, down


def _review_jumps_interactive(fly_id, jump_idxs, frames, diffs,
                               trk_for_display, df, video_path, fps,
                               id_colors, arrow_len, pad_sec,
                               stability_threshold_rad=0.2,
                               interactive_threshold_sec=None):
    """
    Two-phase interactive review for candidate orientation jumps.

    When interactive_threshold_sec is None:
        Each jump is reviewed individually. Phase 1 labels it (artifact/true_flip/skip);
        Phase 2 (artifact only) sets the end frame. A confirmed end that coincides with
        another detected jump auto-consumes it.

    When interactive_threshold_sec is set:
        Jumps are paired greedily in a single sequential pass. Pairs ≤ threshold are
        auto-corrected. Longer pairs enter Phase 1 + Phase 2, with the auto-detected
        pair end pre-filled in the slider. If the user shortens the end, the freed
        paired-end jump is re-evaluated immediately in the same pass (loop restarts
        from it via ji = je), so no separate re-pairing step is needed.

    Phase 1 keys:  a/Enter=artifact  t=true_flip  q/ESC=skip  ←/→=±1f  ↑/↓=±30f  space=play
    Phase 2 keys:  e=set_end  Enter=confirm  ←/→=±1f  ↑/↓=±30f  space=play  q=cancel
    """
    try:
        import cv2 as _cv2
    except ImportError:
        print("OpenCV not available — skipping interactive review.")
        return []

    if id_colors is None:
        _defaults = [(255, 144, 30), (71, 99, 255), (50, 205, 50), (0, 215, 255)]
        id_colors = {fid: _defaults[i % len(_defaults)]
                     for i, fid in enumerate(sorted(trk_for_display['id'].unique()))}

    cap = _cv2.VideoCapture(video_path)
    n_video_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    pad_frames     = int(np.ceil(pad_sec * fps))

    print(f"  fly {fly_id}: building frame lookup for display...")
    full_lookup = _gui_build_lookup(trk_for_display,
                                    set(trk_for_display['frame'].unique()))

    # ── Nested helpers — share cap, fps, n_video_frames, full_lookup, ────────
    # ── id_colors, arrow_len, fly_id via closure                        ────────

    def _phase1(ji, n_total, jump_frame, jump_diff, p1_start, p1_end):
        """Phase-1 GUI: label the jump. Returns 'artifact', 'true_flip', or 'skip'."""
        p1_len     = p1_end - p1_start
        pre_frames = min(p1_len, int(fps * 0.5))
        current    = max(p1_start, jump_frame - pre_frames)
        playing    = False

        win = (f"Phase 1 — Jump {ji+1}/{n_total} — fly {fly_id}  "
               f"f{jump_frame}  diff={jump_diff:+.2f}rad")
        _cv2.namedWindow(win, _cv2.WINDOW_NORMAL)
        _cv2.createTrackbar('Frame', win, current - p1_start, p1_len, lambda _: None)

        label = 'skip'
        while True:
            if not playing:
                tb      = _cv2.getTrackbarPos('Frame', win)
                current = p1_start + max(0, min(p1_len, tb))

            cap.set(_cv2.CAP_PROP_POS_FRAMES, current)
            ret, img = cap.read()
            if not ret:
                break
            h, w = img.shape[:2]

            if current == jump_frame:
                _cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 215, 255), 6)
                _cv2.putText(img, 'JUMP', (10, 60),
                             _cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3, _cv2.LINE_AA)
            _gui_draw_arrows(img, current, full_lookup, fly_id, id_colors, arrow_len, _cv2)
            _cv2.putText(img,
                         f"fly={fly_id}  frame={current}  "
                         f"jump@{jump_frame}  diff={jump_diff:+.2f}rad",
                         (10, 25), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, _cv2.LINE_AA)
            _cv2.putText(img,
                         "a/Enter:artifact  t:true_flip  arrows:step  space:play  q:skip",
                         (10, h - 10), _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, _cv2.LINE_AA)
            _cv2.imshow(win, img)

            _, key, left, right, up, down = _gui_read_key(
                _cv2, max(1, int(1000 / fps)) if playing else 30)

            if playing:
                current = min(p1_end, current + 1)
                _cv2.setTrackbarPos('Frame', win, current - p1_start)
                if current == p1_end:
                    playing = False

            if   key in (ord('a'), 13): label = 'artifact'; break
            elif key == ord('t'):       label = 'true_flip'; break
            elif key in (ord('q'), 27): label = 'skip';      break
            elif key == ord(' '):       playing = not playing
            elif left:
                current = max(p1_start, current - 1);  playing = False
                _cv2.setTrackbarPos('Frame', win, current - p1_start)
            elif right:
                current = min(p1_end,   current + 1);  playing = False
                _cv2.setTrackbarPos('Frame', win, current - p1_start)
            elif up:
                current = max(p1_start, current - 30); playing = False
                _cv2.setTrackbarPos('Frame', win, current - p1_start)
            elif down:
                current = min(p1_end,   current + 30); playing = False
                _cv2.setTrackbarPos('Frame', win, current - p1_start)

        _cv2.destroyWindow(win)
        return label

    def _phase2(jump_frame, default_end):
        """Phase-2 GUI: set the end of the flip interval. Returns (confirmed, man_end)."""
        p2_start         = jump_frame
        p2_end           = min(n_video_frames - 1, jump_frame + int(60 * fps))
        p2_len           = p2_end - p2_start
        man_end          = int(min(p2_end, default_end))
        p2_preview_start = max(p2_start, man_end - 10)
        current          = p2_preview_start
        playing          = False
        confirmed        = False

        win = (f"Phase 2 — Set end — fly {fly_id}  start={jump_frame}  "
               f"[navigate to flip-back, e/Enter=confirm end, q=cancel]")
        _cv2.namedWindow(win, _cv2.WINDOW_NORMAL)
        _cv2.createTrackbar('Frame', win, current - p2_start, p2_len, lambda _: None)
        _cv2.createTrackbar('End',   win, man_end - p2_start, p2_len, lambda _: None)

        while True:
            if not playing:
                tb      = _cv2.getTrackbarPos('Frame', win)
                current = p2_start + max(0, min(p2_len, tb))

            man_end = p2_start + _cv2.getTrackbarPos('End', win)
            if man_end <= jump_frame:
                man_end = jump_frame + 1
                _cv2.setTrackbarPos('End', win, 1)

            cap.set(_cv2.CAP_PROP_POS_FRAMES, current)
            ret, img = cap.read()
            if not ret:
                break
            h, w = img.shape[:2]

            if jump_frame <= current < man_end:
                _cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 200), 6)
                _cv2.putText(img, 'FLIPPED', (10, 60),
                             _cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 3, _cv2.LINE_AA)
            elif current == jump_frame:
                _cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 215, 255), 4)
            _gui_draw_arrows(img, current, full_lookup, fly_id, id_colors, arrow_len, _cv2)
            dur_s = (man_end - jump_frame) / fps
            _cv2.putText(img,
                         f"fly={fly_id}  frame={current}  "
                         f"start={jump_frame}  end={man_end}  ({dur_s:.1f}s)",
                         (10, 25), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, _cv2.LINE_AA)
            _cv2.putText(img,
                         "e:set_end  Enter:confirm  arrows:step  space:play  q:cancel",
                         (10, h - 10), _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, _cv2.LINE_AA)
            _cv2.imshow(win, img)

            _, key, left, right, up, down = _gui_read_key(
                _cv2, max(1, int(1000 / fps)) if playing else 30)

            if playing:
                current = min(p2_end, current + 1)
                _cv2.setTrackbarPos('Frame', win, current - p2_start)
                if current == p2_end:
                    playing = False

            if   key == 13:             confirmed = True; break
            elif key in (ord('q'), 27): break
            elif key == ord('e'):
                man_end = current
                _cv2.setTrackbarPos('End', win, man_end - p2_start)
            elif key == ord(' '):       playing = not playing
            elif left:
                current = max(p2_start, current - 1);  playing = False
                _cv2.setTrackbarPos('Frame', win, current - p2_start)
            elif right:
                current = min(p2_end,   current + 1);  playing = False
                _cv2.setTrackbarPos('Frame', win, current - p2_start)
            elif up:
                current = max(p2_start, current - 30); playing = False
                _cv2.setTrackbarPos('Frame', win, current - p2_start)
            elif down:
                current = min(p2_end,   current + 30); playing = False
                _cv2.setTrackbarPos('Frame', win, current - p2_start)

        _cv2.destroyWindow(win)
        return confirmed, man_end

    def _make_record(jump_frame, jump_in_rad, man_end):
        dur_s     = (man_end - jump_frame) / fps
        end_match = np.where(frames == man_end)[0]
        jout      = float(diffs[end_match[0]]) if len(end_match) > 0 else np.nan
        s_match   = np.where(frames == jump_frame)[0]
        if len(s_match) > 0 and len(end_match) > 0:
            seg_inner = diffs[s_match[0] + 1: end_match[0]]
            seg_std   = float(np.nanstd(seg_inner)) if len(seg_inner) > 0 else 0.0
        else:
            seg_std = np.nan
        seg_mask = ((df['id'] == fly_id) &
                    (df['frame'] >= jump_frame) & (df['frame'] < man_end))
        seg_vel  = df.loc[seg_mask, 'vel'] if 'vel' in df.columns else pd.Series(dtype=float)
        return {
            'fly_id':          fly_id,
            'start_frame':     jump_frame,
            'end_frame':       man_end,
            'duration_frames': man_end - jump_frame,
            'duration_sec':    round(dur_s, 2),
            'jump_in_rad':     round(jump_in_rad, 3),
            'jump_out_rad':    round(jout, 3) if not np.isnan(jout) else np.nan,
            'segment_ori_std': round(seg_std, 4) if not np.isnan(seg_std) else np.nan,
            'mean_vel':        round(float(seg_vel.mean()), 3) if len(seg_vel) > 0 else np.nan,
        }

    # ─────────────────────────────────────────────────────────────────────────

    event_records = []
    consumed      = set()   # positions in jump_idxs already handled
    n_total       = len(jump_idxs)

    if interactive_threshold_sec is not None:
        # ── Threshold mode: pair-based sequential pass ────────────────────────
        # Short pairs (≤ threshold) are auto-corrected; long pairs enter the GUI.
        # When the user shortens the end vs the auto-detected pair end, the freed
        # jump is re-evaluated immediately: ji = je (not je + 1) restarts the loop
        # from the orphaned position, which then pairs with whatever follows it.
        print(f"  fly {fly_id}: {n_total} jump(s) — "
              f"auto≤{interactive_threshold_sec}s, interactive>{interactive_threshold_sec}s")

        ji = 0
        while ji < n_total:
            if ji in consumed:
                ji += 1
                continue

            # find next unconsumed partner
            je = ji + 1
            while je < n_total and je in consumed:
                je += 1
            if je >= n_total:
                break

            s_idx = jump_idxs[ji]
            e_idx = jump_idxs[je]
            inner = diffs[s_idx + 1:e_idx]
            seg_std = float(np.nanstd(inner)) if len(inner) > 0 else 0.0

            if seg_std >= stability_threshold_rad:
                ji += 1
                continue

            sf    = int(frames[s_idx])
            ef    = int(frames[e_idx])
            dur_s = (ef - sf) / fps

            if dur_s <= interactive_threshold_sec:
                seg_mask = ((df['id'] == fly_id) &
                            (df['frame'] >= sf) & (df['frame'] < ef))
                seg_vel  = (df.loc[seg_mask, 'vel']
                            if 'vel' in df.columns else pd.Series(dtype=float))
                event_records.append({
                    'fly_id':          fly_id,
                    'start_frame':     sf,
                    'end_frame':       ef,
                    'duration_frames': ef - sf,
                    'duration_sec':    round(dur_s, 2),
                    'jump_in_rad':     round(float(diffs[s_idx]), 3),
                    'jump_out_rad':    round(float(diffs[e_idx]), 3),
                    'segment_ori_std': round(seg_std, 4),
                    'mean_vel': round(float(seg_vel.mean()), 3) if len(seg_vel) > 0 else np.nan,
                })
                consumed.add(ji)
                consumed.add(je)
                print(f"    jump@{sf:6d}→{ef}  ({dur_s:.2f}s) → AUTO-CORRECTED")
                ji = je + 1
                continue

            # long: interactive phase 1 + phase 2
            jump_diff = float(diffs[s_idx])
            p1_start  = max(0, sf - pad_frames)
            p1_end    = min(n_video_frames - 1, sf + pad_frames)
            label     = _phase1(ji, n_total, sf, jump_diff, p1_start, p1_end)

            consumed.add(ji)
            if label != 'artifact':
                print(f"    jump@{sf:6d}  diff={jump_diff:+.3f}  "
                      f"→ {'TRUE FLIP' if label == 'true_flip' else 'skipped'}")
                ji += 1
                continue

            print(f"    jump@{sf:6d}  diff={jump_diff:+.3f}  → ARTIFACT — set end frame")
            confirmed, man_end = _phase2(sf, default_end=ef)

            if not confirmed or man_end <= sf:
                print(f"    → cancelled — no correction for jump@{sf}")
                ji += 1
                continue

            event_records.append(_make_record(sf, jump_diff, man_end))
            print(f"    → interval [{sf}, {man_end})  ({(man_end - sf) / fps:.1f}s)")

            # Determine fate of je:
            #   man_end >= ef → je falls inside or at the boundary; consume it
            #   man_end <  ef → ef is freed; restart loop from je for immediate re-pairing
            if man_end >= ef:
                consumed.add(je)
                ji = je + 1
            else:
                print(f"    → end shortened past suggested ({ef}); "
                      f"freed jump@{ef} will be re-evaluated next")
                ji = je

    else:
        # ── No-threshold mode: review each jump individually ─────────────────
        print(f"  fly {fly_id}: reviewing {n_total} jump(s)  "
              f"[a/Enter=artifact  t=true_flip  q/ESC=skip]")

        for ji, jidx in enumerate(jump_idxs):
            if ji in consumed:
                print(f"    jump@{int(frames[jidx]):6d}  → auto-consumed as end of previous interval")
                continue

            jump_frame = int(frames[jidx])
            jump_diff  = float(diffs[jidx])
            p1_start   = max(0, jump_frame - pad_frames)
            p1_end     = min(n_video_frames - 1, jump_frame + pad_frames)
            label      = _phase1(ji, n_total, jump_frame, jump_diff, p1_start, p1_end)

            if label != 'artifact':
                print(f"    jump@{jump_frame:6d}  diff={jump_diff:+.3f}  "
                      f"→ {'TRUE FLIP' if label == 'true_flip' else 'skipped'}")
                continue

            print(f"    jump@{jump_frame:6d}  diff={jump_diff:+.3f}  → ARTIFACT — set end frame")
            confirmed, man_end = _phase2(jump_frame, default_end=jump_frame + pad_frames)

            if not confirmed or man_end <= jump_frame:
                print(f"    → cancelled — no correction for jump@{jump_frame}")
                continue

            print(f"    → interval [{jump_frame}, {man_end})  ({(man_end - jump_frame) / fps:.1f}s)")

            # auto-consume any later jump whose frame matches the chosen end
            for jk, other_jidx in enumerate(jump_idxs):
                if jk > ji and int(frames[other_jidx]) == man_end:
                    consumed.add(jk)
                    print(f"    → end frame {man_end} matches jump {jk+1} — auto-consumed")

            event_records.append(_make_record(jump_frame, jump_diff, man_end))

    cap.release()
    _cv2.destroyAllWindows()
    return event_records


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
        interactive_threshold_sec -- when set (and interactive=True), auto-correct events
                                     shorter than this threshold; only interactively review
                                     events that exceed it. Suggested end frames from greedy
                                     pairing are pre-filled in the phase-2 GUI (default: None)

    Returns:
        events -- DataFrame with columns:
                  fly_id, start_frame, end_frame, duration_frames, duration_sec,
                  jump_in_rad, jump_out_rad, segment_ori_std, mean_vel
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
