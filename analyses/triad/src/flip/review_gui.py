"""
OpenCV interactive review GUI for orientation-flip candidates.

Imported by flip_detect.detect_flip_events only when interactive=True. cv2 is
imported lazily inside the functions, so importing this module stays cheap and
the detection logic in flip_detect remains testable without a display.
"""
import numpy as np
import pandas as pd


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

    interactive_threshold_sec is required (detect_flip_events enforces this):
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

    # interactive_threshold_sec is required (enforced by detect_flip_events);
    # the guard is kept as defensive belt-and-suspenders.
    if interactive_threshold_sec is not None:
        # ── Pair-based sequential pass ────────────────────────────────────────
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

    cap.release()
    _cv2.destroyAllWindows()
    return event_records
