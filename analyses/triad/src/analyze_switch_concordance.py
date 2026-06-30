#!/usr/bin/env python3
"""
Analyze concordance between manually annotated and auto-detected switch events.

For each acquisition, extracts manual (switching col) and auto (courtship_auto_switch)
switch events, matches them within a frame window, then reports a confusion matrix
(TP/FP/FN) and a violin of the frame-offset residual for concordant pairs.

Manual annotation is treated as ground truth.

Usage:
    python analyses/triad/src/analyze_switch_concordance.py <rootdir> [--window 10]
"""

import sys
import os
import re
import glob
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import libs.plotting as putil
from analyses.triad.src.data_io import parse_acquisition_metadata
from analyses.triad.src import util as tutil
from analyses.triad.src import corrections as C

PLOT_STYLE  = 'dark'
REVIEW_LOG  = 'fp_review_log.json'
putil.set_sns_style(style=PLOT_STYLE, min_fontsize=12)


def _load_review_log(rootdir):
    """Return {acq: timestamp_str} for acquisitions already FP-reviewed."""
    path = os.path.join(rootdir, REVIEW_LOG)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_review_log(rootdir, log):
    path = os.path.join(rootdir, REVIEW_LOG)
    with open(path, 'w') as f:
        json.dump(log, f, indent=2)


# ── Event extraction ──────────────────────────────────────────────────────────

def _extract_events(df, method):
    """Return list of (acquisition, fly_id, frame) for switch events in df."""
    events = []
    if method == 'manual':
        if 'switching' not in df.columns:
            return events
        mask = (df['switching'] == df['id']) & (df['switching'] != -1)
        rows = df[mask].drop_duplicates(subset=['frame', 'id'])
    else:
        auto_col = ('courtship_auto_switch' if 'courtship_auto_switch' in df.columns
                    else 'courtship_switch' if 'courtship_switch' in df.columns
                    else None)
        if auto_col is None:
            return events
        rows = df[df[auto_col] == 1].drop_duplicates(subset=['frame', 'id'])

    for _, row in rows.iterrows():
        events.append((row['acquisition'], int(row['id']), int(row['frame'])))
    return events


# ── Matching ──────────────────────────────────────────────────────────────────

def _match_events(manual_events, auto_events, window):
    """
    Greedy nearest-neighbour matching of manual → auto events within `window` frames.

    Returns:
        residuals  -- list of (auto_frame - manual_frame) for each TP pair
        fn_events  -- list of (acq, fly_id, frame) for manual events with no auto match
        fp_events  -- list of (acq, fly_id, frame) for auto events with no manual match
    """
    manual_by_fly = defaultdict(list)
    auto_by_fly   = defaultdict(list)
    for acq, fly_id, frame in manual_events:
        manual_by_fly[(acq, fly_id)].append(frame)
    for acq, fly_id, frame in auto_events:
        auto_by_fly[(acq, fly_id)].append(frame)

    residuals = []
    fn_events = []
    fp_events = []

    all_keys = set(manual_by_fly) | set(auto_by_fly)
    for key in all_keys:
        acq, fly_id = key
        m_frames = sorted(manual_by_fly[key])
        a_frames = sorted(auto_by_fly[key])

        matched_auto_idx = set()
        for mf in m_frames:
            best_idx, best_dist = None, window + 1
            for i, af in enumerate(a_frames):
                if i in matched_auto_idx:
                    continue
                dist = abs(af - mf)
                if dist <= window and dist < best_dist:
                    best_idx, best_dist = i, dist
            if best_idx is not None:
                matched_auto_idx.add(best_idx)
                residuals.append(a_frames[best_idx] - mf)
            else:
                fn_events.append((acq, fly_id, mf))

        for i, af in enumerate(a_frames):
            if i not in matched_auto_idx:
                fp_events.append((acq, fly_id, af))

    return residuals, fn_events, fp_events


# ── Example clip extraction ───────────────────────────────────────────────────

_ID_COLORS = {
    0: (255, 144,  30),
    1: ( 71,  99, 255),
    2: ( 50, 205,  50),
    3: (  0, 215, 255),
}


def _extract_example_clips(events, label, acq_dfs, rootdir, clip_dir, n_samples, fps,
                            pad_sec=2.0):
    """
    Write up to n_samples annotated video clips centred on the event frame.

    events   -- list of (acq, fly_id, frame)
    label    -- 'FN' or 'FP' (used in filename and on-screen text)
    acq_dfs  -- dict mapping acq key → DataFrame (for fly-position overlay)
    """
    try:
        import cv2
    except ImportError:
        print(f"  OpenCV not available — skipping {label} example clips")
        return

    if not events:
        return

    rng     = np.random.default_rng(42)
    indices = rng.choice(len(events), size=min(n_samples, len(events)), replace=False)
    sampled = [events[i] for i in sorted(indices)]

    pad_frames = int(pad_sec * fps)

    # Group by acquisition so each video is opened once
    by_acq = defaultdict(list)
    for acq, fly_id, frame in sampled:
        by_acq[acq].append((fly_id, frame))

    border_color = (50, 50, 220) if label == 'FN' else (50, 200, 220)

    for acq, fly_frames in by_acq.items():
        base_acq   = re.sub(r'_ch\d+$', '', acq)
        video_path = os.path.join(rootdir, 'raw_videos', base_acq, f'{base_acq}.avi')
        if not os.path.exists(video_path):
            print(f"  Video not found for {acq} — skipping {label} clips")
            continue

        df  = acq_dfs.get(acq)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Could not open {video_path}")
            continue

        n_vid    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps  = cap.get(cv2.CAP_PROP_FPS)

        for fly_id, event_frame in fly_frames:
            clip_start = max(0, event_frame - pad_frames)
            clip_end   = min(n_vid - 1, event_frame + pad_frames)

            # Build per-frame position lookup from the parquet
            frame_lookup = {}
            if df is not None:
                sub = df[df['frame'].between(clip_start, clip_end)]
                for fi, grp in sub.groupby('frame'):
                    pos, ori_map = {}, {}
                    for _, row in grp.drop_duplicates('id').iterrows():
                        fid = int(row['id'])
                        if pd.notna(row['pos_x']) and pd.notna(row['pos_y']):
                            pos[fid] = (int(row['pos_x']), int(row['pos_y']))
                        if pd.notna(row.get('ori', np.nan)):
                            ori_map[fid] = float(row['ori'])
                    frame_lookup[int(fi)] = (pos, ori_map)

            clip_name = f'{label}_{acq}_fly{fly_id}_f{event_frame}.mp4'
            clip_path = os.path.join(clip_dir, clip_name)

            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            out = cv2.VideoWriter(clip_path,
                                  cv2.VideoWriter_fourcc(*'avc1'),
                                  vid_fps, (frame_w, frame_h))

            for fi in range(clip_start, clip_end + 1):
                ret, img = cap.read()
                if not ret:
                    break

                if fi in frame_lookup:
                    pos, ori_map = frame_lookup[fi]
                    for fid, (px, py) in pos.items():
                        color = _ID_COLORS.get(fid, (200, 200, 200))
                        cv2.circle(img, (px, py), 7, color, -1)
                        if fid in ori_map:
                            ex = int(px + 20 * np.cos(ori_map[fid]))
                            ey = int(py + 20 * np.sin(ori_map[fid]))
                            cv2.arrowedLine(img, (px, py), (ex, ey),
                                            color, 2, tipLength=0.3)
                        cv2.putText(img, f'id={fid}', (px + 10, py - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if fi == event_frame:
                    cv2.rectangle(img, (0, 0), (frame_w-1, frame_h-1), border_color, 6)
                    cv2.putText(img, f'{label}  fly={fly_id}  f={event_frame}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                border_color, 2, cv2.LINE_AA)

                offset_s = (fi - event_frame) / fps
                cv2.putText(img, f'f={fi}  ({offset_s:+.2f}s)',
                            (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1, cv2.LINE_AA)
                out.write(img)

            out.release()
            print(f"  Saved {label} clip: {clip_name}")

        cap.release()


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_concordance(tp, n_fn, n_fp, residuals, window, fps, figdir, n_confirmed=0):
    has_review = n_confirmed > 0
    fig, axes  = plt.subplots(1, 2, figsize=(14 if has_review else 11, 5))

    # ── Left: confusion matrix bar chart (before / after if review was run) ───
    ax     = axes[0]
    labels = ['TP\n(concordant)', 'FN\n(missed by auto)', 'FP\n(spurious auto)']
    colors = ['#4caf50', '#f44336', '#ff9800']

    if has_review:
        before = [tp,                n_fn, n_fp]
        after  = [tp + n_confirmed,  n_fn, n_fp - n_confirmed]
        x, w   = np.arange(len(labels)), 0.35

        bars_b = ax.bar(x - w/2, before, w, color=colors, alpha=0.4,
                        edgecolor='none', label='Before review')
        bars_a = ax.bar(x + w/2, after,  w, color=colors, alpha=0.9,
                        edgecolor='none', label='After review')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=9)

        y_max = max(before + after)
        for bar, cnt in zip(bars_b, before):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.02,
                    str(cnt), ha='center', va='bottom', fontsize=11, alpha=0.55)
        for bar, cnt in zip(bars_a, after):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_max * 0.02,
                    str(cnt), ha='center', va='bottom', fontsize=11, fontweight='bold')

        def _metrics(t, fn, fp):
            prec = t / (t + fp) if (t + fp) > 0 else float('nan')
            rec  = t / (t + fn) if (t + fn) > 0 else float('nan')
            return prec, rec

        prec_b, rec_b = _metrics(tp,              n_fn, n_fp)
        prec_a, rec_a = _metrics(tp + n_confirmed, n_fn, n_fp - n_confirmed)
        ax.set_title(
            f'Switch concordance  (window = ±{window} fr)\n'
            f'Before: prec={prec_b:.2f}  rec={rec_b:.2f}    '
            f'After: prec={prec_a:.2f}  rec={rec_a:.2f}',
            fontsize=9)
        ax.set_ylim(0, y_max * 1.22)
    else:
        counts = [tp, n_fn, n_fp]
        bars   = ax.bar(labels, counts, color=colors, edgecolor='none', width=0.5)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    str(count), ha='center', va='bottom', fontsize=13, fontweight='bold')
        total_manual = tp + n_fn
        total_auto   = tp + n_fp
        precision = tp / total_auto   if total_auto   > 0 else float('nan')
        recall    = tp / total_manual if total_manual > 0 else float('nan')
        ax.set_title(
            f'Switch detection concordance  (window = ±{window} frames)\n'
            f'precision = {precision:.2f}   recall = {recall:.2f}',
            fontsize=10)
        ax.set_ylim(0, max(counts) * 1.18)

    ax.set_ylabel('# switch events')

    # ── Right: violin of frame offset for TP pairs ────────────────────────────
    ax = axes[1]
    if residuals:
        res = np.array(residuals)
        parts = ax.violinplot(res, positions=[0], showmedians=True,
                              showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('dodgerblue')
            pc.set_alpha(0.6)
        ax.scatter(np.zeros(len(res)) + np.random.uniform(-0.05, 0.05, len(res)),
                   res, s=25, color='white', alpha=0.7, zorder=3)
        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)
        ax.set_xticks([0])
        ax.set_xticklabels([f'TP pairs\n(n={tp})'])
        ax.set_ylabel('auto frame − manual frame')
        ax.set_title(
            f'Frame offset for concordant switches\n'
            f'median = {np.median(res):.1f} fr  '
            f'({np.median(res)/fps*1000:.0f} ms)   '
            f'std = {np.std(res):.1f} fr', fontsize=10)

        # secondary x-axis in seconds
        ax2 = ax.secondary_yaxis('right',
                                  functions=(lambda f: f / fps, lambda s: s * fps))
        ax2.set_ylabel('offset (s)')
    else:
        ax.text(0.5, 0.5, 'No concordant events', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    savepath = os.path.join(figdir, f'switch_concordance_w{window}.png')
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {savepath}")
    plt.close(fig)


# ── Interactive FP review ─────────────────────────────────────────────────────

def _overlay_lookup(df, clip_start, clip_end):
    """Build {frame: ({fly_id: (px,py)}, {fly_id: ori})} for video overlay."""
    lookup = {}
    if df is None:
        return lookup
    sub = df[df['frame'].between(clip_start, clip_end)]
    for fi, grp in sub.groupby('frame'):
        pos, oris = {}, {}
        for _, row in grp.drop_duplicates('id').iterrows():
            fid = int(row['id'])
            if pd.notna(row.get('pos_x')) and pd.notna(row.get('pos_y')):
                pos[fid] = (int(row['pos_x']), int(row['pos_y']))
            if pd.notna(row.get('ori', np.nan)):
                oris[fid] = float(row['ori'])
        lookup[int(fi)] = (pos, oris)
    return lookup


def _annotate_frame(img, fi, event_frame, actor_id, overlay_lookup, fps,
                    frame_idx, n_total, frame_w, frame_h,
                    instructions=None):
    """Draw fly overlays, event border, and text annotations onto img (in place)."""
    import cv2

    if fi in overlay_lookup:
        pos, oris = overlay_lookup[fi]
        for fid, (px, py) in pos.items():
            color = _ID_COLORS.get(fid, (200, 200, 200))
            cv2.circle(img, (px, py), 7, color, -1)
            if fid in oris:
                ex = int(px + 22 * np.cos(oris[fid]))
                ey = int(py + 22 * np.sin(oris[fid]))
                cv2.arrowedLine(img, (px, py), (ex, ey), color, 2, tipLength=0.3)
            label = f'id={fid}' + (' (actor)' if fid == actor_id else '')
            cv2.putText(img, label, (px + 10, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if fi == event_frame:
        cv2.rectangle(img, (0, 0), (frame_w - 1, frame_h - 1), (50, 200, 220), 5)

    offset_s = (fi - event_frame) / fps
    auto_marker = '  ◀ AUTO' if fi == event_frame else f'  (auto@{event_frame})'
    cv2.putText(img, f'FP [{frame_idx+1}/{n_total}]  fly={actor_id}  '
                f'f={fi}  ({offset_s:+.2f}s){auto_marker}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 200, 220), 2, cv2.LINE_AA)

    if instructions:
        y = frame_h - 12
        for line in reversed(instructions):
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 100), 1, cv2.LINE_AA)
            y -= 22


def _read_key(cv2, delay):
    key_raw = cv2.waitKey(delay)
    key     = key_raw & 0xFF
    left  = key in (81, 2)  or key_raw in (63234, 65361)
    right = key in (83, 3)  or key_raw in (63235, 65363)
    up_k  = key == 82       or key_raw in (63232, 65362)
    down  = key in (84, 1)  or key_raw in (63233, 65364)
    return key_raw, key, left, right, up_k, down


def _interactive_review_fp(fp_events, acq_dfs, rootdir, fps, pad_sec=3.0):
    """
    Two-phase interactive review of FP switch events.

    Phase 1 — Is this a real switch?
        Auto-plays clip.  SPACE pauses/replays.
        y=yes (→ phase 2)  n=reject  s=skip  q=quit

    Phase 2 — Which frame is the switch?
        Trackbar + arrow key navigation.  Starts at auto-detected frame.
        Enter/y=confirm frame  b=back to phase 1  q=quit

    Target assignment is intentionally left to review_switch_targets.py,
    which handles all switches (manual + confirmed FP) in one consistent pass.

    Returns
    -------
    confirmed : dict  {(acq, fly_id, confirmed_frame): None}
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not available — cannot run interactive review.")
        return {}, set()

    WIN   = 'Switch Review  [FP]'
    delay = max(1, int(1000 / fps))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    confirmed      = {}
    newly_reviewed = set()
    quit_requested = False
    n = len(fp_events)

    # Count how many FP events belong to each acquisition so we know when
    # all of an acquisition's events have been processed.
    acq_total     = defaultdict(int)
    acq_processed = defaultdict(int)
    for acq, _, _ in fp_events:
        acq_total[acq] += 1

    for idx, (acq, fly_id, event_frame) in enumerate(fp_events):
        print(f"\n[{idx+1}/{n}]  {acq}  fly={fly_id}  auto_frame={event_frame}")

        base_acq   = re.sub(r'_ch\d+$', '', acq)
        video_path = os.path.join(rootdir, 'raw_videos', base_acq, f'{base_acq}.avi')
        if not os.path.exists(video_path):
            print("  No video found — skipping.")
            continue

        df = acq_dfs.get(acq)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("  Cannot open video — skipping.")
            continue

        n_vid    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pad_f    = int(pad_sec * fps)
        c_start  = max(0, event_frame - pad_f)
        c_end    = min(n_vid - 1, event_frame + pad_f)
        clip_len = c_end - c_start

        lookup = _overlay_lookup(df, c_start, c_end)
        cv2.createTrackbar('Frame', WIN, 0, clip_len, lambda _: None)

        confirmed_frame = event_frame
        start_phase     = 1

        while True:   # outer loop — re-entered when user presses b

            # ── Phase 1: confirm this is a real switch ────────────────────
            if start_phase == 1:
                p1_pos     = c_start
                p1_playing = True
                p1_result  = None

                while p1_result is None:
                    if p1_playing:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, p1_pos)
                        ret, img = cap.read()
                        if not ret:
                            p1_playing = False
                        else:
                            _annotate_frame(
                                img, p1_pos, event_frame, fly_id, lookup,
                                fps, idx, n, frame_w, frame_h,
                                instructions=['[1/3] Real switch?  '
                                              'y=yes  n=no  s=skip  q=quit  SPACE=pause'])
                            cv2.imshow(WIN, img)
                            key = cv2.waitKey(delay) & 0xFF
                            p1_pos += 1
                            if p1_pos > c_end:
                                p1_playing = False
                            if key == ord('y'):   p1_result = 'confirm'
                            elif key == ord('n'): p1_result = 'reject'
                            elif key == ord('s'): p1_result = 'skip'
                            elif key == ord('q'): p1_result = 'quit'
                            elif key == ord(' '): p1_playing = False
                    else:
                        # paused or clip ended — show still, wait for decision
                        still_pos = min(c_end, max(c_start, p1_pos - 1))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, still_pos)
                        ret, img = cap.read()
                        if ret:
                            _annotate_frame(
                                img, still_pos, event_frame, fly_id, lookup,
                                fps, idx, n, frame_w, frame_h,
                                instructions=['[1/3] Real switch?  '
                                              'y=yes  n=no  s=skip  q=quit  SPACE=replay'])
                            cv2.imshow(WIN, img)
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('y'):   p1_result = 'confirm'
                        elif key == ord('n'): p1_result = 'reject'
                        elif key == ord('s'): p1_result = 'skip'
                        elif key == ord('q'): p1_result = 'quit'
                        elif key == ord(' '): p1_playing = True; p1_pos = c_start

                if p1_result == 'quit':
                    quit_requested = True; break
                if p1_result in ('reject', 'skip'):
                    print(f"  {'REJECTED' if p1_result == 'reject' else 'SKIPPED'}")
                    break   # done with this event

            # ── Phase 2: navigate to exact switch frame ───────────────────
            cv2.setTrackbarPos('Frame', WIN, confirmed_frame - c_start)
            p2_current = confirmed_frame
            p2_playing = False
            p2_result  = None

            while p2_result is None:
                if not p2_playing:
                    tb         = cv2.getTrackbarPos('Frame', WIN)
                    p2_current = c_start + max(0, min(clip_len, tb))

                cap.set(cv2.CAP_PROP_POS_FRAMES, p2_current)
                ret, img = cap.read()
                if not ret:
                    break

                _annotate_frame(
                    img, p2_current, event_frame, fly_id, lookup,
                    fps, idx, n, frame_w, frame_h,
                    instructions=['[2/2] Navigate to switch frame  '
                                  'Enter/y=confirm  b=back  q=quit',
                                  'arrows/trackbar:navigate  SPACE:play/pause'])
                cv2.imshow(WIN, img)

                _, key, left, right, up_k, down = _read_key(
                    cv2, delay if p2_playing else 30)

                if p2_playing:
                    p2_current = min(c_end, p2_current + 1)
                    cv2.setTrackbarPos('Frame', WIN, p2_current - c_start)
                    if p2_current == c_end:
                        p2_playing = False

                if key in (13, ord('y')):  p2_result = 'confirm'
                elif key == ord('b'):      p2_result = 'back'
                elif key == ord('q'):      p2_result = 'quit'
                elif key == ord(' '):      p2_playing = not p2_playing
                elif left:  p2_current = max(c_start, p2_current-1);  p2_playing = False; cv2.setTrackbarPos('Frame', WIN, p2_current-c_start)
                elif right: p2_current = min(c_end,   p2_current+1);  p2_playing = False; cv2.setTrackbarPos('Frame', WIN, p2_current-c_start)
                elif up_k:  p2_current = max(c_start, p2_current-30); p2_playing = False; cv2.setTrackbarPos('Frame', WIN, p2_current-c_start)
                elif down:  p2_current = min(c_end,   p2_current+30); p2_playing = False; cv2.setTrackbarPos('Frame', WIN, p2_current-c_start)

            if p2_result == 'quit':
                quit_requested = True; break
            if p2_result == 'back':
                start_phase = 1; continue

            confirmed_frame = p2_current

            confirmed[(acq, fly_id, confirmed_frame)] = None
            print(f"  CONFIRMED  frame={confirmed_frame} (auto={event_frame})")
            break   # done with this event

        cap.release()

        if quit_requested:
            break

        acq_processed[acq] += 1
        if acq_processed[acq] == acq_total[acq]:
            newly_reviewed.add(acq)

    cv2.destroyAllWindows()
    return confirmed, newly_reviewed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir', help='Root directory containing processed_mats/')
    parser.add_argument('--window', type=int, default=30,
                        help='Frame window for concordant match (default: 30)')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of FP/FN example clips to extract (default: 5)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactively review FP events and update parquets / actions.mat')
    parser.add_argument('--pad-sec', type=float, default=3.0,
                        help='Seconds of video padding around event in review window (default: 3.0)')
    parser.add_argument('--list-reviewed', action='store_true',
                        help=f'Print acquisitions already marked as FP-reviewed in {REVIEW_LOG} and exit')
    args = parser.parse_args()

    if args.list_reviewed:
        log = _load_review_log(args.rootdir)
        if not log:
            print(f"No acquisitions recorded in {REVIEW_LOG} yet.")
        else:
            print(f"{len(log)} acquisition(s) marked as FP-reviewed:")
            for acq, ts in sorted(log.items()):
                print(f"  {acq}  ({ts})")
        sys.exit(0)

    processedmat_dir = os.path.join(args.rootdir, 'processed_mats')
    figdir = os.path.join(args.rootdir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(processedmat_dir, '*.parquet')))
    if not parquet_files:
        print(f"No parquet files in {processedmat_dir}")
        sys.exit(1)

    fps = 60  # fallback; overwritten from data below
    all_manual, all_auto = [], []
    acq_dfs = {}  # acq key → df, kept for clip extraction

    print(f"Loading {len(parquet_files)} parquet files...\n")
    skipped = 0
    for fp in parquet_files:
        acq = os.path.splitext(os.path.basename(fp))[0]
        try:
            meta = parse_acquisition_metadata(acq)
        except ValueError:
            meta = None
        if meta is None or meta['triad_type'] != 'MFF':
            skipped += 1
            continue
        df = pd.read_parquet(fp)
        if 'FPS' in df.columns:
            fps = int(df['FPS'].iloc[0])
        manual = _extract_events(df, 'manual')
        auto   = _extract_events(df, 'auto')
        if manual or auto:
            print(f"  {acq[-55:]:55s}  manual={len(manual):3d}  auto={len(auto):3d}")
        all_manual.extend(manual)
        all_auto.extend(auto)
        acq_dfs[acq] = df
    if skipped:
        print(f"  ({skipped} non-MFF parquet(s) skipped)")

    print(f"\nTotal  manual={len(all_manual)}  auto={len(all_auto)}")

    residuals, fn_events, fp_events = _match_events(all_manual, all_auto, args.window)
    tp   = len(residuals)
    n_fn = len(fn_events)
    n_fp = len(fp_events)

    print(f"\nConfusion matrix (window = ±{args.window} frames = "
          f"±{args.window/fps:.2f}s at {fps} fps):")
    print(f"  TP  (concordant)         : {tp}")
    print(f"  FN  (missed by auto)     : {n_fn}")
    print(f"  FP  (spurious auto)      : {n_fp}")

    total_manual = tp + n_fn
    total_auto   = tp + n_fp
    if total_auto > 0:
        print(f"  Precision                : {tp/total_auto:.2f}")
    if total_manual > 0:
        print(f"  Recall                   : {tp/total_manual:.2f}")

    if residuals:
        res = np.array(residuals)
        print(f"\nFrame offset (auto − manual) for TP pairs:")
        print(f"  median = {np.median(res):.1f} fr  "
              f"({np.median(res)/fps*1000:.0f} ms)")
        print(f"  mean   = {np.mean(res):.1f} fr")
        print(f"  std    = {np.std(res):.1f} fr")
        print(f"  range  = [{res.min():.0f}, {res.max():.0f}] fr")

    # ── Example clips ─────────────────────────────────────────────────────────
    clip_dir = os.path.join(figdir, f'switch_concordance_examples_w{args.window}')
    os.makedirs(clip_dir, exist_ok=True)
    print(f"\nExtracting up to {args.n_samples} FN and {args.n_samples} FP example clips...")
    #_extract_example_clips(fn_events, 'FN', acq_dfs, args.rootdir, clip_dir,
    #                       args.n_samples, fps)
    #_extract_example_clips(fp_events, 'FP', acq_dfs, args.rootdir, clip_dir,
    #                       args.n_samples, fps)

    # ── Interactive FP review ─────────────────────────────────────────────────
    confirmed = {}
    if args.interactive:
        if not fp_events:
            print("\nNo FP events to review.")
        else:
            review_log   = _load_review_log(args.rootdir)
            fp_to_review = [(a, f, fr) for a, f, fr in fp_events
                            if a not in review_log]
            skipped_acqs = {a for a, _, _ in fp_events} & set(review_log)
            if skipped_acqs:
                print(f"\nSkipping {len(skipped_acqs)} already-reviewed acquisition(s) "
                      f"(delete {REVIEW_LOG} or remove entries to re-review).")
            if not fp_to_review:
                print("All FP acquisitions already reviewed.")
            else:
                n_new_acqs = len({a for a, _, _ in fp_to_review})
                print(f"\nStarting interactive review of {len(fp_to_review)} FP event(s) "
                      f"across {n_new_acqs} acquisition(s)...")
                confirmed, newly_reviewed = _interactive_review_fp(
                    fp_to_review, acq_dfs, args.rootdir, fps, pad_sec=args.pad_sec)

                if newly_reviewed:
                    now = datetime.now().isoformat(timespec='seconds')
                    for acq in newly_reviewed:
                        review_log[acq] = now
                    _save_review_log(args.rootdir, review_log)
                    print(f"  Marked {len(newly_reviewed)} acquisition(s) as reviewed "
                          f"in {REVIEW_LOG}.")

                if confirmed:
                    print(f"\n{len(confirmed)} event(s) confirmed as real switches.")
                    cdir = C.corrections_dir(args.rootdir)
                    written = C.write_switches_manifest_from_confirmed(cdir, confirmed)
                    for acq, n in sorted(written.items()):
                        print(f"  {acq}: {n} confirmed switch(es) in manifest")
                    print(f"  Wrote switch manifest(s) → {cdir}")
                    print("  Next: review_switch_targets.py, then apply_corrections.py "
                          "to rebuild reviewed_mats.")
                else:
                    print("\nNo events confirmed — no files updated.")

    _plot_concordance(tp, n_fn, n_fp, residuals, args.window, fps, figdir,
                      n_confirmed=len(confirmed))


if __name__ == '__main__':
    main()
