#!/usr/bin/env python3
"""
Interactive correction of courtship_target around manually annotated switch events.

For each MFF parquet with a 'switching' column, opens an OpenCV window showing
~2 s before and after each annotated switch. Navigate with the trackbar or arrow
keys, then press a number key to set the target fly ID:

    Phase 1 — BEFORE the switch: press 0 / 1 / 2 (or Enter to skip)
    Phase 2 — AFTER  the switch: press 0 / 1 / 2 (or Enter to skip)
    q (in Phase 1) — skip this switch entirely
    space — play / pause

Corrected parquets are written to {rootdir}/processed_mats_corrected/.
The original automated assignment is preserved in 'courtship_target_pre_correction'.
Already-corrected files are skipped unless --redo is passed.

Usage:
    python analyses/triad/src/review_switch_targets.py <rootdir> [--redo]
"""

import sys
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from analyses.triad.src.data_io import parse_acquisition_metadata
from analyses.triad.src import util as tutil

PAD_SEC    = 2.0
ACTION_COL = 'courtship'
TARGET_COL = f'{ACTION_COL}_target'

_ID_COLORS = {      # BGR
    0: (255, 144,  30),
    1: ( 71,  99, 255),
    2: ( 50, 205,  50),
    3: (  0, 215, 255),
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _strip_ch(acq_key):
    return re.sub(r'_ch\d+$', '', acq_key)


def _find_video(rootdir, acq_key):
    base = _strip_ch(acq_key)
    path = os.path.join(rootdir, 'raw_videos', base, f'{base}.avi')
    return path if os.path.exists(path) else None


def _read_key(cv2, delay):
    key_raw = cv2.waitKey(delay)
    key     = key_raw & 0xFF
    left  = key in (81, 2)  or key_raw in (63234, 65361)
    right = key in (83, 3)  or key_raw in (63235, 65363)
    up    = key == 82       or key_raw in (63232, 65362)
    down  = key in (84, 1)  or key_raw in (63233, 65364)
    return key_raw, key, left, right, up, down


# ── Frame lookup ──────────────────────────────────────────────────────────────

def _build_lookup(df):
    """Return {frame: {fly_id: (px, py, ori_rad|None)}}."""
    lookup = {}
    for frame, grp in df.groupby('frame'):
        fdata = {}
        for _, row in grp.drop_duplicates('id').iterrows():
            fid = int(row['id'])
            px, py, ori = row.get('pos_x'), row.get('pos_y'), row.get('ori')
            if pd.notna(px) and pd.notna(py):
                fdata[fid] = (float(px), float(py),
                              float(ori) if pd.notna(ori) else None)
        lookup[int(frame)] = fdata
    return lookup


# ── Video overlay ─────────────────────────────────────────────────────────────

def _draw_overlay(img, frame, switch_frame, lookup, after_tgt, fps, cv2):
    h, w = img.shape[:2]

    # Gold border at switch frame, green after
    if frame == switch_frame:
        cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 215, 255), 6)
        cv2.putText(img, 'SWITCH', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 3, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (0, 0), (w-1, h-1), (40, 160, 40), 3)

    # Fly markers
    if frame in lookup:
        for fid, (px, py, ori) in lookup[frame].items():
            color = _ID_COLORS.get(fid, (200, 200, 200))
            pxi, pyi = int(px), int(py)
            cv2.circle(img, (pxi, pyi), 10, color, 2)
            if ori is not None:
                ex = int(pxi + 22 * np.cos(ori))
                ey = int(pyi + 22 * np.sin(ori))
                cv2.arrowedLine(img, (pxi, pyi), (ex, ey), color, 2, tipLength=0.3)
            cv2.putText(img, f'id={fid}', (pxi + 12, pyi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Top bar
    offset_s = (frame - switch_frame) / fps
    cv2.putText(img, f'frame={frame}  switch@{switch_frame}  ({offset_s:+.2f}s)',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom bar
    cv2.putText(img, 'AFTER target: 0/1/2  |  u:prev switch  q/Enter:skip',
                (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, 'arrows/trackbar:navigate  space:play',
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    astr = f'AFTER: {after_tgt if after_tgt is not None else "?"}'
    cv2.putText(img, astr, (w - 155, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 150), 2, cv2.LINE_AA)


# ── Single-switch review ──────────────────────────────────────────────────────

def _review_switch(cap, n_video_frames, fps, switch_frame,
                   clip_start, clip_end, lookup, sw_idx, n_switches, cv2):
    """
    Single-phase review: identify the AFTER target by pressing 0/1/2.
    The BEFORE target is inferred by the caller as the other available target.

    u → signal caller to revisit the previous switch.
    q / Enter → skip this switch (no correction recorded).

    Returns (after_tgt, undo):
        after_tgt -- int fly ID, or None if skipped.
        undo      -- True if the user wants to go back to the previous switch.
    """
    clip_len  = clip_end - clip_start
    after_tgt = None
    undo      = False
    win       = f'Switch {sw_idx+1}/{n_switches} — frame {switch_frame}'

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Frame', win, switch_frame - clip_start, clip_len, lambda _: None)

    current = switch_frame
    playing = False

    while True:
        if not playing:
            tb      = cv2.getTrackbarPos('Frame', win)
            current = clip_start + max(0, min(clip_len, tb))

        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, img = cap.read()
        if not ret:
            break

        _draw_overlay(img, current, switch_frame, lookup, after_tgt, fps, cv2)
        cv2.imshow(win, img)

        _, key, left, right, up_k, down = _read_key(
            cv2, max(1, int(1000 / fps)) if playing else 30)

        if playing:
            current = min(clip_end, current + 1)
            cv2.setTrackbarPos('Frame', win, current - clip_start)
            if current == clip_end:
                playing = False

        if key in (ord('0'), ord('1'), ord('2'), ord('3')):
            after_tgt = key - ord('0')
            break
        elif key == ord('u'):
            undo = True
            break
        elif key in (13, ord('q'), 27):   # Enter or q: skip
            break
        elif key == ord(' '):
            playing = not playing
        elif left:
            current = max(clip_start, current - 1);  playing = False
            cv2.setTrackbarPos('Frame', win, current - clip_start)
        elif right:
            current = min(clip_end,   current + 1);  playing = False
            cv2.setTrackbarPos('Frame', win, current - clip_start)
        elif up_k:
            current = max(clip_start, current - 30); playing = False
            cv2.setTrackbarPos('Frame', win, current - clip_start)
        elif down:
            current = min(clip_end,   current + 30); playing = False
            cv2.setTrackbarPos('Frame', win, current - clip_start)

    cv2.destroyWindow(win)
    return after_tgt, undo


# ── Per-acquisition review ────────────────────────────────────────────────────

def _review_acquisition(acq_key, df, video_path, processedmat_dir, backup_dir):
    try:
        import cv2
    except ImportError:
        print('  OpenCV not available — skipping.')
        return

    fps        = int(df['FPS'].iloc[0]) if 'FPS' in df.columns else 60
    pad_frames = int(PAD_SEC * fps)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  Could not open {video_path}')
        return
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'  Building frame lookup ({len(df)} rows, fps={fps})...')
    lookup = _build_lookup(df)

    # Back up original parquet before any modification
    backup_path = os.path.join(backup_dir, f'{acq_key}.parquet')
    if not os.path.exists(backup_path):
        df.to_parquet(backup_path, index=False)
        print(f'  Backed up original → {backup_path}')

    # Keep pre-correction column in-df for easy comparison
    if 'courtship_target_pre_correction' not in df.columns:
        df['courtship_target_pre_correction'] = df[TARGET_COL].copy()

    # Collect focal flies that have switching annotations
    sw_any = df[(df['switching'] != -1) & df['switching'].notna()]
    focal_ids = sorted(sw_any['switching'].unique().astype(int))

    # Candidate (non-focal) target IDs for each focal fly, derived from pair strings
    focal_to_candidates = {}
    for fid in focal_ids:
        cands = set()
        for pair_str in df[df['id'] == fid]['pair'].dropna().unique():
            for part in pair_str.split('_'):
                pid = int(part)
                if pid != fid:
                    cands.add(pid)
        focal_to_candidates[fid] = sorted(cands)

    all_corrections = {}

    for focal_id in focal_ids:
        switch_frames = sorted(
            df[(df['switching'] == focal_id) & (df['switching'] != -1)]
            .drop_duplicates('frame')['frame'].tolist()
        )
        if not switch_frames:
            continue

        candidates = focal_to_candidates.get(focal_id, [])
        print(f'  Fly {focal_id}: {len(switch_frames)} switch(es)  '
              f'(candidate targets: {candidates})')
        corrections = []

        sw_idx = 0
        while sw_idx < len(switch_frames):
            sw_frame   = switch_frames[sw_idx]
            clip_start = max(0, sw_frame - pad_frames)
            clip_end   = min(n_video_frames - 1, sw_frame + pad_frames)
            print(f'    [{sw_idx+1}/{len(switch_frames)}] switch @ frame {sw_frame}  '
                  f'({clip_start}–{clip_end})')

            after_tgt, undo = _review_switch(
                cap, n_video_frames, fps, sw_frame,
                clip_start, clip_end, lookup,
                sw_idx, len(switch_frames), cv2)

            if undo:
                if sw_idx == 0:
                    print('    → already at first switch — cannot undo further')
                else:
                    print(f'    → undoing switch {sw_idx+1} → back to {sw_idx}')
                    corrections = [c for c in corrections if c[0] != switch_frames[sw_idx - 1]]
                    sw_idx -= 1
                continue

            if after_tgt is None:
                print(f'    → skipped')
            else:
                # Infer before_tgt as the other candidate target
                before_tgt = next((c for c in candidates if c != after_tgt), None)
                print(f'    → before={before_tgt}  after={after_tgt}')
                corrections.append((sw_frame, before_tgt, after_tgt))
            sw_idx += 1

        all_corrections[focal_id] = corrections

    cap.release()
    cv2.destroyAllWindows()

    for focal_id, corrections in all_corrections.items():
        tutil.apply_switch_corrections_to_target(df, focal_id, corrections,
                                                 action_col=ACTION_COL)

    out_path = os.path.join(processedmat_dir, f'{acq_key}.parquet')
    df.to_parquet(out_path, index=False)
    print(f'  Saved corrected → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir', help='Root directory containing processed_mats/ and raw_videos/')
    parser.add_argument('--redo', action='store_true',
                        help='Re-review acquisitions that already have a backup')
    args = parser.parse_args()

    processedmat_dir = os.path.join(args.rootdir, 'processed_mats')
    backup_dir       = os.path.join(args.rootdir, 'processed_mats_backup')
    os.makedirs(backup_dir, exist_ok=True)

    parquets = sorted(glob.glob(os.path.join(processedmat_dir, '*.parquet')))
    if not parquets:
        print(f'No parquet files in {processedmat_dir}')
        sys.exit(1)

    print(f'Found {len(parquets)} parquet(s) — scanning for MFF with switching annotations...\n')
    skipped = 0

    for fp in parquets:
        acq_key = os.path.splitext(os.path.basename(fp))[0]

        try:
            meta = parse_acquisition_metadata(acq_key)
        except ValueError:
            skipped += 1
            continue
        if meta['triad_type'] != 'MFF':
            skipped += 1
            continue

        backup_path = os.path.join(backup_dir, f'{acq_key}.parquet')
        if not args.redo and os.path.exists(backup_path):
            print(f'{acq_key}: already corrected (backup exists) — skipping (use --redo to redo)')
            continue

        df = pd.read_parquet(fp)

        if 'switching' not in df.columns or not (df['switching'] != -1).any():
            print(f'{acq_key}: no switching annotations — skipping')
            skipped += 1
            continue

        if TARGET_COL not in df.columns:
            print(f'{acq_key}: no {TARGET_COL} column — skipping')
            skipped += 1
            continue

        video_path = _find_video(args.rootdir, acq_key)
        if video_path is None:
            print(f'{acq_key}: video not found — skipping')
            skipped += 1
            continue

        n_sw = (df[(df['switching'] != -1) & df['switching'].notna()]
                .drop_duplicates(['frame', 'switching']).shape[0])
        print(f'\n{acq_key}  ({n_sw} switch event(s))')
        _review_acquisition(acq_key, df.copy(), video_path, processedmat_dir, backup_dir)

    if skipped:
        print(f'\n({skipped} parquet(s) skipped)')
    print('\nDone.')


if __name__ == '__main__':
    main()
