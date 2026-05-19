"""
Extract paired video clips around detected orientation-flip events:
  *_original.mp4  — orientation arrows from raw tracking data
  *_corrected.mp4 — orientation arrows after flip correction (π-shift)

Edit the CONFIG section, then run:
    python analyses/triad/src/extract_flip_clips.py

During the flip window the original clip shows a red border + "FLIP" label;
the corrected clip shows a green border + "CORRECTED" label.
The flipping fly is drawn as an open ring; all others are filled dots.
"""

import cv2
import math
import os
import numpy as np
import pandas as pd

from analyses.triad.src.diagnose_ori_flips import (
    detect_flip_events, correct_flip_events, correct_mounting_nearest_ori,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────

rootdir = "/Users/juliechen/Library/CloudStorage/Dropbox-Dropbox@RU/Julie Chen/Ruta lab rotation 2026/fb_20mm"
acq     = "20260512-1111_MFF-triad2_Dmel_CantonS_4do_gh"

avi_path = os.path.join(rootdir, 'raw_videos', acq, f'{acq}.avi')
trk_path = os.path.join(rootdir, 'processed_mats', f'{acq}.parquet')
save_dir = os.path.join(rootdir, 'figures', acq, 'flip_clips')

n_samples               = 15    # max number of flip events to extract
pad_sec                 = 2.0   # seconds of context before/after each flip
jump_threshold_rad      = 2.0
stability_threshold_rad = 0.3
random_seed             = 42
mount_action_col        = 'mounting attempt'   # col holding actor fly id (-1 when inactive); set None to skip

# fly id → BGR color (dodgerblue, tomato, limegreen, yellow)
id_colors = {
    0: (255, 144,  30),
    1: ( 71,  99, 255),
    2: ( 50, 205,  50),
    3: (  0, 215, 255),
}

ARROW_LEN = 20   # orientation arrow length in pixels

# ─────────────────────────────────────────────────────────────────────────────


def _build_frame_lookup(trk, needed_frames):
    """Build {frame_idx: (fly_positions, fly_orientations)} for a set of frames."""
    lookup = {}
    for frame_idx, frame_df in trk[trk['frame'].isin(needed_frames)].groupby('frame'):
        fly_positions, fly_orientations = {}, {}
        for _, row in frame_df.drop_duplicates('id').iterrows():
            fid = int(row['id'])
            if pd.isna(row['pos_x']) or pd.isna(row['pos_y']):
                continue
            fly_positions[fid] = (int(row['pos_x']), int(row['pos_y']))
            if not pd.isna(row['ori']):
                fly_orientations[fid] = float(row['ori'])
        lookup[frame_idx] = (fly_positions, fly_orientations)
    return lookup


def _mark_frame(frame, frame_idx, frame_lookup, id_colors,
                flip_fly_id=None, in_flip_window=False,
                border_color=None, window_label=None):
    cv2.putText(frame, f'frame {frame_idx}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if in_flip_window and border_color is not None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                      border_color, 6)
    if in_flip_window and window_label is not None:
        label_color = border_color if border_color is not None else (255, 255, 255)
        cv2.putText(frame, window_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3, cv2.LINE_AA)

    if frame_idx not in frame_lookup:
        return

    fly_positions, fly_orientations = frame_lookup[frame_idx]
    for fid, (px, py) in fly_positions.items():
        color = id_colors.get(fid, (255, 255, 255))
        if fid == flip_fly_id:
            cv2.circle(frame, (px, py), radius=10, color=color, thickness=2)
        else:
            cv2.circle(frame, (px, py), radius=6, color=color, thickness=-1)
        if fid in fly_orientations:
            ori = fly_orientations[fid]
            ex = int(px + ARROW_LEN * np.cos(ori))
            ey = int(py + ARROW_LEN * np.sin(ori))
            cv2.arrowedLine(frame, (px, py), (ex, ey),
                            color=color, thickness=2, tipLength=0.3)
        cv2.putText(frame, f'id={fid}', (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _write_clip(cap, out_path, fourcc, fps, frame_w, frame_h,
                clip_start, clip_stop, ev, frame_lookup, id_colors,
                border_color, window_label):
    flip_fly_id = int(ev['fly_id'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
    for frame_idx in range(clip_start, clip_stop + 1):
        ret, frame = cap.read()
        if not ret:
            break
        in_flip = int(ev['start_frame']) <= frame_idx < int(ev['end_frame'])
        _mark_frame(frame, frame_idx, frame_lookup, id_colors,
                    flip_fly_id=flip_fly_id, in_flip_window=in_flip,
                    border_color=border_color, window_label=window_label)
        out.write(frame)
    out.release()


def main():
    assert os.path.exists(avi_path), f"Video not found:\n  {avi_path}"
    assert os.path.exists(trk_path), f"Parquet not found:\n  {trk_path}"

    cap = cv2.VideoCapture(avi_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Video: {frame_w}x{frame_h}  |  {fps:.1f} fps  |  {n_frames} frames  "
          f"|  {n_frames/fps:.1f} s")

    trk = pd.read_parquet(trk_path)

    # step 1: fix gradual mounting flips using line-of-sight to nearest fly
    trk_corrected = (correct_mounting_nearest_ori(trk, action_col=mount_action_col)
                     if mount_action_col else trk.copy())

    # step 2: detect remaining sharp flip events (NaN gaps handled)
    events = detect_flip_events(trk_corrected, fps=int(fps),
                                jump_threshold_rad=jump_threshold_rad,
                                stability_threshold_rad=stability_threshold_rad)
    if len(events) == 0:
        print("No flip events detected — nothing to extract.")
        return

    # step 3: correct detected flips by adding π
    trk_corrected = correct_flip_events(trk_corrected, events)

    # sample up to n_samples, spread across flies
    n_flies = events['fly_id'].nunique()
    per_fly = max(1, n_samples // n_flies)
    sample = (events
              .groupby('fly_id', group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_fly), random_state=random_seed))
              .head(n_samples)
              .reset_index(drop=True))
    print(f"Sampling {len(sample)} of {len(events)} detected events "
          f"({n_flies} flies, up to {per_fly} each)")

    # collect all frames needed across all sampled events
    pad_frames = int(math.ceil(pad_sec * fps))
    needed_frames = set()
    for _, ev in sample.iterrows():
        s = max(0, int(ev['start_frame']) - pad_frames)
        e = min(n_frames - 1, int(ev['end_frame']) + pad_frames)
        needed_frames.update(range(s, e + 1))

    print("Building frame lookups...")
    lookup_orig = _build_frame_lookup(trk, needed_frames)
    lookup_corr = _build_frame_lookup(trk_corrected, needed_frames)
    print(f"  {len(lookup_orig)} frames indexed")

    os.makedirs(save_dir, exist_ok=True)
    cap    = cv2.VideoCapture(avi_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    for i, (_, ev) in enumerate(sample.iterrows()):
        clip_start  = max(0, int(ev['start_frame']) - pad_frames)
        clip_stop   = min(n_frames - 1, int(ev['end_frame']) + pad_frames)
        clip_len    = clip_stop - clip_start + 1
        stem = (f"flip_{i+1:02d}_fly{int(ev['fly_id'])}"
                f"_f{int(ev['start_frame'])}-{int(ev['end_frame'])}"
                f"_{ev['duration_sec']:.1f}s")
        print(f"  {stem}  ({clip_len / fps:.1f} s)")

        # original clip — red border during flip window
        _write_clip(cap, os.path.join(save_dir, f"{stem}_original.mp4"),
                    fourcc, fps, frame_w, frame_h,
                    clip_start, clip_stop, ev, lookup_orig, id_colors,
                    border_color=(0, 0, 200), window_label='FLIP')

        # corrected clip — green border over the same window
        _write_clip(cap, os.path.join(save_dir, f"{stem}_corrected.mp4"),
                    fourcc, fps, frame_w, frame_h,
                    clip_start, clip_stop, ev, lookup_corr, id_colors,
                    border_color=(0, 200, 0), window_label='CORRECTED')

    cap.release()
    print(f"\nDone. Clips saved to:\n  {save_dir}")


if __name__ == '__main__':
    main()