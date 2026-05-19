"""
Extract paired video clips around detected orientation-flip events:
  *_original.mp4  — orientation arrows from raw tracking data
  *_corrected.mp4 — orientation arrows after flip correction (π-shift)

Loads raw FlyTracker data (track.mat / feat.mat) so the before/after
comparison is meaningful even after processed parquets have been corrected.

Edit the CONFIG section, then run:
    python analyses/triad/src/extract_flip_clips.py

During the flip window the original clip shows a red border + "FLIP" label;
the corrected clip shows a green border + "CORRECTED" label.
The flipping fly is drawn as an open ring; all others are filled dots.
"""

import cv2
import itertools
import math
import os
import numpy as np
import pandas as pd

import libs.utils as util
from analyses.triad.src import multi_funcs as mf
from analyses.triad.src.diagnose_ori_flips import (
    detect_flip_events, correct_flip_events, correct_mounting_nearest_ori,
    print_jump_frames,
)
from analyses.triad.src.util import find_copulation_frame

# ── CONFIG ───────────────────────────────────────────────────────────────────

rootdir = "/Volumes/Julie/fb_MMF_MFF_triad_38mm"
acq     = "20260501-1031_MMF-triad3_Dyak__3do_gh"

acq_dir  = os.path.join(rootdir, 'raw_videos', acq)
avi_path = os.path.join(acq_dir, f'{acq}.avi')
save_dir = os.path.join(rootdir, 'figures', acq, 'flip_clips')

n_samples               = 15    # max number of flip events to extract
pad_sec                 = 2.0   # seconds of context before/after each flip
jump_threshold_rad      = 2.0
stability_threshold_rad = 0.2
random_seed             = 42
mount_action_col        = 'mounting attempt'   # set None to skip mounting correction
force_frames            = [3270] # extract clips around these frames regardless of detected events
interactive             = True   # if True, show GUI to label each jump before pairing

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
    assert os.path.exists(acq_dir),  f"Acquisition dir not found:\n  {acq_dir}"

    cap = cv2.VideoCapture(avi_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Video: {frame_w}x{frame_h}  |  {fps:.1f} fps  |  {n_frames} frames  "
          f"|  {n_frames/fps:.1f} s")

    # load raw FlyTracker data (track.mat / feat.mat)
    calib, trk, feat = util.load_flytracker_data(acq_dir,
                            calib_is_upstream=False, filter_ori=True)

    # ── pairwise metrics: dist_to_other + abs_ang_between into trk ───────────
    all_fly_ids = sorted(trk['id'].unique())
    pairs = list(itertools.combinations(all_fly_ids, 2))

    pair_trk_list = []
    for flyid1, flyid2 in pairs:
        trk_ = trk[trk['id'].isin([flyid1, flyid2])].copy()
        feat_ = feat[feat['id'].isin([flyid1, flyid2])].copy()
        trk_, feat_ = mf.add_pairwise_metrics(trk_, feat_, calib,
                                               flyid1=flyid1, flyid2=flyid2)
        for fid, oid in [(flyid1, flyid2), (flyid2, flyid1)]:
            fmask = trk_['id'] == fid
            omask = trk_['id'] == oid
            trk_.loc[fmask, 'dist_to_other'] = feat_.loc[feat_['id'] == fid, 'dist_to_other'].values
            f_pos = trk_.loc[fmask, ['pos_x', 'pos_y']].values
            o_pos = trk_.loc[omask, ['pos_x', 'pos_y']].values
            vec = o_pos - f_pos
            trk_.loc[fmask, 'abs_ang_between'] = np.arctan2(vec[:, 1], vec[:, 0])
        trk_['pair'] = f"{flyid1}_{flyid2}"
        pair_trk_list.append(trk_)
    trk_combined = pd.concat(pair_trk_list, ignore_index=True)

    # flip ori sign (FT math/y-up → image/y-down) so arrows render correctly
    # on video and ori is consistent with abs_ang_between for mounting correction
    trk_combined['ori'] = -1 * trk_combined['ori']

    # ── action annotations (needed for mounting correction) ───────────────────
    trk_combined = util.load_and_assign_ft_actions(trk_combined, acq_dir, acq)

    # ── trim at copulation before flip correction ─────────────────────────────
    cop_frame = find_copulation_frame(trk_combined)
    if cop_frame is not None:
        print(f"First copulation frame: {cop_frame} — trimming before flip correction.")
        trk_combined = trk_combined[trk_combined['frame'] <= cop_frame].copy()

    # ── orientation flip correction ───────────────────────────────────────────
    # keep a copy of the raw (pre-correction) tracking for the original clips
    trk_orig = trk_combined.copy()

    if mount_action_col and mount_action_col in trk_combined.columns:
        trk_combined = correct_mounting_nearest_ori(trk_combined,
                                                    action_col=mount_action_col)
    print("\n── Raw jumps above threshold (before pairing/stability filter) ──")
    print_jump_frames(trk_combined, jump_threshold_rad=jump_threshold_rad)
    print()
    events = detect_flip_events(trk_combined, fps=int(fps),
                                jump_threshold_rad=jump_threshold_rad,
                                stability_threshold_rad=stability_threshold_rad,
                                interactive=interactive,
                                video_path=avi_path,
                                trk_for_display=trk_orig,
                                id_colors=id_colors,
                                arrow_len=ARROW_LEN,
                                pad_sec=pad_sec)
    if len(events) == 0 and not force_frames:
        print("No flip events detected — nothing to extract.")
        return
    if len(events) > 0:
        print(events[['fly_id', 'start_frame', 'end_frame', 'duration_sec',
                       'jump_in_rad', 'jump_out_rad', 'segment_ori_std']].to_string())
        trk_combined = correct_flip_events(trk_combined, events)

    # ── sample events ─────────────────────────────────────────────────────────
    if len(events) > 0:
        n_flies = events['fly_id'].nunique()
        per_fly = max(1, n_samples // n_flies)
        sample = (events
                  .groupby('fly_id', group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), per_fly), random_state=random_seed))
                  .head(n_samples)
                  .reset_index(drop=True))
        print(f"Sampling {len(sample)} of {len(events)} detected events "
              f"({n_flies} flies, up to {per_fly} each)")
    else:
        sample = pd.DataFrame(columns=['fly_id', 'start_frame', 'end_frame', 'duration_sec'])
        print("No detected events — extracting manual frames only.")

    # ── build frame lookups ───────────────────────────────────────────────────
    pad_frames = int(math.ceil(pad_sec * fps))
    needed_frames = set()
    for _, ev in sample.iterrows():
        s = max(0, int(ev['start_frame']) - pad_frames)
        e = min(n_frames - 1, int(ev['end_frame']) + pad_frames)
        needed_frames.update(range(s, e + 1))
    for ff in (force_frames or []):
        needed_frames.update(range(max(0, ff - pad_frames),
                                   min(n_frames - 1, ff + pad_frames) + 1))

    print("Building frame lookups...")
    lookup_orig = _build_frame_lookup(trk_orig,     needed_frames)
    lookup_corr = _build_frame_lookup(trk_combined, needed_frames)
    print(f"  {len(lookup_orig)} frames indexed")

    # ── write clips ───────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    cap    = cv2.VideoCapture(avi_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    for i, (_, ev) in enumerate(sample.iterrows()):
        clip_start = max(0, int(ev['start_frame']) - pad_frames)
        clip_stop  = min(n_frames - 1, int(ev['end_frame']) + pad_frames)
        clip_len   = clip_stop - clip_start + 1
        stem = (f"flip_{i+1:02d}_fly{int(ev['fly_id'])}"
                f"_f{int(ev['start_frame'])}-{int(ev['end_frame'])}"
                f"_{ev['duration_sec']:.1f}s")
        print(f"  {stem}  ({clip_len / fps:.1f} s)")

        _write_clip(cap, os.path.join(save_dir, f"{stem}_original.mp4"),
                    fourcc, fps, frame_w, frame_h,
                    clip_start, clip_stop, ev, lookup_orig, id_colors,
                    border_color=(0, 0, 200), window_label='FLIP')

        _write_clip(cap, os.path.join(save_dir, f"{stem}_corrected.mp4"),
                    fourcc, fps, frame_w, frame_h,
                    clip_start, clip_stop, ev, lookup_corr, id_colors,
                    border_color=(0, 200, 0), window_label='CORRECTED')

    for ff in (force_frames or []):
        clip_start = max(0, ff - pad_frames)
        clip_stop  = min(n_frames - 1, ff + pad_frames)
        clip_len   = clip_stop - clip_start + 1
        stem = f"manual_f{ff}"
        print(f"  {stem}  ({clip_len / fps:.1f} s)")

        # if a detected event covers this frame, use its range for the border
        # so the highlighted window matches what was actually corrected
        covering = (events[(events['start_frame'] <= ff) & (events['end_frame'] > ff)]
                    if len(events) > 0 else pd.DataFrame())
        if len(covering) > 0:
            ev_row = covering.iloc[0]
            manual_ev = pd.Series({'fly_id':      ev_row['fly_id'],
                                   'start_frame': ev_row['start_frame'],
                                   'end_frame':   ev_row['end_frame'],
                                   'duration_sec': ev_row['duration_sec']})
            print(f"    → covered by detected event: fly {int(ev_row['fly_id'])} "
                  f"frames {int(ev_row['start_frame'])}–{int(ev_row['end_frame'])} "
                  f"({ev_row['duration_sec']:.2f}s)")
        else:
            manual_ev = pd.Series({'fly_id': -1, 'start_frame': ff, 'end_frame': ff + 1,
                                   'duration_sec': 1 / fps})
            print(f"    → frame {ff} not within any detected flip event")

        _write_clip(cap, os.path.join(save_dir, f"{stem}_original.mp4"),
                    fourcc, fps, frame_w, frame_h,
                    clip_start, clip_stop, manual_ev, lookup_orig, id_colors,
                    border_color=(200, 0, 200), window_label='MANUAL')
        _write_clip(cap, os.path.join(save_dir, f"{stem}_corrected.mp4"),
                    fourcc, fps, frame_w, frame_h,
                    clip_start, clip_stop, manual_ev, lookup_corr, id_colors,
                    border_color=(200, 0, 200), window_label='MANUAL')

    cap.release()
    print(f"\nDone. Clips saved to:\n  {save_dir}")


if __name__ == '__main__':
    main()
