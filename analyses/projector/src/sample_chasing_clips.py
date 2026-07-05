"""CLI: sample video clips of chasing-either-dot, by species × LED intensity.

For each (species, LED intensity) it first bridges the chasing-either-dot mask
(chasing_innerdot OR chasing_outerdot) -- gaps shorter than `bridge_sec` are linked
so a brief orientation between chasing segments stays one bout -- and then samples
from the resulting *continuous* bouts. Only bouts at least `clip_sec` long and at
least `min_chasing_frac` chasing (after bridging) qualify; a random `clip_sec`
window is taken from inside each sampled bout and written as a short annotated .mp4
(male + both dots overlaid, with a per-frame chasing readout). Bouts never cross an
LED-block boundary or a frame gap, so each clip is at a single intensity.

Output: {assay_type_dir}/figures/chasing_clips/{species}/{intensity}pct/*.mp4
Filenames carry the assay, frame range, and the clip's chasing fraction.

Usage:
    python -m analyses.projector.src.sample_chasing_clips <assay_type_dir>
    python -m analyses.projector.src.sample_chasing_clips <dir> --clip-sec 4 --n 5 --min-frac 0.5
    python -m analyses.projector.src.sample_chasing_clips <dir> --bridge-sec 0.5
    python -m analyses.projector.src.sample_chasing_clips <dir> --no-mark   # raw video, no overlay

Run with the flytracker env.
"""

import os
import argparse

import numpy as np

from . import data_io as dio
from . import calibration as cal
from . import led_metadata as lm
from . import analyze_outerdot as ao

# BGR overlay colors: male = white, dots match plotting.TRACK_COLORS (orange / green)
_BGR = {'fly': (255, 255, 255), 'innerdot': (12, 131, 232), 'outerdot': (44, 160, 44)}
_CHASE_COLS = ('chasing_innerdot', 'chasing_outerdot')


def _chasing_any(df):
    """Per-frame boolean: chasing either dot (OR of whichever chasing columns exist)."""
    cols = [c for c in _CHASE_COLS if c in df.columns]
    if not cols:
        raise KeyError("no chasing_innerdot/chasing_outerdot columns; rebuild the cache "
                       "with the JAABA scores")
    return (df[cols].astype(float).fillna(0) > 0).any(axis=1).to_numpy()


def _write_clip(video, path, gdf, start, end, fps, mark=True):
    """Write frames [start, end] of `video` to `path`, overlaying male+dots from gdf."""
    import cv2
    cap = cv2.VideoCapture(video)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), fps, (fw, fh))
    pos = gdf.set_index('frame')
    arrow = 22
    for fi in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if mark and fi in pos.index:
            r = pos.loc[fi]
            if getattr(r, 'ndim', 1) > 1:        # paranoia: one row per frame expected
                r = r.iloc[0]
            for t in ('innerdot', 'outerdot'):   # dots first so the male draws on top
                x, y = r.get(f'{t}_x'), r.get(f'{t}_y')
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(x), int(y)), 7, _BGR[t], 2)
            fx, fy, th = r.get('fly_x'), r.get('fly_y'), r.get('fly_theta')
            if np.isfinite(fx) and np.isfinite(fy):
                cv2.circle(frame, (int(fx), int(fy)), 6, _BGR['fly'], -1)
                if np.isfinite(th):              # image coords (y-down), matches the video
                    cv2.arrowedLine(frame, (int(fx), int(fy)),
                                    (int(fx + arrow * np.cos(th)), int(fy + arrow * np.sin(th))),
                                    _BGR['fly'], 2, tipLength=0.3)
            inner = float(r.get('chasing_innerdot', 0) or 0) > 0
            outer = float(r.get('chasing_outerdot', 0) or 0) > 0
            dot = 'inner' if inner else ('outer' if outer else '-')
            col = (50, 220, 50) if (inner or outer) else (180, 180, 180)
            cv2.putText(frame, f'fr {fi}  chasing: {dot}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        out.write(frame)
    cap.release()
    out.release()


def sample_chasing_clips(df, assay_type_dir, out_dir=None, clip_sec=3.0,
                         min_chasing_frac=0.5, bridge_sec=1.0,
                         n_clips_per_group=3, seed=0, mark=True):
    """Sample chasing-either-dot clips per (species, LED intensity). See module docstring."""
    df = df[lm.valid_led(df)].copy()
    fps = float(df['fps'].iloc[0])
    clip_len = int(round(clip_sec * fps))
    out_dir = out_dir or os.path.join(assay_type_dir, 'figures', 'chasing_clips')
    rng = np.random.default_rng(seed)

    # Bridge sub-`bridge_sec` gaps, then keep only continuous bouts long enough for
    # a clip and chasing enough (after bridging) to be worth sampling.
    bouts = ao.chasing_bouts(df, bridge_sec=bridge_sec, members=_CHASE_COLS)
    if bouts.empty:
        print("No chasing bouts found.")
        return []
    bouts = bouts[(bouts['n_frames'] >= clip_len)
                  & (bouts['frac_chasing'] >= min_chasing_frac)]

    all_paths = []
    for sp in sorted(df['species'].dropna().unique(), reverse=True):  # reverse so the "wildtype" species is last and easier to find-
        sp_bouts = bouts[bouts['species'] == sp]
        for inten in sorted(sp_bouts['led_intensity'].dropna().unique()):
            cands = sp_bouts[sp_bouts['led_intensity'] == inten]
            if cands.empty:
                print(f"  {sp} {inten:g}%: no continuous bout >= {clip_sec:g}s with "
                      f">= {min_chasing_frac:.0%} chasing (bridge {bridge_sec:g}s)")
                continue
            pick = rng.choice(len(cands), size=min(n_clips_per_group, len(cands)), replace=False)
            sel = cands.iloc[sorted(pick)]
            gdir = os.path.join(out_dir, sp, f'{inten:g}pct')
            os.makedirs(gdir, exist_ok=True)
            print(f"  {sp} {inten:g}%: {len(cands)} candidate bouts, writing {len(sel)}")
            for _, bout in sel.iterrows():
                assay = bout['assay']
                # random clip_len window inside the (continuous) bout
                off = int(rng.integers(0, int(bout['n_frames']) - clip_len + 1))
                st = int(bout['start_frame']) + off
                en = st + clip_len - 1
                video = cal.assay_video(assay_type_dir, assay)
                if not video:
                    print(f"    [skip] no video for {assay}")
                    continue
                gdf = df[df['assay'] == assay]
                wdf = gdf[(gdf['frame'] >= st) & (gdf['frame'] <= en)]
                frac = float(_chasing_any(wdf).mean())   # this clip's actual chasing fraction
                fname = f'{sp}_{inten:g}pct_{assay}_fr{st:06d}-{en:06d}_chase{frac*100:.0f}.mp4'
                path = os.path.join(gdir, fname)
                _write_clip(video, path, gdf, st, en, fps, mark=mark)
                print(f"    {fname}")
                all_paths.append(path)
    print(f"\nTotal: {len(all_paths)} clips -> {out_dir}")
    return all_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('assay_type_dir', help='a single {assay_type} folder')
    ap.add_argument('--clip-sec', type=float, default=3.0, help='clip length in seconds (default: 3)')
    ap.add_argument('--min-frac', type=float, default=0.5,
                    help='min chasing fraction of a (bridged) bout to qualify (default: 0.5)')
    ap.add_argument('--bridge-sec', type=float, default=1.0,
                    help='link chasing bouts separated by gaps shorter than this, in '
                         'seconds, before sampling; 0 disables bridging (default: 1.0)')
    ap.add_argument('--n', type=int, default=3, dest='n_clips',
                    help='clips per species×intensity (default: 3)')
    ap.add_argument('--seed', type=int, default=0, help='sampling seed (default: 0)')
    ap.add_argument('--no-mark', action='store_true', help='do not overlay male/dot markers')
    args = ap.parse_args()

    df = dio.load_all_assays(args.assay_type_dir)
    if df is None or df.empty:
        print('No data loaded; build the parquet cache first with build_assays.py')
        return
    sample_chasing_clips(df, args.assay_type_dir, clip_sec=args.clip_sec,
                         min_chasing_frac=args.min_frac, bridge_sec=args.bridge_sec,
                         n_clips_per_group=args.n_clips,
                         seed=args.seed, mark=not args.no_mark)


if __name__ == '__main__':
    main()