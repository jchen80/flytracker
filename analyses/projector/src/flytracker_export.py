"""Export a projector dot-assay to FlyTracker format so it can be loaded in the
FlyTracker visualizer (to manually annotate target-switch actions there instead of
the custom GUI).

Writes the standard FlyTracker layout next to the projector data:
    {assay_type}/flytracker/{assay}/
        {assay}.avi                       (symlink to the real movie)
        calibration.mat                   (calib struct: FPS, PPM, ROI, masks)
        {assay}/
            {assay}-track.mat             (trk struct: names/data/flags/flies_in_chamber)
            {assay}-feat.mat              (feat struct: names/units/data)

The 3 FlyTracker "flies" are the male (id 1) + inner dot (id 2) + outer dot (id 3),
matching the JAABA trx ordering. Positions/orientation/axes come from trx.mat;
wing keypoints + per-fly features come from the perframe mats. Fields FlyTracker
doesn't have for this assay (legs, areas, fg metrics) are left NaN.

a/b convention: trx stores quarter-axis lengths (FlyTracker trx.a = major/4), so
major axis len = 4*a, minor axis len = 4*b.

Already-exported assays (those with a {assay}-track.mat) are skipped by default so
re-running won't clobber a {assay}-actions.mat you've annotated alongside them; pass
--overwrite (CLI) or overwrite=True to force a re-export.
"""

import os
import numpy as np
import scipy.io as sio

from . import mat_loaders as ml
from . import calibration as cal

# FlyTracker's standard 35 track fields (order matters; data column index = position)
TRACK_NAMES = [
    'pos x', 'pos y', 'ori', 'major axis len', 'minor axis len', 'body area',
    'fg area', 'img contrast', 'min fg dist', 'wing l x', 'wing l y', 'wing r x',
    'wing r y', 'wing l ang', 'wing l len', 'wing r ang', 'wing r len',
    'leg 1 x', 'leg 1 y', 'leg 2 x', 'leg 2 y', 'leg 3 x', 'leg 3 y', 'leg 4 x',
    'leg 4 y', 'leg 5 x', 'leg 5 y', 'leg 6 x', 'leg 6 y', 'leg 1 ang', 'leg 2 ang',
    'leg 3 ang', 'leg 4 ang', 'leg 5 ang', 'leg 6 ang',
]
FEAT_NAMES = ['vel', 'ang_vel', 'min_wing_ang', 'max_wing_ang', 'mean_wing_length',
              'axis_ratio', 'fg_body_ratio', 'contrast', 'dist_to_wall']
FEAT_UNITS = ['mm/s', 'rad/s', 'rad', 'rad', 'mm', 'ratio', 'ratio', '', 'mm']

TARGET_ORDER = ['fly', 'innerdot', 'outerdot']      # -> FlyTracker ids 1, 2, 3


def _cell_col(strings):
    """List of str -> (len,1) object array (MATLAB cell column of char)."""
    arr = np.empty((len(strings), 1), dtype=object)
    for i, s in enumerate(strings):
        arr[i, 0] = s
    return arr


def _video_size(video_path):
    """(width, height) of the movie, or a default if cv2/video unavailable."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w and h:
            return w, h
    except Exception:
        pass
    return 0, 0


def _build_track(trx, perframe, ppm, fps):
    """Assemble (trk_dict, feat_dict) from the per-target trx + fly perframe data."""
    targets = [t for t in TARGET_ORDER if t in trx]
    nflies = len(targets)
    nframes = len(trx['fly'])
    col = {name: i for i, name in enumerate(TRACK_NAMES)}

    data = np.full((nflies, nframes, len(TRACK_NAMES)), np.nan)
    feat = np.full((nflies, nframes, len(FEAT_NAMES)), np.nan)
    fcol = {name: i for i, name in enumerate(FEAT_NAMES)}

    def diff_rate(arr):
        return np.concatenate([[0.0], np.diff(arr)])

    for k, t in enumerate(targets):
        d = trx[t]
        x, y, th = d['x'].to_numpy(), d['y'].to_numpy(), d['theta'].to_numpy()
        a, b = d['a'].to_numpy(), d['b'].to_numpy()
        major, minor = 4 * a, 4 * b
        data[k, :, col['pos x']] = x
        data[k, :, col['pos y']] = y
        data[k, :, col['ori']] = th
        data[k, :, col['major axis len']] = major
        data[k, :, col['minor axis len']] = minor

        # feats available for every target
        speed_px = np.hypot(diff_rate(x), diff_rate(y))          # px/frame
        feat[k, :, fcol['vel']] = speed_px * fps / ppm           # mm/s
        feat[k, :, fcol['ang_vel']] = diff_rate(np.unwrap(th)) * fps
        with np.errstate(divide='ignore', invalid='ignore'):
            feat[k, :, fcol['axis_ratio']] = np.where(minor > 0, major / minor, np.nan)

        if t == 'fly':                                           # wings only for the fly
            for side, wx, wy in (('l', 'xwingl', 'ywingl'), ('r', 'xwingr', 'ywingr')):
                if wx in perframe and wy in perframe:
                    wpx, wpy = perframe[wx], perframe[wy]
                    data[k, :, col[f'wing {side} x']] = wpx
                    data[k, :, col[f'wing {side} y']] = wpy
                    data[k, :, col[f'wing {side} len']] = np.hypot(wpx - x, wpy - y)
            if 'wing_anglel' in perframe:
                data[k, :, col['wing l ang']] = perframe['wing_anglel']
            if 'wing_angler' in perframe:
                data[k, :, col['wing r ang']] = perframe['wing_angler']
            if 'wing_angle_min' in perframe:
                feat[k, :, fcol['min_wing_ang']] = perframe['wing_angle_min']
            if 'wing_angle_max' in perframe:
                feat[k, :, fcol['max_wing_ang']] = perframe['wing_angle_max']
            ll = data[k, :, col['wing l len']]
            rl = data[k, :, col['wing r len']]
            feat[k, :, fcol['mean_wing_length']] = np.nanmean(
                np.vstack([ll, rl]), axis=0) / ppm

    trk = {
        'names': _cell_col(TRACK_NAMES),
        'data': data,
        'flags': np.zeros((0, 6)),                  # no identity-swap flags
        'flies_in_chamber': np.array([[np.arange(1, nflies + 1, dtype=float)]], dtype=object),
    }
    feat_struct = {'names': _cell_col(FEAT_NAMES), 'units': _cell_col(FEAT_UNITS), 'data': feat}
    return trk, feat_struct, nflies


def _build_calibration(ppm, fps, nflies, w, h):
    """Minimal FlyTracker calib struct: enough for the visualizer to open."""
    cx, cy = w / 2.0, h / 2.0
    mask = np.ones((h, w), dtype=np.uint8) if (w and h) else np.ones((1, 1), np.uint8)
    return {
        'auto_detect': 0, 'n_chambers': 1, 'n_rows': 1, 'n_cols': 1,
        'roi_type': 1,                              # rectangle (full frame)
        'FPS': float(fps), 'PPM': float(ppm),
        'centroids': np.array([cx, cy], dtype=float),
        'r': 0.0, 'w': float(w), 'h': float(h),
        'mask': mask, 'full_mask': mask, 'masks': mask,
        'rois': np.array([1, 1, w, h], dtype=float),
        'n_flies': int(nflies), 'magnet': 0, 'dead_female': 0, 'valid_chambers': 1,
    }


def _export_dirs(assay_type_dir, assay, outroot=None):
    """(out_dir, sub_dir, track_path) for an assay's FlyTracker export."""
    out_dir = outroot or os.path.join(assay_type_dir, 'flytracker', assay)
    sub_dir = os.path.join(out_dir, assay)            # FlyTracker's <vid>/<vid>/ nesting
    return out_dir, sub_dir, os.path.join(sub_dir, f'{assay}-track.mat')


def is_exported(assay_type_dir, assay, outroot=None):
    """True if this assay has already been exported (its track.mat exists)."""
    return os.path.exists(_export_dirs(assay_type_dir, assay, outroot)[2])


def export_assay(assay_type_dir, assay, ppm=None, fps=None, outroot=None, overwrite=False):
    """Write FlyTracker track/feat/calibration mats + a movie symlink for one assay.

    ppm defaults to the assay-type calibration (px_per_mm); fps to the trx fps.
    Returns the export dir. Raises if the assay isn't a 2-dot assay.

    If the assay is already exported (its {assay}-track.mat exists) it is skipped
    and the existing dir returned, unless overwrite=True. This preserves any
    {assay}-actions.mat annotated alongside it (the export never touches actions).
    """
    out_dir, sub_dir, track_path = _export_dirs(assay_type_dir, assay, outroot)
    if not overwrite and os.path.exists(track_path):
        print(f"[{assay}] already exported, skipping (use overwrite to redo)")
        return out_dir

    jaaba_dir = os.path.join(assay_type_dir, 'JAABA', assay)
    trx = ml.load_trx(os.path.join(jaaba_dir, 'trx.mat'))
    if 'innerdot' not in trx or 'outerdot' not in trx:
        raise ValueError(f"{assay}: need inner+outer dots for a 3-fly FlyTracker export")
    order = list(trx.keys())
    perframe = ml.load_perframe(os.path.join(jaaba_dir, 'perframe'),
                                target_index=order.index('fly'))
    if fps is None:
        fps = float(trx['fly']['fps'].iloc[0])
    if ppm is None:
        ppm = cal.load_calibration(assay_type_dir) if cal.has_calibration(assay_type_dir) else 1.0

    video = cal.assay_video(assay_type_dir, assay)
    w, h = _video_size(video) if video else (0, 0)

    trk, feat_struct, nflies = _build_track(trx, perframe, ppm, fps)
    calib = _build_calibration(ppm, fps, nflies, w, h)

    os.makedirs(sub_dir, exist_ok=True)

    sio.savemat(os.path.join(sub_dir, f'{assay}-track.mat'), {'trk': trk}, do_compression=True)
    sio.savemat(os.path.join(sub_dir, f'{assay}-feat.mat'), {'feat': feat_struct}, do_compression=True)
    sio.savemat(os.path.join(out_dir, 'calibration.mat'), {'calib': calib}, do_compression=True)

    if video:
        link = os.path.join(out_dir, f'{assay}.avi')
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(video), link)

    print(f"[{assay}] FlyTracker export ({nflies} flies, {len(trx['fly'])} frames, "
          f"PPM={ppm:.3f}, fps={fps:g}) -> {out_dir}")
    return out_dir


def main():
    import argparse
    from . import data_io as dio
    ap = argparse.ArgumentParser(
        description='Export projector assays to FlyTracker format for manual annotation.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('assay_type_dir')
    ap.add_argument('--assay', default=None, help='export only this assay (default: all 2-dot assays)')
    ap.add_argument('--ppm', type=float, default=None, help='pixels per mm (default: assay-type calibration)')
    ap.add_argument('--fps', type=float, default=None, help='override fps (default: from trx)')
    ap.add_argument('--overwrite', action='store_true',
                    help='re-export assays that were already exported (default: skip them)')
    args = ap.parse_args()

    assays = [args.assay] if args.assay else dio.list_assays(args.assay_type_dir)
    for a in assays:
        try:
            export_assay(args.assay_type_dir, a, ppm=args.ppm, fps=args.fps,
                         overwrite=args.overwrite)
        except ValueError as e:
            print(f"  [skip] {e}")


if __name__ == '__main__':
    main()
