"""Per-assay-type pixel->mm calibration via an interactive scale-bar GUI.

The projected arena/dot sizes are not in true mm, and the projector geometry is
fixed for a whole assay type, so calibration is done once per assay type: we open
one representative assay's video frame, the user clicks two points to draw a scale
bar over a feature of known length, then types that length in mm. The resulting
px_per_mm is stored once per assay type in
    {assay_type_dir}/processed/calibration.json

cv2 is imported lazily (only when the GUI actually runs), matching the pattern in
analyses/triad/src/flip/review_gui.py.
"""

import os
import json
import glob
import datetime

CALIB_NAME = 'calibration.json'


def _calib_path(assay_type_dir):
    return os.path.join(assay_type_dir, 'processed', CALIB_NAME)


def _read_calib(assay_type_dir):
    """Return the single per-type calibration dict, or {} if not calibrated.

    Tolerates the legacy per-assay format (a dict keyed by assay name): in that
    case the first entry is reused, since one scale applies to the whole type.
    """
    p = _calib_path(assay_type_dir)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        data = json.load(f)
    if 'px_per_mm' in data:                       # current per-type format
        return data
    for entry in data.values():                   # legacy per-assay format
        if isinstance(entry, dict) and 'px_per_mm' in entry:
            return entry
    return {}


def _write_calib(assay_type_dir, calib):
    p = _calib_path(assay_type_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(calib, f, indent=2)
    return p


def assay_video(assay_type_dir, assay):
    """Path to an assay's video: prefer the JAABA movie.avi symlink, else raw_videos."""
    cand = os.path.join(assay_type_dir, 'JAABA', assay, 'movie.avi')
    if os.path.exists(cand):
        return cand
    avis = glob.glob(os.path.join(assay_type_dir, 'raw_videos', assay, '*.avi'))
    return avis[0] if avis else None


def has_calibration(assay_type_dir):
    return 'px_per_mm' in _read_calib(assay_type_dir)


def load_calibration(assay_type_dir):
    """px_per_mm for the assay type; raises KeyError if it hasn't been calibrated."""
    calib = _read_calib(assay_type_dir)
    if 'px_per_mm' not in calib:
        raise KeyError(f"no calibration in {_calib_path(assay_type_dir)}; "
                       f"run calibrate_assay_type first")
    return float(calib['px_per_mm'])


def _grab_frame(video_path, frame_idx):
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = n // 2 if frame_idx is None else max(0, min(frame_idx, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx} from {video_path}")
    return img, idx


def calibrate_assay_type(assay_type_dir, assay, frame_idx=None, mm=None):
    """Open a representative assay's video, draw a scale bar, store px_per_mm.

    `assay` is just the assay whose video is shown for clicking; the resulting
    px_per_mm is saved once for the whole assay type. Controls: left-click the two
    endpoints of a feature of known length; press ENTER to accept (or 'r' to reset
    the points, ESC to cancel). The length in mm is taken from `mm` if given,
    otherwise typed at the terminal prompt. Returns px_per_mm and writes
    processed/calibration.json.
    """
    import cv2
    import numpy as np

    video = assay_video(assay_type_dir, assay)
    if not video:
        raise FileNotFoundError(f"no video found for assay '{assay}' under {assay_type_dir}")
    base, idx = _grab_frame(video, frame_idx)

    pts = []
    win = f'calibrate: {assay}  (click 2 pts, ENTER=accept, r=reset, ESC=cancel)'

    def redraw():
        img = base.copy()
        for p in pts:
            cv2.circle(img, p, 4, (0, 215, 255), -1)
        if len(pts) == 2:
            cv2.line(img, pts[0], pts[1], (0, 215, 255), 2)
            d = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
            cv2.putText(img, f'{d:.1f} px', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) >= 2:
                pts.clear()
            pts.append((int(x), int(y)))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    accepted = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(pts) == 2:      # ENTER
            accepted = True
            break
        if key == ord('r'):
            pts.clear()
            redraw()
        if key == 27:                              # ESC
            break
    cv2.destroyWindow(win)

    if not accepted:
        print(f"[{assay}] calibration cancelled")
        return None

    px = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
    if mm is None:
        mm = float(input(f"[{assay}] scale bar = {px:.1f} px. Enter its length in mm: "))
    px_per_mm = px / mm

    calib = {
        'px_per_mm': px_per_mm, 'mm': mm, 'px': px,
        'points': pts, 'frame_idx': idx,
        'calibrated_on': assay, 'video': os.path.basename(video),
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    p = _write_calib(assay_type_dir, calib)
    print(f"[{assay}] px_per_mm = {px_per_mm:.4f}  ({px:.1f} px = {mm} mm)  -> {p}")
    return px_per_mm
