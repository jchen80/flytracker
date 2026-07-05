"""Read the per-assay ProjectorExperiment run JSON: fps, LED block schedule, and
the per-camera-frame LED intensity.

The run json lives at raw_videos/{assay}/{assay}_run_*.json. Its `led_sequence`
holds an `events` list; each event has a projector `frame`, a `time_sec`, a
`command` (e.g. 'c1' camera trigger, 's021' set-intensity) and a human `label`
(e.g. 'Block 2: 2% intensity'). The camera does not start recording until the
camera-trigger event, so camera-frame f maps to absolute time
    t(f) = camera_trigger_sec + f / fps
which is what aligns the trx/scores frames to the LED blocks.

Ported from the preprocessing helper jaaba_dots/io_utils.py (not in this repo) so
the module is self-contained.
"""

import os
import re
import glob
import json

DEFAULT_FPS = 60
# The LEDs cannot physically exceed this; higher block intensities (e.g. a stray
# 100% block) are acquisition errors and their frames are ignored everywhere.
MAX_LED_INTENSITY = 99
# Per-species intensity ceiling: Dmel frames above 10% are excluded from all
# analyses. Species not listed here fall back to MAX_LED_INTENSITY (no extra cap).
SPECIES_MAX_LED_INTENSITY = {'Dmel': 10}
_BLOCK_RE = re.compile(r'block\s+\d+:\s*([\d.]+)\s*%', re.IGNORECASE)


def valid_led(df):
    """Boolean mask of frames with a usable LED intensity.

    A frame is valid when its intensity is assigned and at or below the ceiling for
    its species: SPECIES_MAX_LED_INTENSITY[species] when listed (e.g. Dmel <= 10),
    otherwise MAX_LED_INTENSITY. Frames with no/NaN species use MAX_LED_INTENSITY.
    """
    cap = MAX_LED_INTENSITY
    if 'species' in df.columns:
        cap = df['species'].map(SPECIES_MAX_LED_INTENSITY).fillna(MAX_LED_INTENSITY)
    return df['led_intensity'].notna() & (df['led_intensity'] <= cap)


def find_run_json(assay_type_dir, assay):
    """Path to an assay's run json, or None."""
    base = os.path.join(assay_type_dir, 'raw_videos', assay)
    js = glob.glob(os.path.join(base, '*_run_*.json')) or glob.glob(os.path.join(base, '*.json'))
    return js[0] if js else None


def parse_run_json(json_path):
    """Load the run json. Returns the parsed dict (raises on bad/missing file)."""
    with open(json_path) as f:
        return json.load(f)


def species_hint(json_path):
    """Species token embedded in the json filename ('..._2dot_Dmel.json'), or None."""
    m = re.search(r'_((?:D)[a-z]{3})\.json$', os.path.basename(json_path), re.IGNORECASE)
    return m.group(1).title() if m else None


def trajectory_name(run_json):
    """Basename of the trajectory CSV referenced in the run json (handles Windows paths)."""
    p = run_json.get('files', {}).get('trajectory_csv')
    if not p:
        return None
    return re.split(r'[\\/]', p)[-1]


_SPEEDS_RE = re.compile(r'_([\d.]+(?:-[\d.]+)+|\d+(?:\.\d+)?)mmps', re.IGNORECASE)
_SEGS_RE = re.compile(r'_(\d+)x([\d.]+)s', re.IGNORECASE)


def parse_trajectory_speeds(traj_name):
    """Parse a stimulus-speed schedule from a trajectory filename, or None.

    Expects the convention '..._<s1>-<s2>-...mmps_<N>x<D>s_...', e.g.
    'circle_1r_10-20-30-40-50mmps_5x20s_16.5mm_ccw_6blocks_prj5ms' -> speeds
    [10,20,30,40,50] held <D>=20 s each (one segment per speed). Returns
    {'speeds': [...], 'seg_dur_sec': D, 'cycle_sec': N*D} only when the number of
    speeds matches N (so the mapping is unambiguous); otherwise None.
    """
    if not traj_name:
        return None
    ms, mg = _SPEEDS_RE.search(traj_name), _SEGS_RE.search(traj_name)
    if not ms or not mg:
        return None
    speeds = [float(x) for x in ms.group(1).split('-')]
    n, dur = int(mg.group(1)), float(mg.group(2))
    if len(speeds) != n:
        return None
    return {'speeds': speeds, 'seg_dur_sec': dur, 'cycle_sec': n * dur}


def frame_speed(run_json, nframes, fps):
    """Per-frame stimulus speed (mm/s) from the trajectory filename, or None.

    The speed schedule is assumed to play continuously from the start of the
    recording and repeat every cycle (N*D s). Camera frame f maps to trajectory
    time camera_trigger_sec + f/fps; the segment is that time modulo the cycle.
    Returns a length-nframes list, or None if the filename isn't parseable.
    """
    sched = parse_trajectory_speeds(trajectory_name(run_json))
    if sched is None:
        return None
    t0 = camera_trigger_sec(run_json.get('led_sequence', {}))
    speeds, dur, cycle = sched['speeds'], sched['seg_dur_sec'], sched['cycle_sec']
    out = []
    for f in range(nframes):
        seg = int(((t0 + f / fps) % cycle) / dur)
        out.append(speeds[min(seg, len(speeds) - 1)])
    return out


# --- TEMP / HEURISTIC ---------------------------------------------------------
# Relative rotation direction of the two dots inferred from a '2x45' trajectory
# filename. Convention (calibration set only): within each 90 s cycle the first
# 45 s have both dots rotating ccw ('same'), the second 45 s have the outer ccw
# and the inner cw ('opposite'). This is a stopgap -- direction is NOT generally
# recoverable from the filename, so only the '2x45' case is encoded. Revisit when
# the trajectory metadata records direction explicitly.
_TWO_BY_45_RE = re.compile(r'2x45s?', re.IGNORECASE)
_DIR_LABELS_2X45 = ['same', 'opposite']   # seg 0: both ccw; seg 1: outer ccw, inner cw


def frame_target_direction(run_json, nframes, fps):
    """Per-frame relative dot direction ('same'/'opposite') for a '2x45' trajectory.

    Returns a length-nframes list, or None when the trajectory filename is not a
    recognized '2x45' schedule (see the TEMP note above). Frame f maps to
    trajectory time camera_trigger_sec + f/fps; the 90 s cycle splits into the
    first 45 s ('same') and the second 45 s ('opposite').
    """
    name = trajectory_name(run_json)
    if not name or not _TWO_BY_45_RE.search(name):
        return None
    t0 = camera_trigger_sec(run_json.get('led_sequence', {}))
    dur, cycle = 45.0, 90.0
    return [_DIR_LABELS_2X45[int(((t0 + f / fps) % cycle) / dur)] for f in range(nframes)]


def camera_trigger_sec(led_seq):
    """Time (s) the camera trigger fires; recording starts here, so it offsets frames."""
    for e in led_seq.get('events', []):
        if e.get('command') == 'c1' or 'trigger on' in str(e.get('label', '')).lower():
            return float(e.get('time_sec', 0.0))
    cf, pim = led_seq.get('camtrigger_frame'), led_seq.get('prj_interval_ms')
    if cf is not None and pim is not None:
        return cf * pim / 1000.0
    return 0.0


def assay_fps(run_json, nframes, default=DEFAULT_FPS):
    """fps = nframes / (total_dur_sec - camera_trigger_sec), from the run json."""
    ls = run_json.get('led_sequence', {})
    dur = ls.get('total_dur_sec')
    if dur:
        denom = dur - camera_trigger_sec(ls)
        if denom > 0:
            return int(round(nframes / denom))
    return default


def led_blocks(run_json):
    """Ordered [(start_time_sec, intensity_pct)] for each real LED block.

    A new block is counted ONLY when the intensity changes from the previous one.
    The led_sequence often contains 'backup' set-intensity events that repeat the
    current intensity (and camera/non-intensity events); these do not start a new
    block, so a 6-step ramp with backups yields 6 blocks, not 13. Intensity is read
    from the 'Block N: X% intensity' label; events without it are ignored.
    """
    events = []
    for e in run_json.get('led_sequence', {}).get('events', []):
        m = _BLOCK_RE.search(str(e.get('label', '')))
        if m:
            events.append((float(e.get('time_sec', 0.0)), float(m.group(1))))
    events.sort(key=lambda b: b[0])

    blocks = []
    for t, inten in events:
        if not blocks or blocks[-1][1] != inten:   # only on an intensity change
            blocks.append((t, inten))
    return blocks


def frame_intensity(run_json, nframes, fps):
    """Per-camera-frame LED intensity and block index.

    Returns (intensity, block_idx), each a length-nframes list. Camera frame f ->
    absolute time camera_trigger_sec + f/fps -> the most recent block whose onset
    is <= that time. Frames before the first block get intensity NaN / block -1.
    """
    ls = run_json.get('led_sequence', {})
    t0 = camera_trigger_sec(ls)
    blocks = led_blocks(run_json)
    starts = [b[0] for b in blocks]
    intens = [b[1] for b in blocks]

    intensity = [float('nan')] * nframes
    block_idx = [-1] * nframes
    for f in range(nframes):
        t = t0 + f / fps
        bi = -1
        for j, s in enumerate(starts):
            if t >= s:
                bi = j
            else:
                break
        if bi >= 0 and intens[bi] <= MAX_LED_INTENSITY:   # ignore impossible (>99%) blocks
            intensity[f] = intens[bi]
            block_idx[f] = bi
    return intensity, block_idx
