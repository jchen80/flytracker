"""Build and load the tidy per-assay parquet cache for the projector dot-assay.

Mirrors analyses/triad/src/data_io.py: a build step bakes the JAABA mats + run
json + calibration into one frame-indexed parquet per assay (the fly's row per
camera frame, with dot positions, relational features, JAABA behavior scores, LED
intensity, and assay metadata as columns), then load_all_* concatenates them for
cross-assay analysis.

Parquet cache lives at {assay_type_dir}/processed/{assay}.parquet.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import mat_loaders as ml
from . import led_metadata as lm

_SPECIES_RE = re.compile(r'(D[a-z]{3})(\d+)?', re.IGNORECASE)
_AGE_RE = re.compile(r'(\d+)do', re.IGNORECASE)
_ATR_RE = re.compile(r'(\d+)dATR', re.IGNORECASE)


def parse_assay_metadata(assay, json_path=None):
    """Parse an assay name like '20260611-1742_Dmel3_3do_1dATR_sh' into metadata.

    Returns dict with: species (Dmel/Dyak), fly_num, age_days, atr_days.
    A species hint from the run json filename, if given, takes precedence when
    the name itself is ambiguous.
    """
    meta = {'species': None, 'fly_num': None, 'age_days': None, 'atr_days': None}
    m = _SPECIES_RE.search(assay)
    if m:
        meta['species'] = m.group(1).title()
        meta['fly_num'] = int(m.group(2)) if m.group(2) else None
    if json_path:
        hint = lm.species_hint(json_path)
        if hint and not meta['species']:
            meta['species'] = hint
    a = _AGE_RE.search(assay)
    if a:
        meta['age_days'] = int(a.group(1))
    t = _ATR_RE.search(assay)
    if t:
        meta['atr_days'] = int(t.group(1))
    return meta


def _target_ang_vel_fov(fly_x, fly_y, dot_x, dot_y, angle_to, fps):
    """Retinal bearing rate (rad/s) of a dot from the DOT's own motion only.

    theta_dot = (dx*vy - dy*vx)/(dx^2+dy^2) with d = dot-fly and v = d(dot)/dt
    (the dot's lab-frame velocity). This is the exact derivative of the
    line-of-sight angle with the fly's velocity set to zero, so it isolates the
    angular visual motion of the dot on the fly's retina independent of the fly's
    own translation and rotation (the triad target_ang_vel_fov metric;
    transform_data/relative_metrics.calculate_theta_error). Returns (fov, signed)
    where signed = sign(angle_to)*fov is +progressive / -regressive (invariant to
    the image-y-down vs y-up handedness, so it matches triad's sign convention).
    """
    dt = 1.0 / fps
    dx = dot_x - fly_x
    dy = dot_y - fly_y
    vx = np.concatenate([[0.0], np.diff(dot_x)]) / dt
    vy = np.concatenate([[0.0], np.diff(dot_y)]) / dt
    r2 = dx ** 2 + dy ** 2
    fov = np.divide(dx * vy - dy * vx, r2, out=np.zeros_like(r2, dtype=float), where=r2 > 0)
    return fov, np.sign(angle_to) * fov


def processed_dir(assay_type_dir):
    return os.path.join(assay_type_dir, 'processed')


def list_assays(assay_type_dir):
    """JAABA assay folders (those that contain a trx.mat)."""
    jaaba = os.path.join(assay_type_dir, 'JAABA')
    if not os.path.isdir(jaaba):
        return []
    return sorted(a for a in os.listdir(jaaba)
                  if os.path.exists(os.path.join(jaaba, a, 'trx.mat')))


def build_assay_df(assay_type_dir, assay, px_per_mm, add_stim_speed=False,
                   add_target_direction=False, add_switches=False):
    """Build the tidy frame-indexed fly DataFrame for one assay.

    Combines: fly trx kinematics + dot positions (px and mm), fly perframe
    relational features, JAABA behavior scores (binary + continuous), per-frame
    LED intensity, and baked assay metadata.

    add_stim_speed -- if True, also parse the per-frame stimulus speed (mm/s) from
    the trajectory filename in the run json (added as `stim_speed`; left NaN when
    the filename can't be parsed). Off by default since not all trajectory names
    encode the speed schedule.

    add_target_direction -- if True, add `target_direction` ('same'/'opposite'):
    a TEMP/HEURISTIC inference of the two dots' relative rotation direction from a
    '2x45' trajectory filename (None when not a recognized 2x45 schedule). See
    led_metadata.frame_target_direction -- this is a stopgap, not generally valid.

    add_switches -- if True, merge switch annotations read from the FlyTracker
    `-actions.mat` (flytracker_actions) as `switching` (0/1) + `switch_target`
    ('inner'/'outer'); all 0/None when no actions.mat exists for the assay.
    """
    jaaba_dir = os.path.join(assay_type_dir, 'JAABA', assay)
    trx = ml.load_trx(os.path.join(jaaba_dir, 'trx.mat'))
    order = list(trx.keys())                 # trx/perframe target order
    fly_idx = order.index('fly')
    fly = trx['fly']
    n = len(fly)
    fps = float(fly['fps'].iloc[0])

    df = pd.DataFrame({'frame': np.arange(n), 'time_sec': fly['timestamps'].to_numpy()})

    # fly kinematics: px + recomputed mm (trx _mm fields were written with scale=1)
    for c in ['x', 'y', 'theta', 'a', 'b']:
        df[f'fly_{c}'] = fly[c].to_numpy()
    for c in ['x', 'y', 'a', 'b']:
        df[f'fly_{c}_mm'] = fly[c].to_numpy() / px_per_mm

    # dot positions (px + mm)
    for dot in ('innerdot', 'outerdot'):
        if dot in trx:
            for c in ['x', 'y']:
                df[f'{dot}_{c}'] = trx[dot][c].to_numpy()
                df[f'{dot}_{c}_mm'] = trx[dot][c].to_numpy() / px_per_mm

    # fly perframe relational features; _mm columns (written at scale=1 == px) re-scaled
    for feat, arr in ml.load_perframe(os.path.join(jaaba_dir, 'perframe'),
                                      target_index=fly_idx).items():
        arr = np.asarray(arr, dtype=float)
        df[feat] = arr / px_per_mm if feat.endswith('_mm') else arr

    # retinal FOV bearing-rate of each dot from the dot's OWN motion (fly held
    # fixed): the angular visual motion on the fly's retina, independent of the
    # fly's own translation/rotation. `_signed` > 0 = progressive, < 0 = regressive.
    for dot in ('innerdot', 'outerdot'):
        if f'{dot}_x' in df.columns and f'angle_to_{dot}' in df.columns:
            fov, signed = _target_ang_vel_fov(
                df['fly_x'].to_numpy(), df['fly_y'].to_numpy(),
                df[f'{dot}_x'].to_numpy(), df[f'{dot}_y'].to_numpy(),
                df[f'angle_to_{dot}'].to_numpy(), fps)
            df[f'target_ang_vel_fov_{dot}'] = fov
            df[f'target_ang_vel_fov_{dot}_signed'] = signed
            df[f'target_ang_vel_fov_{dot}_deg'] = np.degrees(fov)
            df[f'target_ang_vel_fov_{dot}_signed_deg'] = np.degrees(signed)

    # JAABA behavior scores for the fly target
    for sm in ml.find_score_mats(jaaba_dir):
        behavior, binary, cont = ml.load_scores(sm, target_index=fly_idx)
        df[behavior] = binary
        df[f'{behavior}_score'] = cont

    # per-frame LED intensity (and optionally stimulus speed) from the run json
    json_path = lm.find_run_json(assay_type_dir, assay)
    if json_path:
        run = lm.parse_run_json(json_path)
        intensity, block = lm.frame_intensity(run, n, fps)
        df['led_intensity'] = intensity
        df['led_block'] = block

        # heads-up (but proceed) when the LED schedule runs past the video, e.g. a
        # recording cropped at the end: trailing block onsets never reach a frame.
        # Alignment is anchored at the camera trigger, so the captured frames are
        # still labeled correctly -- only the un-captured tail blocks are dropped.
        blocks = lm.led_blocks(run)
        if blocks:
            t0 = lm.camera_trigger_sec(run.get('led_sequence', {}))
            last_frame_t = t0 + (n - 1) / fps
            missed = [b for b in blocks if b[0] > last_frame_t]
            if missed:
                print(f"  [warn] {assay}: LED schedule extends ~{missed[0][0] - last_frame_t:.1f}s "
                      f"past the video ({len(missed)} block(s) at "
                      f"{[f'{b[1]:g}%' for b in missed]} never captured); "
                      f"proceeding with the {n} frames present")
        if add_stim_speed:
            speed = lm.frame_speed(run, n, fps)
            if speed is None:
                df['stim_speed'] = np.nan
                print(f"  [warn] {assay}: trajectory '{lm.trajectory_name(run)}' "
                      f"has no parseable speed schedule; stim_speed left NaN")
            else:
                df['stim_speed'] = speed
        # TEMP: same/opposite dot direction inferred from a '2x45' trajectory name
        if add_target_direction:
            direction = lm.frame_target_direction(run, n, fps)
            if direction is None:
                df['target_direction'] = None
                print(f"  [warn] {assay}: trajectory '{lm.trajectory_name(run)}' "
                      f"is not a recognized 2x45 schedule; target_direction left None")
            else:
                df['target_direction'] = direction
    else:
        df['led_intensity'] = np.nan
        df['led_block'] = -1
        if add_stim_speed:
            df['stim_speed'] = np.nan
        if add_target_direction:
            df['target_direction'] = None
        print(f"  [warn] no run json for {assay}; led_intensity left NaN")

    # baked metadata
    meta = parse_assay_metadata(assay, json_path)
    df['assay'] = assay
    df['assay_type'] = os.path.basename(os.path.normpath(assay_type_dir))
    df['species'] = meta['species']
    df['age_days'] = meta['age_days']
    df['atr_days'] = meta['atr_days']
    df['fps'] = fps
    df['px_per_mm'] = px_per_mm

    # switch annotations from the FlyTracker actions.mat -> switching / switch_target
    if add_switches:
        from . import flytracker_actions as fa
        df = fa.merge_switches(df, fa.load_switches(assay_type_dir, assay))
    return df


def save_assay_df(df, assay_type_dir, assay):
    out_dir = processed_dir(assay_type_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{assay}.parquet')
    df.to_parquet(path, index=False)
    return path


def load_all_assays(assay_type_dir, verbose=True):
    """Concatenate every processed parquet under one assay_type."""
    files = sorted(glob.glob(os.path.join(processed_dir(assay_type_dir), '*.parquet')))
    if not files:
        print(f"No parquet files in {processed_dir(assay_type_dir)}")
        return None
    dfs = []
    for fp in tqdm(files, desc='Loading assays', disable=not verbose):
        dfs.append(pd.read_parquet(fp))
    combined = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"Combined: {len(combined)} rows, {combined['assay'].nunique()} assays, "
              f"species: {combined['species'].dropna().unique().tolist()}")
    return combined


def load_all_assay_types(root, verbose=True):
    """Concatenate processed parquets across every assay_type under root."""
    dfs = []
    for name in sorted(os.listdir(root)):
        atd = os.path.join(root, name)
        if os.path.isdir(processed_dir(atd)):
            df = load_all_assays(atd, verbose=verbose)
            if df is not None:
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def get_species_dfs(combined_df):
    """Split a combined df into {species: df}."""
    out = {sp: g.copy() for sp, g in combined_df.groupby('species')}
    for sp, df in out.items():
        print(f"  {sp}: {df['assay'].nunique()} assays, {len(df)} rows")
    return out
