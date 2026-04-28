#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example preprocessing and visualization for FlyTracker DYAD data.

Loads raw FlyTracker output (.mat files) for a single acquisition,
transforms coordinates, and produces diagnostic plots.

Expected directory layout::

    <rootdir>/<experiment>/                      (basedir)
    ├── videos/                                  (acquisition_parentdir)
    │   └── <acq>/                               (acq_dir, one per acquisition)
    │       ├── <acq>.avi                         video file
    │       ├── calibration.mat                   FlyTracker calibration
    │       └── <acq>/                            FlyTracker output subfolder
    │           ├── <acq>-track.mat               raw tracking data
    │           └── <acq>-feat.mat                derived features
    ├── processed_mats/                           (created by this script)
    │   └── <acq>_df.parquet                      processed DataFrame
    └── figures/                                  (created by this script)
        └── <acq>/                                per-acquisition figures
            ├── timecourse.png
            ├── timecourse_zoom.png
            ├── trajectory_2d.png
            └── video_overlay.png

Steps
-----
1. Set paths & plot style
   - rootdir / experiment / videos / <acq>  — raw FlyTracker output
   - Creates: <basedir>/processed_mats/     — saved .parquet DataFrames
   - Creates: <basedir>/figures/<acq>/       — per-acquisition figure folder

2. Load FlyTracker data
   - Reads calibration.mat, -track.mat, -feat.mat via libs.utils.
   - If calibration w/h are missing, they are estimated from rois or tracking data.

3. Process & transform
   - Centers coordinates on frame center, translates to focal-fly origin,
     computes relative metrics (theta_error, dist_to_other, etc.).
   - Saves result as <acq>_df.parquet in processed_mats/.

4. Add actions
   - Optionally loads pre-labelled action bouts (e.g., chasing) via add_ft_actions.
   - Otherwise, manually defines a chasing window for demonstration.

5. Prepare fly & target DataFrames
   - Splits id==0 (fly) and id==1 (target/dot).
   - Recomputes angular velocity, NaNs head-tail flip frames,
     converts angles to degrees.

6. Find chasing snippet & plot diagnostics
   - Finds a continuous chasing bout of snippet_dur_sec seconds.

   Figures (saved to <basedir>/figures/<acq>/):
     timecourse.png       — theta_error, angular velocity, distance vs time
     timecourse_zoom.png  — zoomed-in view of the same signals
     trajectory_2d.png    — 2D fly + target trajectory with heading arrows,
                            plus relative target position in fly-centric frame
     video_overlay.png    — sampled video frames with trajectory overlay
                            (only if .avi video is found)
"""
#%%
import os
import glob
import importlib
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rel
from analyses.preprocessing.src.add_ft_actions import add_ft_actions

#%%
# Set plot params
# -------------------------------------------------------------------
plot_style = 'dark'
min_fontsize = 12
putil.set_sns_style(style=plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style == 'dark' else 'k'

#%%
# Paths
# -------------------------------------------------------------------
rootdir = '/Volumes/Julie/2d-projector/example-data' # root directory containing experiment data
experiment = 'one-dot' 
basedir = os.path.join(rootdir, experiment)  # base directory for 
acquisition_parentdir = os.path.join(basedir, 'videos')
# Get list of acquisitions (no hidden)
acqs = [f for f in os.listdir(acquisition_parentdir) if not f.startswith('.')]
print(f"Found {len(acqs)} acquisitions")

# Processed data directory
processedmat_dir = os.path.join(basedir, 'processed_mats')
if not os.path.exists(processedmat_dir):
    os.makedirs(processedmat_dir)

# Figure directory
figdir = os.path.join(basedir, 'figures')
os.makedirs(figdir, exist_ok=True)
print(f"Saving figures to {figdir}")
# %%
# Load FlyTracker data 
# -----------------------------------------
acq = acqs[0]
acq_dir = os.path.join(acquisition_parentdir, acq)
calib, trk, feat = util.load_flytracker_data(acq_dir, 
                        calib_is_upstream=False, filter_ori=True)
print(f"Loaded data for {acq}")
print(calib)

# Save directory for this acquisition
# -------------------------------------------------------------------
save_dir = os.path.join(figdir, acq)
os.makedirs(save_dir, exist_ok=True)


# %%
# Process and transform
# -----------------------------------------
frame_width = calib['w']
frame_height = calib['h']
cop_ix = None
fps = calib['FPS']

trk['ori'] = -1*trk['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
df = rel.do_transformations_on_df(trk, frame_width, frame_height, 
                                feat_=feat, cop_ix=None,
                                flyid1=0, flyid2=1, 
                                get_relative_sizes=False)
# Save processed df
df_fpath = os.path.join(processedmat_dir, f'{acq}_df.parquet')
df.to_parquet(df_fpath)
print(f"Saved processed df to {df_fpath}")

# %%
# Add actions if not already present
# -----------------------------------------
load_actions = False
#if 'chasing' not in df.columns:
if load_actions:
    df = add_ft_actions(processedmat_dir, acquisition_parentdir, acq, 
                verbose=True)
else:
    # Manually examine specific frames:
    df['chasing'] = 0
    frame_start = round(200*fps)
    frame_end = frame_start + round(100*fps)
    frames_chasing = np.arange(frame_start, frame_end)
    df.loc[frames_chasing, 'chasing'] = 1

#%%
# Prepare fly (id==0) and target (id==1) dataframes
# -----------------------------------------
f1 = df[df['id'] == 0].sort_values('frame').copy()
targ_df = df[df['id'] == 1].sort_values('frame').copy()
f1['file_name'] = acq

# Recompute ang_vel_fly with head-tail flip correction
# f1['ang_vel_fly'] = util.recompute_ang_vel(f1['ori'], fps=fps)

# NaN out head-tail flip frames (ori jumps > 90 deg) + 1 frame margin
ori_diff = f1['ori'].diff().abs()
flip_mask = ori_diff > np.pi / 2
flip_mask = flip_mask | flip_mask.shift(1, fill_value=False) | flip_mask.shift(-1, fill_value=False)
n_flips = flip_mask.sum()
pct_flips = n_flips / len(f1) * 100
print(f"Head-tail flips: {n_flips} frames ({pct_flips:.2f}%)")
f1.loc[flip_mask, ['ori', 'ang_vel_fly']] = np.nan

# Convert some stuff to deg 
f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])
f1['ang_vel_fly_deg'] = np.rad2deg(f1['ang_vel_fly'])
f1['ang_vel_deg'] = -1 * np.rad2deg(f1['ang_vel'])
f1['ori_deg'] = np.rad2deg(f1['ori'])
f1['sec'] = f1['frame'] / fps


#%%
# Find a chasing snippet
# -------------------------------------------------------------------
snippet_dur_sec = 3

assert 'chasing' in f1.columns, "'chasing' column missing -- run add_ft_actions first"
assert f1['chasing'].sum() > 0, f"No chasing frames found in {example_file}"

stretch, f_start, f_end = util.find_action_snippet(f1, action='chasing', fps=fps,
                                                    snippet_dur_sec=snippet_dur_sec)
stretch_targ = targ_df[(targ_df['frame'] >= f_start) & (targ_df['frame'] <= f_end)]
print(f"Snippet: frames {f_start}–{f_end} ({(f_end - f_start)/fps:.1f}s)")

#%%
example_lag = 12
frame_step = 6
arrow_len = 20

# -------------------------------------------------------------------
# Figure 1: time-course
# -------------------------------------------------------------------
fig_tc, _ = putil.diagnostics_plot_timecourses(
    stretch, f_start, f_end, example_lag=example_lag,
    fps=fps, bg_color=bg_color,
    title=f'{acq}  |  frames {f_start}–{f_end}')
fig_tc.savefig(os.path.join(save_dir, 'timecourse.png'),
               dpi=150, bbox_inches='tight')

# -------------------------------------------------------------------
# Figure 1b: zoomed time-course
# -------------------------------------------------------------------
fig_zoom, _ = putil.diagnostics_plot_timecourses_zoom(stretch, bg_color=bg_color)
fig_zoom.savefig(os.path.join(save_dir, 'timecourse_zoom.png'),
                 dpi=150, bbox_inches='tight')

# -------------------------------------------------------------------
# Figure 2: 2D trajectory + relative target position
# -------------------------------------------------------------------
fig2, _ = putil.diagnostics_plot_2d_traj_and_rel(
    stretch, stretch_targ, f_start, f_end,
    arrow_len=arrow_len,
    title=f'{acq}  |  {(f_end - f_start)/fps:.1f}s snippet')
fig2.savefig(os.path.join(save_dir, 'trajectory_2d.png'),
             dpi=150, bbox_inches='tight')

#%%
# -------------------------------------------------------------------
# Figure 3: Video frame overlay
# -------------------------------------------------------------------
video_path = os.path.join(acquisition_parentdir, acq)
found_vid = util.find_video(video_path)

if found_vid is not None:
    print(f"Loading video: {found_vid}")
    frames_to_show = list(range(f_start, f_end, frame_step))
    n_overlay = len(frames_to_show)

    fig_vid, ax_vid = plt.subplots(1, 1, figsize=(8, 8))
    putil.plot_video_overlay(ax_vid, found_vid, frames_to_show,
                             flydf=stretch, targdf=stretch_targ,
                             f_start=f_start, f_end=f_end,
                             arrow_len=arrow_len)
    ax_vid.set_title(f'{acq}\nframes {f_start}–{f_end}, '
                     f'every {frame_step}th ({n_overlay} frames)')
    plt.tight_layout()
    fig_vid.savefig(os.path.join(save_dir, 'video_overlay.png'),
                    dpi=150, bbox_inches='tight')
else:
    print(f"No video found at {video_path}")

print(f"\nAll figures saved to {save_dir}")

# %%
