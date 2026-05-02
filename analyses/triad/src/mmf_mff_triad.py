#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-05-01
Author: Julie Chen
"""
#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import glob
from adjustText import adjust_text

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
rootdir = '/Volumes/Julie/20260427_free_behavior/' # root directory containing experiment data
acquisition_parentdir = os.path.join(rootdir, 'raw_videos')
# Get list of acquisitions (no hidden)
acqs = [f for f in os.listdir(acquisition_parentdir) if not f.startswith('.')]
print(f"Found {len(acqs)} acquisitions")

# Processed data directory
processedmat_dir = os.path.join(rootdir, 'processed_mats')
if not os.path.exists(processedmat_dir):
    os.makedirs(processedmat_dir)

# Figure directory
figdir = os.path.join(rootdir, 'figures')
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
# Grab first frame from .avi
# -------------------------------------------------------------------
avi_files = glob.glob(os.path.join(acq_dir, '*.avi'))
assert len(avi_files) > 0, f"No .avi found in {acq_dir}"
avi_path = avi_files[0]

cap = cv2.VideoCapture(avi_path)
ret, frame = cap.read()
cap.release()
assert ret, "Failed to read first frame"
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# %%
# Get first-frame fly data and ROI
# -------------------------------------------------------------------
frame_height, frame_width = frame_rgb.shape[:2]

# ROI: [x_start, y_start, width, height]
roi = calib['rois']  # [y0, x0, roi_h, roi_w]
y0, x0, roi_h, roi_w = roi

# First frame per fly
trk0 = trk.groupby('id').first().reset_index()

# %%
# Plot
# -------------------------------------------------------------------
# Colors per fly id
id_colors = {0: 'dodgerblue', 1: 'tomato', 2: 'limegreen'}

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(frame_rgb, origin='upper')

# ROI bounding box
rect = plt.Rectangle((x0, y0), roi_w, roi_h,
                      linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
ax.add_patch(rect)

texts = []
for i, row in trk0.iterrows():
    fly_id = int(row['id'])
    px, py = row['pos_x'], row['pos_y']
    ori = row['ori']
    color = id_colors.get(fly_id, 'white')

    ax.scatter(px, py, s=120, color=color, zorder=5, label=f'Fly {fly_id}')

    arrow_len = 30
    dx = arrow_len * np.cos(ori)
    dy = arrow_len * np.sin(ori)
    ax.annotate('', xy=(px + dx, py + dy), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

    t = ax.text(px, py,
                f"id={fly_id}\n({px:.0f}, {py:.0f})\nori={np.rad2deg(ori):.1f}°",
                color=color, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
    texts.append(t)

adjust_text(texts, ax=ax,
            expand=(1.5, 1.5),
            arrowprops=dict(arrowstyle='-', color='white', lw=0.8, alpha=0.6))

ax.set_xlim(0, frame_width)
ax.set_ylim(frame_height, 0)
ax.set_xlabel('x (pixels)')
ax.set_ylabel('y (pixels)')
ax.set_title(f'{acq}\nFirst frame — fly positions & orientations')
ax.legend(loc='upper right', fontsize=9)

ax.set_xticks(np.arange(0, frame_width, 100))
ax.set_yticks(np.arange(0, frame_height, 100))
ax.grid(True, color='white', linewidth=0.5, alpha=0.3)

plt.tight_layout()
savepath = os.path.join(save_dir, 'first_frame_sanity_check.png')
fig.savefig(savepath, dpi=150)
print(f"Saved to {savepath}")
plt.show()

# %%
