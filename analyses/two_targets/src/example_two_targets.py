#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example pipeline for two-target (3-object) FlyTracker data.

Loads raw FlyTracker output for an acquisition with one fly and two
dot targets, computes pairwise relative metrics for each fly-target
pair, and produces trajectory plots.

Expected directory layout::

    <rootdir>/<experiment>/                      (basedir)
    ├── videos/                                  (acquisition_parentdir)
    │   └── <acq>/                               (acq_dir, one per acquisition)
    │       ├── <acq>.avi                         video file
    │       ├── calibration.mat                   FlyTracker calibration
    │       └── <acq>/                            FlyTracker output subfolder
    │           ├── <acq>-track.mat               raw tracking (3 objects)
    │           └── <acq>-feat.mat                derived features
    ├── processed_mats/                           (created by this script)
    │   └── <acq>_df.parquet                      pairwise-transformed DataFrame
    └── figures/                                  (created by this script)
        ├── trajectories_overlaid.png             all 3 objects
        └── trajectories_two_dots.png             pairwise scatter plots

Steps
-----
1. Set paths & plot style
   - rootdir / experiment / videos / <acq>  — raw FlyTracker output
   - Creates: <basedir>/figures/            — saved figure files
   - Creates: <basedir>/processed_mats/     — saved .parquet DataFrames

2. Load FlyTracker data
   - Reads calibration.mat, -track.mat, -feat.mat via libs.utils.
   - Expects 3 tracked objects: id 0 (fly), id 1 (target A), id 2 (target B).

3. Plot raw trajectories
   - Overlays all three object trajectories color-coded by id.

4. Pairwise transform
   - Loops over pairs (0,1) and (0,2).
   - Adds dist_to_other and facing_angle if missing (via multi_funcs).
   - Runs do_transformations_on_df for each pair (centering, translating,
     computing theta_error, etc.).
   - Concatenates results with a 'pair' label column.
   - Saves result as <acq>_df.parquet in processed_mats/.

5. Plot pairwise trajectories
   - Side-by-side scatter of the fly trajectory for each pair, colored by
     dist_to_other, with the corresponding target path overlaid.

   Figures (saved to <basedir>/figures/):
     trajectories_overlaid.png   — all 3 objects on one axis
     trajectories_two_dots.png   — side-by-side pairwise scatter plots
"""
#%%
import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import analyses.two_targets.src.multi_funcs as mf
import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rel


#%% 
# Plotting
# -------------------------------------------------------------------
plot_style = 'dark'
min_fontsize = 12
putil.set_sns_style(plot_style, min_fontsize=min_fontsize)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'
species_colors = {'mel': 'cyan', 'yak': 'magenta'}

#%%
# Set data paths
# -------------------------------------------------------------------
rootdir = '/Volumes/Julie/2d-projector/example-data' # root directory containing experiment data
experiment = 'two-dots' 
basedir = os.path.join(rootdir, experiment)  # base directory for 
acquisition_parentdir = os.path.join(basedir, 'videos')
acqs = os.listdir(acquisition_parentdir)
print(f"Found {len(acqs)} acquisitions")

# Ouptut dir for figures
figdir = os.path.join(basedir, 'figures')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print(f"Saving figs to: {figdir}")

#%%
# Load FlyTracker data
acq = acqs[0]
acq_dir = os.path.join(acquisition_parentdir, acq)
calib, trk, feat = util.load_flytracker_data(acq_dir, 
                        calib_is_upstream=False, filter_ori=True)
print(f"Loaded data for {acq}")
print(calib)

#%%
# Plot trajectories
target_colors = {0: 'r', 1: 'b', 2: 'g'}
fig, ax = plt.subplots(figsize=(10, 10))
for id, trk_ in trk.groupby('id'):
    ax.plot(trk_['pos_x'], trk_['pos_y'], 
        color=target_colors[id], alpha=0.5, lw=0.5)

# Save fig
figname = 'trajectories_overlaid'
fig_fpath = os.path.join(figdir, f'{figname}.png')
plt.savefig(fig_fpath)
print(fig_fpath)

# %%
# Transform data
processedmat_dir = os.path.join(basedir, 'processed_mats')
if not os.path.exists(processedmat_dir):
    os.makedirs(processedmat_dir)


frame_width = calib['w']
frame_height = calib['h']
cop_ix = None

# Get pairwise info
df_list = []
for flyid1, flyid2 in [(0, 1), (0, 2)]:
    curr_ids = [flyid1, flyid2]
    trk_ = trk[trk['id'].isin(curr_ids)].copy()
    feat_ = feat[feat['id'].isin(curr_ids)].copy()

    # Add dist_to_other and facing_angle if not in columns
    if 'dist_to_other' not in feat_.columns:
        trk_, feat_ = mf.add_pairwise_metrics(trk_, feat_, calib,
                                                   flyid1=flyid1, flyid2=flyid2)

    trk_['ori'] = -1*trk_['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
    tmpdf = rel.do_transformations_on_df(trk_, frame_width, frame_height, 
                                    feat_=feat_, cop_ix=cop_ix,
                                    flyid1=flyid1, flyid2=flyid2, 
                                    get_relative_sizes=False)

    tmpdf['pair'] = f"{flyid1}_{flyid2}"
    tmpdf['acquisition'] = acq
    tmpdf['species'] = 'mel' if 'mel' in acq else 'yak'

    df_list.append(tmpdf)

df = pd.concat(df_list)

# Save processed df
df.to_parquet(os.path.join(processedmat_dir, f'{acq}_df.parquet'), engine='pyarrow', compression='snappy')
print(f"Saved processed df to {os.path.join(processedmat_dir, f'{acq}_df.parquet')}")

#%%
# Plot trajectories for each:
hue_var = 'dist_to_other'
palette = 'viridis'
hue_min, hue_max = df[hue_var].min(), df[hue_var].max()
hue_norm = mpl.colors.Normalize(vmin=hue_min, vmax=hue_max)

fig, axn = plt.subplots(1, 2, figsize=(10, 5),
                            sharex=True, sharey=True)
for i, (fpair, pairdf_) in enumerate(df.groupby('pair')):
    ax=axn[i]
    sns.scatterplot(data=pairdf_[pairdf_['id']==0], ax=ax, 
                    x='pos_x', y='pos_y', 
                    #hue='theta_error', palette='RdBu',
                    hue=hue_var, palette=palette,
                    hue_norm=hue_norm,
                    size='dist_to_other', size_norm=(0, 500),
                    alpha=1, edgecolor='none', legend=0)
    # other flyid
    other_id = int(fpair.split('_')[1])
    ax.plot(pairdf_[pairdf_['id']==other_id]['pos_x'], 
            pairdf_[pairdf_['id']==other_id]['pos_y'], 
            color=bg_color, 
            markersize=10)
    ax.set_title(f'{fpair}')
    ax.set_aspect(1)

# add shared colormap
putil.colorbar_from_mappable(axn[0], hue_norm, palette, 
                axes=[0.92, 0.3, 0.01, 0.4], fontsize=7,
                hue_title=hue_var)

# save fig
figname = 'trajectories_two_dots'
fig_fpath = os.path.join(figdir, f'{figname}.png')
plt.savefig(fig_fpath)
print(fig_fpath)
# %%
