#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Julie Chen
Date: 2026-05-01
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import glob

import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rel
from analyses.triad.src import multi_funcs as mf
from analyses.triad.src import util as tutil
from analyses.triad.src import data_io
from analyses.triad.src.flip import (
    detect_flip_events, correct_flip_events, correct_mounting_nearest_ori,
)

# Set plot params
# -------------------------------------------------------------------
PLOT_STYLE = 'dark'
MIN_FONT_SIZE = 12
putil.set_sns_style(style=PLOT_STYLE, min_fontsize=MIN_FONT_SIZE)
BG_COLOR = [0.7]*3 if PLOT_STYLE == 'dark' else 'k'

# Focal fly ids per triad/assay type — only pairs involving at least one focal fly are processed
FOCAL_FLIES = {
    'MMF':  [0, 1],
    'MFF':  [0],
    'MMMF': [0, 1, 2],
}

DELTA_ANGLE_DEG = 30  # for assigning courtship targets based on orientation
MIN_COURTSHIP_BOUT_SEC = 2.0  # minimum duration of courtship bout to assign targets
JUMP_THRESHOLD_RAD = 1.5  # for detecting sharp flips in orientation
STABILITY_THRESHOLD_RAD = 0.8  # for confirming stability of new orientation after flip
INTERACTIVE_FLIP_THRESHOLD_SEC = 4.0  # auto-detected events longer than this trigger interactive review

# Helper functions 
def create_output_dirs(rootdir):
     # Processed data directory
    processedmat_dir = os.path.join(rootdir, 'processed_mats')
    if not os.path.exists(processedmat_dir):
        os.makedirs(processedmat_dir)

    # Figure directory
    figdir = os.path.join(rootdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    print(f"Saving figures to {figdir}")
    
    return processedmat_dir, figdir

def main():
    # user specified rootdir as command line argument
    if len(sys.argv) > 1:
        rootdir = sys.argv[1]

    acquisition_parentdir = os.path.join(rootdir, 'raw_videos')
    # Get list of acquisitions (no hidden)
    acqs = sorted([f for f in os.listdir(acquisition_parentdir) if not f.startswith('.')])
    print(f"Found {len(acqs)} acquisitions")
    processedmat_dir, figdir = create_output_dirs(rootdir)

    # iterate over acquisitions
    for acq in acqs:
        print(f"Processing acquisition {acq}...")
        acq_dir = os.path.join(acquisition_parentdir, acq)

        # Skip if track.mat not yet available
        trk_files = glob.glob(os.path.join(acq_dir, '*', '*-track.mat'))
        if len(trk_files) == 0:
            print(f"  No -track.mat found in {acq_dir}, skipping.")
            continue

        # Load FlyTracker data
        calib, trk, feat = util.load_flytracker_data(acq_dir,
                                calib_is_upstream=False, filter_ori=True)
        print(f"Loaded data for {acq}")

        # Split into per-chamber subsets (no-op for single-chamber data)
        chambers = mf.split_by_chamber(trk, feat, calib)
        print(f"  {len(chambers)} chamber(s) detected")

        for ch_idx, (trk_ch, feat_ch, calib_ch) in enumerate(chambers):
            acq_key = f'{acq}_ch{ch_idx}' if len(chambers) > 1 else acq

            # Skip if already processed
            processed_fp = os.path.join(processedmat_dir, f'{acq_key}.parquet')
            if os.path.exists(processed_fp):
                print(f"  Processed df already exists for {acq_key}, skipping.")
                continue

            FPS = calib_ch['FPS']
            CENTROID_X = calib_ch['centroids'][0]
            CENTROID_Y = calib_ch['centroids'][1]
            cop_ix = None

            # Save directory for this acquisition / chamber
            save_dir = os.path.join(figdir, acq_key)
            os.makedirs(save_dir, exist_ok=True)

            # Process and transform
            # -----------------------------------------
            # Determine pairs to process: all pairs where at least one fly is focal
            meta = data_io.parse_acquisition_metadata(acq)
            triad_type = meta['triad_type']
            focal_fly_ids = FOCAL_FLIES.get(triad_type)
            all_fly_ids = sorted(trk_ch['id'].unique())
            if focal_fly_ids is not None:
                pairs = [(i, j) for i, j in itertools.combinations(all_fly_ids, 2)
                         if i in focal_fly_ids or j in focal_fly_ids]
            else:
                print(f"  Warning: no FOCAL_FLIES entry for triad type '{triad_type}', using all pairs")
                pairs = list(itertools.combinations(all_fly_ids, 2))
            print(f"  Triad type: {triad_type}, focal flies: {focal_fly_ids}, pairs: {pairs}")

            # ── Round 1: pairwise metrics (needed for flip correction) ────────
            pair_trk_list = []
            pair_feat_cache = {}
            for flyid1, flyid2 in pairs:
                trk_ = trk_ch[trk_ch['id'].isin([flyid1, flyid2])].copy()
                feat_ = feat_ch[feat_ch['id'].isin([flyid1, flyid2])].copy()
                trk_, feat_ = mf.add_pairwise_metrics(trk_, feat_, calib_ch,
                                                       flyid1=flyid1, flyid2=flyid2)
                # copy dist_to_other from feat_ into trk_ and compute abs_ang_between
                # so that df_pairwise has the columns needed for flip correction
                for fid, oid in [(flyid1, flyid2), (flyid2, flyid1)]:
                    fmask = trk_['id'] == fid
                    omask = trk_['id'] == oid
                    trk_.loc[fmask, 'dist_to_other'] = feat_.loc[feat_['id'] == fid, 'dist_to_other'].values
                    f_pos = trk_.loc[fmask, ['pos_x', 'pos_y']]
                    o_pos = trk_.loc[omask, ['pos_x', 'pos_y']]
                    vec = o_pos.values - f_pos.values
                    trk_.loc[fmask, 'abs_ang_between'] = np.arctan2(vec[:, 1], vec[:, 0])
                trk_['pair'] = f"{flyid1}_{flyid2}"
                pair_trk_list.append(trk_)
                pair_feat_cache[(flyid1, flyid2)] = feat_
            df_pairwise = pd.concat(pair_trk_list, ignore_index=True)

            # flip ori sign (FT math/y-up → image/y-down convention) so that ori is
            # consistent with abs_ang_between (derived from pixel positions, y-down)
            # before flip correction and mounting correction run.
            df_pairwise['ori'] = -1 * df_pairwise['ori']

            # ── Action annotations ────────────────────────────────────────────
            num_columns_before = df_pairwise.shape[1]
            df_pairwise = util.load_and_assign_ft_actions(df_pairwise, acq_dir, acq)
            has_actions = df_pairwise.shape[1] > num_columns_before
            if not has_actions:
                print("No new action columns added. Check if FT actions file exists and has expected columns.")

            # ── Trim at copulation before flip correction ─────────────────────
            cop_frame = tutil.find_copulation_frame(df_pairwise)
            if cop_frame is not None:
                print(f"First copulation frame: {cop_frame} — trimming trk and feat before flip correction.")
                df_pairwise = df_pairwise[df_pairwise['frame'] <= cop_frame].copy()
                pair_feat_cache = {k: v[v['frame'] <= cop_frame].copy()
                                   for k, v in pair_feat_cache.items()}

            # ── Orientation flip correction ───────────────────────────────────
            if not has_actions:
                print("  No actions file — skipping mounting and flip correction.")
            else:
                if 'mounting attempt' in df_pairwise.columns:
                    df_pairwise = correct_mounting_nearest_ori(df_pairwise,
                                                               action_col='mounting attempt')

                avi_path = os.path.join(acq_dir, f'{acq}.avi')
                has_video = os.path.exists(avi_path)
                if not has_video:
                    print(f"  No video found at {avi_path} — flip correction will be auto-only")

                events = detect_flip_events(df_pairwise, fps=FPS,
                                            jump_threshold_rad=JUMP_THRESHOLD_RAD,
                                            stability_threshold_rad=STABILITY_THRESHOLD_RAD,
                                            interactive=has_video,
                                            video_path=avi_path if has_video else None,
                                            interactive_threshold_sec=INTERACTIVE_FLIP_THRESHOLD_SEC)

                if len(events) > 0:
                    df_pairwise = correct_flip_events(df_pairwise, events)

            # ── Round 2: transformations on focal fly with corrected ori ──────
            pair_df_list = []
            for flyid1, flyid2 in pairs:
                pair_key = f"{flyid1}_{flyid2}"
                trk_ = df_pairwise[df_pairwise['pair'] == pair_key].copy()
                feat_ = pair_feat_cache[(flyid1, flyid2)]

                pair_df = rel.do_transformations_on_df(trk_, CENTROID_X, CENTROID_Y,
                                                       feat_=feat_, cop_ix=cop_ix,
                                                       flyid1=flyid1, flyid2=flyid2,
                                                       get_relative_sizes=False)
                pair_df['pair'] = pair_key
                pair_df['acquisition'] = acq_key
                pair_df['species'] = 'mel' if 'mel' in acq else 'yak'
                pair_df_list.append(pair_df)
            df = pd.concat(pair_df_list, ignore_index=True)

            # add metadata about calibration and acquisition to df for potential future use
            df['PPM'] = calib_ch['PPM']
            df['FPS'] = calib_ch['FPS']

            # ── Target annotation ─────────────────────────────────────────────
            if not has_actions:
                data_io.save_processed_df(df, acq_key, calib_ch, processedmat_dir)
                continue

            if 'circling' in df.columns:
                df = tutil.assign_target_nearest(df, action_col='circling')
            if 'mounting attempt' in df.columns:
                df = tutil.assign_target_nearest(df, action_col='mounting attempt')
            if 'copulation' in df.columns:
                df = tutil.assign_target_nearest(df, action_col='copulation')
            if 'courtship' in df.columns:
                ### TODO - need to tune delta_theta_deg and min_bout_sec params eventually
                df = tutil.assign_target_orientation(df, action_col="courtship", fps=FPS,
                                                     delta_theta_deg=DELTA_ANGLE_DEG,
                                                     min_bout_sec=MIN_COURTSHIP_BOUT_SEC)

            data_io.save_processed_df(df, acq_key, calib_ch, processedmat_dir)

if __name__ == "__main__":
    main()