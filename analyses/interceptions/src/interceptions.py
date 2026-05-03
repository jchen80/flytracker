#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 01, 14:14:00 2024

@filename: interceptions.py
@author: julianarhee

"""
#%%
import os
import sys
import glob
import importlib

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import pylab as pl
import matplotlib as mpl

import matplotlib.gridspec as gridspec

import scipy.stats as spstats

#from relative_metrics import load_processed_data
import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rel
import parallel_pursuit as pp
import libs.dlc as dlc

import cv2

#%%

def plot_heading_vs_travelingdir_frames(curr_frames, f1, f2, cap=None, ax=None,
                                     var1='ori', var2='traveling_dir', plot_interval=2,
                                     col1='magenta', col2='dodgerblue', target_col='g'):
    if ax is None:
        fig = pl.figure(figsize=(10,8))
        ax = fig.add_subplot(121) # axn = pl.subplots(1, 2)

    if cap is not None:
        cap.set(1, curr_frames[0])
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

        ax.imshow(im, cmap='gray')
        ax.set_title("Frame {}".format(curr_frames[0]), fontsize=8, loc='left')
        ax.invert_yaxis()

    for ix in curr_frames[0::plot_interval]:
        # set fly oris as arrows
        fly_marker = '$\u2192$' # https://en.wikipedia.org/wiki/Template:Unicode_chart_Arrows
        m_ori = np.rad2deg(f1.loc[ix][var1])
        m_trav = np.rad2deg(f1.loc[ix][var2])

        f_ori = np.rad2deg(f2.loc[ix][var2])

        marker_m1 = mpl.markers.MarkerStyle(marker=fly_marker)
        marker_m1._transform = marker_m1.get_transform().rotate_deg(m_ori)

        marker_m2 = mpl.markers.MarkerStyle(marker=fly_marker)
        marker_m2._transform = marker_m2.get_transform().rotate_deg(m_trav)

        marker_f = mpl.markers.MarkerStyle(marker=fly_marker)
        marker_f._transform = marker_f.get_transform().rotate_deg(f_ori)
        #print(np.rad2deg(fly1.loc[ix]['ori'])) #m_ori)
        #print(f_ori)


        # make a markerstyle class instance and modify its transform prop
        ax.plot([f1['pos_x'].loc[ix]], [f1['pos_y'].loc[ix]], 'w', 
                marker=marker_m1, markerfacecolor=col1, markersize=20, markeredgewidth=0.5, label=var1) 
        
        ax.plot([f1['pos_x'].loc[ix]], [f1['pos_y'].loc[ix]], 'w', 
                marker=marker_m2, markerfacecolor=col2, markersize=20, markeredgewidth=0.5, label=var2) 

        ax.plot([f2['pos_x'].loc[ix]], [f2['pos_y'].loc[ix]], 'w',
                marker=marker_f, markerfacecolor=target_col, markersize=20, markeredgewidth=0.5)
        ax.set_aspect(1)

    leg =putil.custom_legend(labels=[var1, var2, 'target'], colors=[col1, col2, target_col])

    ax.set_title('Frames: {}-{}'.format(int(curr_frames[0]), int(curr_frames[-1]) ))
    ax.legend(handles=leg, loc='upper left', bbox_to_anchor=(1,1), frameon=False)

    return


def get_acceleration_start(plotdf):
    # Get median value 
    med_val = plotdf['acc_smoothed2'].median()
    # Find where the first 20 consecutive frames are above the median:
    above_median_bool = plotdf['acc_smoothed2'] > med_val
    # Group by consecutive values
    above_median_groups = above_median_bool.ne(above_median_bool.shift()).cumsum()
    # Find the first group that has 20 or more True values
    first_group = above_median_groups[above_median_bool].value_counts().idxmax()
    # Get the first index of the group
    accel_start = above_median_groups[above_median_groups == first_group].index[0]

    return accel_start

def get_f1_and_f2(df, win=10, fps=60):
    f1 = df[df['id']==0].copy()
    f2 = df[df['id']==1].copy()
    f1.index = f1['frame'].values
    f2.index = f1['frame'].values

    f1 = rel.calculate_theta_error(f1, f2, xvar='pos_x', yvar='pos_y')
    f1 = rel.calculate_traveling_direction(f1, winsize=5)
    f2 = rel.calculate_traveling_direction(f2, winsize=5)

    # Calculate difference in ori between consecutive rows 
    f1['turn_size'] = f1['ori'].transform(lambda x: x.diff())
    f1['turn_size_smoothed'] = f1['turn_size'].rolling(win, 1).mean()

    # Calculate relative vel
    f1['rel_vel'] = f1['dist_to_other'].interpolate().diff() / f1['sec'].diff().mean()

    #%
    f1['vel_smoothed'] = dlc.lpfilter(f1['vel'], win) #, fps)
    f2['vel_smoothed'] = dlc.lpfilter(f2['vel'], win) #, fps)

    f1['acc_smoothed'] = np.concatenate((np.zeros(1), 
                            np.diff(f1['vel_smoothed']))) / (win/fps)
    f2['acc_smoothed'] = np.concatenate((np.zeros(1), 
                            np.diff(f2['vel_smoothed']))) / (win/fps)
    win = 10 
    f1['acc_smoothed2'] = f1['acc_smoothed'].rolling(win, 1).mean()

    return f1, f2

def get_targ_ang_vel_and_int_angle(boutdf, f1, 
                                   epoch_pre=60, epoch_post=0,
                                   pre_acc=0.03, post_acc=0.4):

    nframes_pre_acc = int(np.ceil(pre_acc * fps))
    nframes_post_acc = int(np.ceil(post_acc * fps))

    ang_int = {'interception_angle':[], 'targ_ang_vel': [],
               'targ_ang_vel_smoothed': []}
    for i, b_ in boutdf[boutdf['action']=='interception'].iterrows():
        start_, end_ = b_[['start', 'end']]
        curr_frames = np.arange(start_ - epoch_pre, end_ + epoch_post)
        plotdf = f1[f1['frame'].isin(curr_frames)].copy()
        accel_start = get_acceleration_start(plotdf)

        pre_frame = accel_start - nframes_pre_acc
        post_frame = accel_start + nframes_post_acc
        targ_ang_vel = f1.loc[pre_frame]['theta_error_dt']
        targ_ang_vel_sm = f1.loc[pre_frame]['theta_error_dt_smoothed']
        #interception_angle = f1.loc[post_frame]['abs_ang_between']
        interception_angle = f1.loc[post_frame]['traveling_dir']

        #if targ_ang_vel < -6:
        #    continue
        ang_int['interception_angle'].append(interception_angle)
        ang_int['targ_ang_vel'].append(targ_ang_vel)
        ang_int['targ_ang_vel_smoothed'].append(targ_ang_vel_sm)

    #%
    ang_int = pd.DataFrame(ang_int)

    return ang_int



# %%
# Set plotting
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=24)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#% plotting settings
curr_species = ['Dele', 'Dmau', 'Dmel', 'Dsant', 'Dyak']
species_cmap = sns.color_palette('colorblind', n_colors=len(curr_species))
print(curr_species)
species_palette = dict((sp, col) for sp, col in zip(curr_species, species_cmap))

# %%
assay = '2d-projector' # '38mm-dyad'

minerva_base = '/Volumes/Juliana'
procdir = os.path.join(minerva_base, '2d-projector-analysis/circle_diffspeeds/DeepLabCut/processed')

# local savedir for giant pkl
localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/{}/circle_diffspeeds/DeepLabCut'.format(assay)

# figure save dir
base_figdir = os.path.join(os.path.split(procdir)[0], 'interceptions')
if not os.path.exists(base_figdir):
    os.makedirs(base_figdir)
print("Saving figures to: {}".format(base_figdir))

#%%
# Load
outfile = os.path.join(localdir.replace('DeepLabCut', 'FlyTracker'), 'ftjaaba.pkl')
ftjaaba = pd.read_pickle(outfile)
ftjaaba[ftjaaba['species']=='Dele']['acquisition'].unique()

# Load processed and transformed DLC data
out_fpath_local = os.path.join(localdir, 'processed.pkl')
df0 = pd.read_pickle(out_fpath_local)

#%%
df0 = df0[df0['species'].isin(['ele'])]

#%% ------------------------------------------
# PURSUIT v. INTERCEPTIONS
# --------------------------------------------
# %% Get manually annotated actions -- annoted with FlyTracker

#20240214-0945_f1_Dele-wt_5do_sh_prj10_sz6x6
#20240214-0954_f1_Dele-wt_5do_sh_prj10_sz2x2
#20240214-1002_f1_Dele-wt_5do_sh_prj10_sz8x8
#20240214-1010_f1_Dele-wt_5do_sh_prj10_sz4x4
#20240214-1018_f1_Dele-wt_5do_sh_prj10_sz2x2_2
#20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10
#20240214-1033_f1_Dele-wt_5do_sh_prj10_sz6x6_2

#acq = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'
acq = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'
# acq = '20240214-0945_f1_Dele-wt_5do_sh_prj10_sz6x6'
# acq = '20240214-1033_f1_Dele-wt_5do_sh_prj10_sz6x6_2'

# Get video file
cap, viddir = util.get_video_cap_check_multidir(acq, assay=assay, return_viddir=True)

# Get path to actions file for current acquisition
action_fpaths = glob.glob(os.path.join(viddir, 'fly-tracker', '{}*'.format(acq), '*actions.mat'))
action_fpath = action_fpaths[0]
print(action_fpath)

# Load actions to df
boutdf = util.ft_actions_to_bout_df(action_fpath)
boutdf['acquisition'] = acq
boutdf.head()

#%%
acqdir = os.path.join(viddir, acq)

# DLC
projectname='projector-1dot-jyr-2024-02-18' 
#procdir = os.path.join(minerva_base, '2d-projector-analysis/DeepLabCut', projectname)
#print(len(os.listdir(procdir)))

#% get src paths
import libs.dlc as dlc
localroot = '/Users/julianarhee/DeepLabCut' # all these functions assume this is the local rootdir
#% Look at 1 data file
analyzed_dir = dlc.get_dlc_analysis_dir(projectname=projectname)

flyid = 'fly' # double check in the plots for abdomen lengths
dotid = 'single'
fps = 60  # Hz
max_jump = 6
pcutoff=0.8 #0.99

#%%
#acq_prefix = '20240214-1025_f1_*sz10x10'
#acq_prefix = '20240216-*fly3*6x6'
#acq_prefix = '20240222-*fly7_Dmel*8x8'
# acq_prefix = '20240216-*fly3_Dmel*4x4'
#acq_prefix = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'

#fpath = dlc.get_fpath_from_acq_prefix(analyzed_dir, acq_prefix)

# includes calc theta error
#df = dlc.load_and_transform_dlc(fpath, winsize=10,
#                                    localroot=localroot, projectname=projectname,
#                                    assay=assay, flyid=flyid, dotid=dotid, fps=fps, max_jump=max_jump, pcutoff=pcutoff,
#                                    heading_var='ori')
#df = df0[df0['acquisition']==acq_prefix].copy()
#df.head()

curr_fpath = glob.glob(os.path.join(procdir, '{}*.pkl'.format(acq)))[0]
print(curr_fpath)
df = pd.read_pickle(curr_fpath)
df.head()

#df = pp.load_processed_data(acqdir, load=True, savedir=procdir)
#processed_df_fpath = os.path.join(procdir, '{}_df.pkl'.format(acq))
#df = pd.read_pickle(processed_df_fpath)

figid = curr_fpath.split(minerva_base)[-1]
print(figid)

if plot_style == 'white':
   figdir = os.path.join(base_figdir, acq, 'white') 
else:
    figdir = os.path.join(base_figdir, acq)
if not os.path.exists(figdir):
    os.makedirs(figdir)

#%%
win = 10
fps = 60
f1, f2 = get_f1_and_f2(df, win=win, fps=fps)

assert f1.shape[0] == f2.shape[0]

# %%
#df['action'] = None
#for i, b_ in boutdf.iterrows():
#    start = b_['start']-1
#    end = b_['end']-1
#    action = b_['action']
#    curr_frames = np.arange(start, end+1)
#    df.loc[df['frame'].isin(curr_frames), 'action'] = action
#    df.loc[df['frame'].isin(curr_frames), 'action_num'] = i

# %% Plot trajectory with vectors

ix = 2
plot_interval = 3
action ='interception' # 'interception'
# 
start_, end_ = boutdf[boutdf['action']==action].iloc[ix][['start', 'end']]
print(start_, end_)

curr_frames = np.arange(start_, end_+1)

#% Plot traveling dir, heading, etc.
fig, ax =pl.subplots() #1, 2, figsize=(12,5))
plot_heading_vs_travelingdir_frames(curr_frames[0::plot_interval], f1, f2, ax=ax, cap=cap,
                                          var1='ori', var2='traveling_dir')                                       
ax.set_xticks([0, 200, 400, 600, 800])
pl.subplots_adjust(wspace=0.5)

putil.label_figure(fig, figid)
figname = 'trajectory_{}_example-bout_frames-{}-{}'.format(action, start_, end_)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

# %% Plot time courses

keypoint = 100 #80+35 #100

fps = 60
n_pre = 2*fps
n_post = -0.5*fps #-0.2*fps

curr_frames = np.arange(start_ - n_pre, end_ + n_post)
# curr_frames = curr_frames[0:100]
plotdf = f1[f1['frame'].isin(curr_frames)].copy()

accel_start = get_acceleration_start(plotdf)

#% Plot
plot_vars = ['theta_error', 'theta_error_dt', 'abs_ang_between',
             'dist_to_other', 'vel_smoothed', 'acc_smoothed2']
             #'acc_smoothed', 'acc_smoothed2', 'ang_vel_smoothed']

fig, axn = pl.subplots(len(plot_vars), 1, figsize=(10,20), sharex=True)
for var, ax in zip(plot_vars, axn):
    if var in ['theta_error',  'abs_ang_between']:
        yvals = np.rad2deg(plotdf[var]) #np.unwrap(plotdf[var])
    else:
        yvals = plotdf[var]
    ax.plot(plotdf['frame'], yvals, color=bg_color)
    ax.set_ylabel(var)
    ax.axvline(accel_start, color='red', linestyle='--')
    ax.axvline(start_+0, color=bg_color, linestyle='-')
    ax.axvline(start_+20, color='red', linestyle='--')
    if var in ['theta_error', 'vel_smoothed', 'acc_smoothed', 'acc_smoothed2']:
        ax.axhline(0, color=bg_color, linestyle='--')
    ax.axvline(curr_frames[keypoint], color='cornflowerblue', linestyle='--')

putil.label_figure(fig, figid)
figname = 'timecourses_example-bout_frames-{}-{}'.format(start_, end_)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%% zoom in to pre-post fixation
estimated_zero = start_ + 20 #curr_frames[keypoint]

fig, ax = pl.subplots()
ax.plot(plotdf['theta_error'], color=bg_color)
ax.axhline(0, color='gray', linestyle='--')
ax.axvline(accel_start, color='red', linestyle='--')
ax.axvline(estimated_zero, color='gray', linestyle='--')

nsec_to_zero = (estimated_zero - (accel_start))/fps
ax.set_title("Time to 0 theta-E: {:.2f} sec".format(nsec_to_zero))

putil.label_figure(fig, figid)
figname = 'timecourses_example-bout_frames-{}-{}_zoom'.format(start_, end_) 
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

# %%
#% Plot up until the keypoint (100 frames) 
fig, axn =pl.subplots(1, 2, figsize=(10, 5)) #1, 2, figsize=(12,5))
ax=axn[0]
plot_heading_vs_travelingdir_frames(curr_frames[25:keypoint][0::2], f1, f2, ax=ax, cap=cap,
                                          var1='ori', var2='traveling_dir'
                                          )
ax.legend_.remove()

ax=axn[1]
plot_heading_vs_travelingdir_frames(np.arange(accel_start, start_+60)[0::2], f1, f2, ax=ax, cap=cap,
                                          var1='ori', var2='traveling_dir'
                                          )
for ax in axn:
      ax.axis('off')

putil.label_figure(fig, figid)
figname = 'trajectory_example-bout_frames_PRE_PERI' #.format()
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

# %%
fig, axn =pl.subplots(2, 1) #.figure()
ax=axn[0]
ax.plot(f1['vel_smoothed'].loc[curr_frames])
ax=axn[1]
ax.plot(f1['acc_smoothed2'].loc[curr_frames])
ax.axvline(accel_start)


#%% 

# LEAD ANGLE vs. DISTance?
# -------------------------------
#%
f1['theta_error_abs'] = np.abs(f1['theta_error'])
#%% Look at 1 bout
ix = 0# #2
plot_interval = 3
action ='pursuit' # 'interception'
# 
start_, end_ = boutdf[boutdf['action']==action].iloc[ix][['start', 'end']]
#print(start_, end_)

curr_frames = np.arange(start_, end_+1)
plotdf = f1[f1['frame'].isin(curr_frames)].copy()

fig, ax =pl.subplots()
sns.scatterplot(x='dist_to_other', y='theta_error', data=plotdf, ax=ax)
#ax.set_ylim([-0.5, 0.5])

#%% Look at all of the bouts

action_type = 'pursuit'
c_list = []
for ix, bdf in boutdf[boutdf['action']==action_type].iterrows():
    start_, end_ = bdf[['start', 'end']].values                    
    curr_frames = np.arange(start_, end_+1)
    plotdf = f1[f1['frame'].isin(curr_frames)].copy()
    plotdf['boutnum'] = ix
    c_list.append(plotdf)
df_ = pd.concat(c_list)

#%% plot
fig, ax =pl.subplots()
sns.scatterplot(x='dist_to_other', y='theta_error', data=df_, ax=ax,
                hue='boutnum', palette='colorblind', alpha=0.5)
ax.legend_.remove()
ax.set_ylim([-2, 1]) 


#%%

#%%
# =======================================================
# RANGE VECTORS
# =======================================================
def get_current_bout(start_, end_, nsec_pre=0.5, nsec_post=0, fps=60):
    #bout_ = boutdf[boutdf['action']=='interception'].iloc[ix][['start', 'end']]
    #start_, end_ = bout_[['start', 'end']]
    bout_start = start_ - nsec_pre*fps
    bout_end = end_ + nsec_post*fps # look at a few sec before/after

    return bout_start, bout_end

def get_vectors_between(f1, f2, bout_start, bout_end, 
                        xvar='pos_x', yvar='pos_y'):
    # get current df
    curr_frames = f1.loc[bout_start:bout_end]['frame'].copy()
    f1_ = f1[(f1['frame'].isin(curr_frames)) & (~f1['pos_x'].isnull()) ]
    f2_ = f2[(f2['frame'].isin(f1_['frame']))] # & (~f1['pos_x'].isnull())]
    # update current frames to exclude nulls
    curr_frames = sorted(f1_['frame'].unique())

    # get vectors between f1 and f2    
    vecs  = pp.get_vector_between_flies(f1_, f2_, curr_frames, 
                                        xvar=xvar, yvar=yvar)
    return vecs, f1_, f2_

def get_range_vector_df(vecs, curr_frames, fps=60):
    #% calculate differences between pursuer-target vectors
    # get angle between consecutive vectors
    ang_diffs=[]
    for vi, v in enumerate(vecs):
        if vi == len(vecs)-1:
            break
        next_v = vecs[vi+1]
        ang_diffs.append(np.rad2deg(util.angle_between(next_v, v)))

    # calculate change in length between consecutive vectors
    vec_lens = [np.linalg.norm(v) for v in vecs]
    vec_len_diffs = np.diff(vec_lens)

    vecdf = pd.DataFrame({'ang_diff': ang_diffs, 'vec_len': vec_lens[1:],
                        'vec_len_diff': vec_len_diffs,
                'frame': curr_frames[1:]}).dropna()
    

    vecdf['sec'] = (vecdf['frame']-vecdf['frame'].iloc[-1]) / fps

    return vecdf


def plot_range_vectors(vecdf, vecs, f1_, f2_, ax=None, xvar='pos_x', yvar='pos_y', 
                       color_ang_diff=False, plot_style='dark', bg_color=[0.7]*3):

    min_val, max_val = vecdf['ang_diff'].min(), vecdf['ang_diff'].max()
    cmap = mpl.cm.Greys if plot_style=='dark' else mpl.cm.Greys_r
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    if color_ang_diff:
        color_list = cmap(vecdf['ang_diff'].values)
    else:
        color_list = [bg_color]*len(vecs) #v

    if ax is None:
        fig, ax = pl.subplots(figsize=(5,8)) #figsize=(6,6))

    ax.scatter(f1_[xvar], f1_[yvar], c=f1_['sec'], cmap='viridis', s=5)
    ax.scatter(f2_[xvar], f2_[yvar], c=f2_['sec'], cmap='viridis', s=5)
    ax.set_aspect(1)
    for i, v in enumerate(vecs[0:-1]): 
        c='k' if i==0 else color_list[i]
        pos1 = f1_.iloc[i][[xvar, yvar]].values
        pos2 = f2_.iloc[i][[xvar, yvar]].values
        ax.plot([ pos1[0], pos1[0]+v[0]], 
                [ pos1[1], pos1[1]+v[1]], c=c, lw=0.5)
    if color_ang_diff:
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                shrink=0.5, label='ang. diff between vectors')
    pl.subplots_adjust(left=0.1, right=0.9)

    return ax

from scipy.stats import pearsonr

def rolling_correlation(X, Y, window_size=5):
    # Pad X and Y arrays with NaNs to handle edges
    X_padded = np.pad(X, (window_size // 2, window_size // 2), mode='constant', constant_values=np.nan)
    Y_padded = np.pad(Y, (window_size // 2, window_size // 2), mode='constant', constant_values=np.nan)
    
    # Initialize correlation vector
    correlation_vector = np.zeros(len(X))
    
    # Calculate correlation for each timepoint using rolling window
    for i in range(len(X)):
        if not np.isnan(X_padded[i]):
            x_window = X_padded[i - window_size // 2: i + window_size // 2 + 1]
            y_window = Y_padded[i - window_size // 2: i + window_size // 2 + 1]
            valid_indices = ~np.isnan(x_window) & ~np.isnan(y_window)
            if np.sum(valid_indices) >= 2:
                correlation_vector[i] = pearsonr(x_window[valid_indices], y_window[valid_indices])[0]
            else:
                correlation_vector[i] = np.nan
    
    return correlation_vector

def pad_df(vecdf, winsize=20):
    # get correlation vector across time points bewteen angle diff and vector length across a moving window of size 5 frames:
    # pad vecdf with NaNs of winsize in the beginning and end to handle edges
    # add winsize NaNs in the beginning 
    vecdf_pads = pd.DataFrame(np.nan, index=np.arange(0, winsize), columns=vecdf.columns)
    vecdf_padded = pd.concat([vecdf_pads, vecdf, vecdf_pads], axis=0)#.reset_index(drop=True)
    # fill vecdf_padded with mirrored values at the edges
    #vecdf_padded.iloc[0:winsize] = vecdf.iloc[0:winsize].iloc[::-1]
    #vecdf_padded.iloc[-winsize:] = vecdf.iloc[-winsize:].iloc[::-1]
    vecdf_padded.iloc[0:winsize] = vecdf.iloc[0] #.iloc[::-1]
    vecdf_padded.iloc[-winsize:] = vecdf.iloc[-1] #-winsize:].iloc[::-1]
    #print(vecdf_padded.shape)
    return vecdf_padded

def rolling_window_corr(vecdf_padded, winsize=20):
    # winsize=20
    r_vals=[]; p_vals=[];
    for i in range(len(vecdf_padded)-winsize):
        r_, p_ = pearsonr(vecdf_padded['ang_diff'].iloc[i:i+winsize], 
                vecdf_padded['vec_len'].iloc[i:i+winsize])
        r_vals.append(r_)
        p_vals.append(p_)
    corrdf_padded = pd.DataFrame({'pearson_r': r_vals, 'p_val': p_vals})
    corrdf_padded = corrdf_padded.iloc[winsize:].reset_index(drop=True)

    return corrdf_padded

def pad_and_corr_range_vectors(vecdf, winsize=20):
    vecdf_padded = pad_df(vecdf, winsize=winsize)
    corrdf_padded = rolling_window_corr(vecdf_padded, winsize=winsize)

    corrdf_padded['frame'] = vecdf['frame'].values
    corrdf_padded['sec'] = vecdf['sec'].values

    return corrdf_padded

def get_range_vector_corrs_all_bouts(boutdf, f1, f2, action_type='interception', 
                                     nsec_pre=0, nsec_post=0, winsize=20,
                                     xvar='pos_x', yvar='pos_y'):
    c_list = []
    for ix, bdf in boutdf[boutdf['action']=='interception'].iterrows():
        start_, end_ = bdf[['start', 'end']].values                    
        #(start_, end_) in enumerate(boutdf[boutdf['action']=='interception'][['start', 'end']].values):
        #print(start_, end_)
        bout_start, bout_end = get_current_bout(start_, end_, nsec_pre=nsec_pre, nsec_post=nsec_post)

        vecs, f1_, f2_ = get_vectors_between(f1, f2, bout_start, bout_end,
                                             xvar=xvar, yvar=yvar)
        curr_frames = f1_['frame'].values
        vecdf = get_range_vector_df(vecs, curr_frames)

        corr = vecdf['ang_diff'].corr(vecdf['vec_len'])
        corrdf_padded = pad_and_corr_range_vectors(vecdf, winsize=winsize)

        corrdf_padded['boutnum'] = bdf['boutnum']
        corrdf_padded['likelihood'] = bdf['likelihood']
        corrdf_padded['overall_corr'] = corr
        corrdf_padded['acquisition'] = bdf['acquisition']
        corrdf_padded['start'] = bdf['start']
        corrdf_padded['end'] = bdf['end']

        #print(corrdf_padded.shape)

        c_list.append(corrdf_padded)

    corrdf = pd.concat(c_list).reset_index(drop=True)

    return corrdf



#%%
#ix = 2#5 #7
#nsec_pre = 0.5
#nsec_post = -0.2
nsec_pre = 0 #n_pre/fps
nsec_post = 0# n_post/fps #-0.2*fps

#xvar = 'pos_x_smoothed'
#yvar = 'pos_y_smoothed'
xvar = 'pos_x'
yvar = 'pos_y'
start_, end_ = boutdf[boutdf['action']=='interception'].iloc[ix][['start', 'end']]
bout_start, bout_end = get_current_bout( start_, end_, nsec_pre=nsec_pre, nsec_post=nsec_post)

vecs, f1_, f2_ = get_vectors_between(f1, f2, bout_start, bout_end,
                                     xvar=xvar, yvar=yvar)
curr_frames = f1_['frame'].values
vecdf = get_range_vector_df(vecs, curr_frames)

    
#% plot trajectories, color by angular diff
color_ang_diff = False
fig, ax= pl.subplots(figsize=(5,8))
ax = plot_range_vectors(vecdf, vecs, f1_, f2_, ax=ax, xvar=xvar, yvar=yvar, 
                         color_ang_diff=color_ang_diff, plot_style=plot_style,
                         bg_color=bg_color)
ax.set_title('Frames: {}-{}'.format(bout_start, bout_end))

putil.label_figure(fig, acq)
figname = 'vectors_example-bout_frames-{}-{}_nsec-pre-{}'.format(bout_start, bout_end, nsec_pre)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

#%% -----------------------------------------
# Range Vector Correlation
# ------------------------------------------
# Line-of-sight (LoS) vector -> facing_angle

# Get angle difference between successive LoS vectors,
# i.e., differences in facng angle
#ang_diffs = np.diff(np.unwrap(f1['facing_angle']))
#f1['facing_angle_diff'] = np.concatenate([[np.nan], ang_diffs])
#print(f1['facing_angle_diff'].min(), f1['facing_angle_diff'].max())

# Get difference in magnitude (length) of successive LoS vectors


#%%a
winsize = 20

importlib.reload(putil)
# get correlation bewteen angle diff and vector length:
corr = vecdf['ang_diff'].corr(vecdf['vec_len'])
print(corr)

# pad df of range vectors
#vecdf_padded = pad_df(vecdf, winsize=winsize)
# calculate rolling window correlation for plotting against time
corrdf_padded = pad_and_corr_range_vectors(vecdf, winsize=winsize)

# plot
fig, axn = pl.subplots(1, 2)
ax=axn[0]
sns.scatterplot(data=vecdf, ax=ax, x='ang_diff', y='vec_len', 
                hue='sec', palette='viridis', legend=0)
ax=axn[1]
sns.scatterplot(data=corrdf_padded, ax=ax, x='sec', y='pearson_r', 
                hue='sec', palette='viridis', legend=0)
fig.text(0.1, 0.9, 'Corr. between angle diff and vector length = {:.2f}'.format(corr))
ax.axhline(0, color='gray', linestyle='--')
ax.set_ylim([-1, 1])

fig.text(0.1, 0.85, 'Frames: {}-{}'.format(bout_start, bout_end))

#putil.add_colorbar(fig, ax, vmin=vecdf['sec'].min(), vmax=vecdf['sec'].max())
cmap='viridis'
norm = mpl.colors.Normalize(vmin=vecdf['sec'].min(), vmax=vecdf['sec'].max())
putil.colorbar_from_mappable(ax, norm, cmap, hue_title='sec', axes=[0.93, 0.3, 0.01, 0.4],
                        fontsize=7) #pad=0.05):

for ax in axn:
    ax.set_box_aspect(1)
pl.subplots_adjust(wspace=0.5)

putil.label_figure(fig, figid)
figname = 'range_vector_corr_example-bout_frames-{}-{}'.format(start_, end_)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figname)

#%%
fig, ax = pl.subplots()
ax.plot(corrdf_padded['sec'], corrdf_padded['pearson_r'])
#fig.text(0.1, 0.9, 'Corr. between angle diff and vector length = {:.2f}'.format(corr))
ax.axhline(0, color='gray', linestyle='--')
ax.set_ylim([-1, 1])



#%% Aggregate current aquisition, all interception bouts
#%
#ix = 7
nsec_pre = 0.0
nsec_post = 0
winsize = 20

xvar = 'pos_x'
yvar = 'pos_y'

corrdf = get_range_vector_corrs_all_bouts(boutdf, f1, f2, action_type='interception', 
                     nsec_pre=nsec_pre, nsec_post=nsec_post, winsize=winsize,
                     xvar=xvar, yvar=yvar)

#%% Plot all range vec correlations for all bouts

c_list = []
fig, ax = pl.subplots()
for ix, corrdf_ in corrdf.groupby('boutnum'):
    imax = corrdf_['pearson_r'].argmax()
    if imax < len(corrdf_)/4:
        imax = -1

    #corr_ = corrdf_.iloc[0:imax].copy() #['pearson_r']
    corr_ = corrdf_.copy()
    overall_corr = round(corr_['overall_corr'].unique()[0], 2)
    corr_['sec'] = corr_['sec'] - corr_['sec'].iloc[-1]
    corr_['overall_corr'] = overall_corr
    bnum = corrdf_['boutnum'].unique()[0]
    ax.plot(corr_['sec'], corr_['pearson_r'], label='{}, r={}'.format(bnum, overall_corr)) 
    #sns.scatterplot(data=corrdf_, ax=ax, x='sec', y='pearson_r', 
    #            hue='sec', palette='viridis', legend=0
    c_list.append(corr_)
ax.axhline(y=0, color='gray', linestyle='--')

tmpcorr = pd.concat(c_list)

ax.legend(bbox_to_anchor=(1,1), loc='upper left', frameon=False)

mean_corr_val = tmpcorr['overall_corr'].mean()

fig.text(0.1, 0.9, 'Corr. between angle diff and vector length = {:.2f}'.format(mean_corr_val))
ax.set_ylim([-1, 1])
ax.set_xlim([-1, 0])

putil.label_figure(fig, acq)
figname = 'range_vector_corrs_by_time_nsec-pre-{}'.format( nsec_pre)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))


#%%
fig, ax =pl.subplots()
sns.lineplot(data=tmpcorr, x='sec', y='pearson_r', ax=ax)
ax.axhline(0, color='gray', linestyle='--')

#%%


# %%
# =======================================================
# Get all actions
# =======================================================

basedir = os.path.join(minerva_base, assay)
len(os.listdir(basedir))

found_actions_paths = glob.glob(os.path.join(basedir, 
                        '20*', 'fly-tracker', '*ele*', '*actions.mat'))

#%%

#a_ = []
#for fp in found_actions_paths:
#    actions_ = util.ft_actions_to_bout_df(fp)
#    basename = '_'.join(os.path.split(fp)[-1].split('_')[0:-1])
#    print(basename)
#    actions_['acquisition'] = basename
#    #actions_['ction_num'] = actions_.index.tolist()
#    a_.append(actions_)
#
#all_actions = pd.concat(a_)
all_actions = util.load_ft_actions(found_actions_paths)

#%%
actions = all_actions.copy() #all_actions[all_actions['likelihood']>2].copy()


#%%
# AGGREGATE all F1
create_new=False
tmp_outfile = os.path.join(localdir, 'processed_elegans_f1_f2.pkl')

if create_new:
    f_list = []
    for acq_prefix, boutdf in actions[actions['action']=='interception'].groupby('acquisition'):

        fpath = dlc.get_fpath_from_acq_prefix(analyzed_dir, acq_prefix)

        # includes calc theta error
        df = dlc.load_and_transform_dlc(fpath, winsize=10,
                                localroot=localroot, projectname=projectname,
                                assay=assay, flyid=flyid, 
                                dotid=dotid, fps=fps, 
                                max_jump=max_jump, pcutoff=pcutoff,
                                heading_var='ori')
        # get f1 and f2 info
        f1, f2 = get_f1_and_f2(df, win=win, fps=fps)
        f_list.append(f1)
        f_list.append(f2)
    flydf = pd.concat(f_list)
    #%
    # Save
    print(tmp_outfile)
    flydf.to_pickle(tmp_outfile)
else:
    flydf = pd.read_pickle(tmp_outfile)

flydf.head()


#%%

# Select subset of traj for csv

acq = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'
ints = actions[(actions['acquisition']==acq) 
             & (actions['action']=='interception')].copy()
df_ = flydf[flydf['acquisition'] == acq].copy()
df_

currlist = []
for (start_, end_, bnum) in ints[['start', 'end', 'boutnum']].values:
    if bnum in [2, 5, 10]:
        curr_frames = np.arange(start_, end_+1)
        currdf = df_.loc[df_['frame'].isin(curr_frames)][['id', 'frame', 'pos_x', 'pos_y', 'sec']]
        currdf['bout'] = bnum #ints[ints['start']==start_]['boutnum'].values[0]

        currlist.append(currdf)

examples = pd.concat(currlist)
examples.to_csv(os.path.join(localdir, 'example_traj.csv'))

#%%
win=10
f_list = []
for (acq, fid), f1 in flydf.groupby(['acquisition', 'id']):
    f1['theta_error_smoothed'] = f1['theta_error'].rolling(win, 1).mean()  
    #f1['theta_error_dt_smoothed'] = f1['theta_error_dt'].rolling(win, 1).mean()  
    f1['theta_error_smoothed_dt'] = pd.Series(np.unwrap(f1['theta_error_smoothed'].interpolate().ffill().bfill())).diff() / f1['sec_diff'].mean()
    f1['theta_error_dt_smoothed'] = f1['theta_error_dt'].rolling(win, 1).mean()  

    f_list.append(f1)

flydf = pd.concat(f_list)
#%% # plot each bout

pos_smooth_win = 10
xvar = 'pos_x'
yvar = 'pos_y'

color_ang_diff = False

curr_actions = actions[(actions['action']=='interception')
                       & (actions['likelihood']>0)].copy()

print(curr_actions.shape)

nr = 5
nc = 6

#fig, axn = pl.subplots(nr, nc, figsize=(15,20))
ai = 0
for acq, acq_bouts in curr_actions.groupby('acquisition'):

    # Get tracked data
    curr_fpath = glob.glob(os.path.join(procdir, '{}*.pkl'.format(acq)))[0]
    df = pd.read_pickle(curr_fpath)
    f1, f2 = get_f1_and_f2(df, win=pos_smooth_win, fps=fps)

    print(acq)
    fig, axn = pl.subplots(1, len(acq_bouts), figsize=(len(acq_bouts)*2.5, 5))
    ai = 0
    # cycle thru bouts:
    fig.text(0.1, 0.9, acq)

    for ix, boutdf in acq_bouts.groupby('boutnum'):
        
        if len(acq_bouts) > 1:
            ax = axn[ai]
        else:
            ax = axn
        start_, end_, = boutdf[['start', 'end']].loc[ix] #.values
        bout_start, bout_end = get_current_bout( start_, end_, 
                                            nsec_pre=nsec_pre, nsec_post=nsec_post)

        vecs, f1_, f2_ = get_vectors_between(f1, f2, bout_start, bout_end,
                                            xvar=xvar, yvar=yvar)
        curr_frames = f1_['frame'].values
        vecdf = get_range_vector_df(vecs, curr_frames)

        ax = plot_range_vectors(vecdf, vecs, f1_, f2_, ax=ax, xvar=xvar, yvar=yvar, 
                            color_ang_diff=color_ang_diff, plot_style=plot_style,
                            bg_color=bg_color)
        ax.axis('off')
        ax.set_title(ix)

        ai += 1

#%%
included_bouts = {
    '20240214-0945_f1_Dele-wt_5do_sh_prj10_sz6x6': [0, 3, 4, 5],
    '20240214-0954_f1_Dele-wt_5do_sh_prj10_sz2x2': [2],
    '20240214-1002_f1_Dele-wt_5do_sh_prj10_sz8x8': [0, 2, 3],
    '20240214-1010_f1_Dele-wt_5do_sh_prj10_sz4x4': [0],
    '20240214-1018_f1_Dele-wt_5do_sh_prj10_sz2x2_2': [],
    '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10': [2,3, 4, 8, 10, 12],

    '20240214-1033_f1_Dele-wt_5do_sh_prj10_sz6x6_2': [6, 7, 8]
                  }

#%% Range Vector Correlations

nsec_pre = -0
nsec_post = -0.05
xvar = 'pos_x'
yvar = 'pos_y'


c_list = []
for acq, fly_ in flydf.groupby('acquisition'):
    incl_bouts = included_bouts[acq]
    boutdf_ = actions[(actions['acquisition']==acq) 
                      & (actions['action']=='interception')
                      & (actions['boutnum'].isin(incl_bouts))].copy()
    if boutdf_.shape[0]==0:
        continue
    f1 = fly_[fly_['id']==0].copy()
    f1.index = f1['frame'].values
    f2 = fly_[fly_['id']==1].copy()
    f2.index = f2['frame'].values
    corrdf_ = get_range_vector_corrs_all_bouts(boutdf_, f1, f2, action_type='interception', 
                     nsec_pre=nsec_pre, nsec_post=nsec_post, winsize=winsize,
                     xvar=xvar, yvar=yvar)
    corrdf_['acquisition'] = acq
    
    c_list.append(corrdf_)

aggr_corrs = pd.concat(c_list).reset_index(drop=True)

#%% look at range vec corrs for each acquisition

color_each_bout = True
likelihood_cdict = {3: 'r', 2: 'darkorange', 1: 'lightgray'}

for acq, acqdf_ in aggr_corrs[aggr_corrs['likelihood']>1].groupby(['acquisition']):

    fig, ax = pl.subplots()

    for ix, corrdf_ in acqdf_.groupby('boutnum'):
        imax = corrdf_['pearson_r'].argmax()
        if imax < corrdf_.shape[0]/4:
            imax = -1
        corr_ = corrdf_.iloc[0:imax].copy() #['pearson_r']
        
        overall_corr = round(corr_['overall_corr'].unique()[0], 2)
        curr_likelihood = corr_['likelihood'].unique()[0]
        corr_['sec'] = corr_['sec'] - corr_['sec'].iloc[-1]
        bnum = corr_['boutnum'].unique()[0]  
        if color_each_bout:
            ax.plot(corr_['sec'], corr_['pearson_r'], 
                label='{}, r={}'.format(bnum, overall_corr)) 
        else: # color by likelihood
            ax.plot(corr_['sec'], corr_['pearson_r'],
                    color=likelihood_cdict[curr_likelihood])
            
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', frameon=False)

    ax.set_title(acq, loc='left')

#%%
smooth = True
smooth_win = 10

c_list = []
fig, ax = pl.subplots()
for (acq, ix), corrdf_ in aggr_corrs[aggr_corrs['likelihood']>1].groupby(['acquisition', 'boutnum']):

    imax = corrdf_['pearson_r'].argmax()
    if imax < corrdf_.shape[0]/4:
        imax = -1
    corr_ = corrdf_.iloc[0:imax].copy() #['pearson_r']
    
    overall_corr = round(corr_['overall_corr'].unique()[0], 2)
    if smooth:
        corr_sm = corr_.copy()
        corr_sm['pearson_r'] = corr_['pearson_r'].rolling(smooth_win, 1).mean()
    else:
        corr_sm = corr_.copy()
    corr_sm['sec'] = corr_sm['sec'] - corr_sm['sec'].iloc[-1]
    ax.plot(corr_sm['sec'], 
            corr_sm['pearson_r'], color=[0.7]*3, linewidth=0.25) #label='{}, {}'.format(ix, overall_corr)) 

    c_list.append(corr_sm)
    #sns.scatterplot(data=corrdf_, ax=ax, x='sec', y='pearson_r', 
    #            hue='sec', palette='viridis', legend=0)

tmpcorr = pd.concat(c_list)

mean_corr = tmpcorr.groupby('sec')['pearson_r'].mean().reset_index()
mean_corr_smoothed = mean_corr.rolling(smooth_win, 1).mean()
ax.plot(mean_corr_smoothed['sec'], mean_corr_smoothed['pearson_r'], 
        linewidth=1, color=bg_color)
ax.axhline(0, color='gray', linestyle='--')

fig.text(0.1, 0.9, 'Corr. between angle diff and vector length = {:.2f}'.format(corr))
ax.set_ylim([-1, 1])
ax.set_xlim([-1, 0])
#%
fig, ax =pl.subplots()
sns.lineplot(data=tmpcorr, x='sec', y='pearson_r', ax=ax)
ax.axhline(0, color='gray', linestyle='--')


#%%

mean_overall_corr = aggr_corrs.groupby(['acquisition', 'boutnum'])['overall_corr']\
                              .mean().reset_index()

fig, ax = pl.subplots()
sns.histplot(mean_overall_corr, x='overall_corr', ax=ax) # bins=20)
ax.set_xlim([-1, 1])

#%%





#%
curr_acq = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'
boutdf = actions[actions['acquisition'] == curr_acq]
# Does turn size correlate with theta error?

ix = 7

pre_acc = 0.03
post_acc = 0
nframes_pre = int(np.ceil(pre_acc * fps))
nframes_post = int(np.ceil(post_acc * fps))

epoch_pre = 1*fps
epoch_post = 0

b_ = boutdf[boutdf['action']=='interception'].iloc[ix]
start_, end_ = b_[['start', 'end']] 

curr_frames = np.arange(start_ - epoch_pre, end_ + epoch_post)
plotdf = f1[f1['frame'].isin(curr_frames)].copy()
accel_start = get_acceleration_start(plotdf)

fig, axn =pl.subplots(2, 1, figsize=(8,8))
ax=axn[0]
ax.plot(plotdf['theta_error'])
ax=axn[1]
ax.plot(plotdf['turn_size_smoothed'])

for ax in axn:
    ax.axvline(accel_start, color='red', linestyle='--')


#%%
ang_int = {'interception_angle':[], 'targ_ang_vel': [],
            'targ_ang_vel_smoothed': []}
for i, b_ in boutdf[boutdf['action']=='interception'].iterrows():
    start_, end_ = b_[['start', 'end']]
    curr_frames = np.arange(start_ - epoch_pre, end_ + epoch_post)
    plotdf = f1[f1['frame'].isin(curr_frames)].copy()
    accel_start = get_acceleration_start(plotdf)

    pre_frame = accel_start - nframes_pre
    post_frame = accel_start + nframes_post 
    targ_ang_vel = f1.loc[pre_frame]['theta_error_dt']
    targ_ang_vel_sm = f1.loc[pre_frame]['theta_error_dt_smoothed']
    #interception_angle = f1.loc[post_frame]['abs_ang_between']
    interception_angle = f1.loc[post_frame]['traveling_dir']

    #if targ_ang_vel < -6:
    #    continue
    ang_int['interception_angle'].append(interception_angle)
    ang_int['targ_ang_vel'].append(targ_ang_vel)
    ang_int['targ_ang_vel_smoothed'].append(targ_ang_vel_sm)

#%
ang_int = pd.DataFrame(ang_int)




# %%
# =======================================================
# INTERCEPTION ANGLE?
# =======================================================
fps = 60
epoch_pre = 1*fps
epoch_post = 0

pre_acc = 0.03
post_acc = 0.4
ang_int = get_targ_ang_vel_and_int_angle(boutdf,f1,
                    epoch_pre=epoch_pre, epoch_post=epoch_post,
                    pre_acc=pre_acc, post_acc=post_acc)
ang_int
#%%
fig, ax = pl.subplots()

sns.regplot(data=ang_int, x='targ_ang_vel', y='interception_angle', ax=ax)  
ax.set_aspect(1)





#%%
#
# negative theta_error_dt means target is moving to fly's right (negative, CW)
# positive theta_error_dt means target is moving to fly's left (positive, CCW)

# Land & Collett, 1978
# The course angle Beta (Figure 6) is the angle between the course and
# the direction of first sighting. 
# This is distinct from the "angle through which the fly has to turn" (delta-phi)
# 
pre_acc = 0.03 #0.03
post_acc = 0.9
win = 5
#acq_prefix = 
acq_prefix = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'
boutdf = actions[(actions['action']=='interception') 
                 & (actions['acquisition']==acq_prefix)].copy() #roupby('acquisition'):

a_ = []
for acq_prefix, boutdf in actions[\
                        (actions['action']=='interception')
                        & (actions['likelihood']>1)].groupby('acquisition'):
    f1 = flydf[(flydf['acquisition']==acq_prefix) 
               & (flydf['id']==0)].copy()

    # get number of frames before and after acceleration start
    nframes_pre_acc = int(np.ceil(pre_acc * fps))
    nframes_post_acc = int(np.ceil(post_acc * fps))

    plot_vars = ['interception_angle', 'theta_error', 'theta_error_smoothed', 
                 'theta_error_dt', 'theta_error_dt_smoothed',
                 'los_at_sighting', 'beta', 'bearing_angle', 'bearing_angle_dt', 'ang_vel']
    
    ang_int = dict((k, []) for k in plot_vars)
    for i, b_ in boutdf[boutdf['action']=='interception'].iterrows():

        #b_ = boutdf[boutdf['action']=='interception'].iloc[i]
        start_, end_ = b_[['start', 'end']]
        curr_frames = np.arange(start_ - epoch_pre, end_ + epoch_post)
        plotdf = f1[f1['frame'].isin(curr_frames)].copy()
        accel_start = get_acceleration_start(plotdf)

        pre_frame = accel_start - nframes_pre_acc
        post_frame = start_ + nframes_post_acc

        theta_error = spstats.circmean(f1.loc[pre_frame-win:pre_frame+win]['theta_error'].dropna(),
                                       high=np.pi, low=-np.pi)
        ang_vel_fly = np.mean(f1.loc[pre_frame-win:pre_frame+win]['ang_vel'].dropna())

        theta_error_sm = spstats.circmean(f1.loc[pre_frame-win:pre_frame+win]['theta_error_smoothed'].dropna(),
                                          high=np.pi, low=-np.pi)

        los_at_sighting = spstats.circmean(f1.loc[pre_frame-win:pre_frame+win]['abs_ang_between'].dropna(),
                                          high=np.pi, low=-np.pi)
        bearing_angle_dt = np.nanmean( f1.loc[pre_frame-win:pre_frame+win]['abs_ang_between'].diff()/(1/fps) )

        
        theta_error_dt = f1.loc[pre_frame-win: pre_frame+win]['theta_error_dt'].mean()
        #theta_error_dt = f1.loc[pre_frame-win: pre_frame+win]['theta_error'].transform(lambda x: x.diff()/(1/fps)).mean()
        #theta_error_dt_sm = f1.loc[pre_frame-win: pre_frame+win]['theta_error_smoothed'].transform(lambda x: x.diff()/(1/fps)).mean()

        theta_error_dt_sm = f1.loc[pre_frame-win:pre_frame+win]['theta_error_dt_smoothed'].mean()

        #interception_angle = f1.loc[post_frame]['abs_ang_between']
        interception_angle = spstats.circmean(f1.loc[post_frame-win:post_frame+win]['traveling_dir'].dropna(),
                                              high=np.pi, low=-np.pi)

        #if targ_ang_vel < -6:
        #    continue
        ang_int['interception_angle'].append(interception_angle)
        ang_int['theta_error'].append(theta_error)
        ang_int['theta_error_smoothed'].append(theta_error_sm)
        ang_int['theta_error_dt'].append(theta_error_dt)
        ang_int['theta_error_dt_smoothed'].append(theta_error_dt_sm)

        ang_int['los_at_sighting'].append(los_at_sighting)
        ang_int['beta'].append( interception_angle - los_at_sighting )

        ang_int['bearing_angle'].append(los_at_sighting)
        ang_int['bearing_angle_dt'].append(bearing_angle_dt)
        ang_int['ang_vel'].append(ang_vel_fly)

        ang_int['boutnum'] = i
        ang_int['start_frame'] = start_
        ang_int['end_frame'] = end_
        ang_int['acquisition'] = acq_prefix
    #%
    ang_int = pd.DataFrame(ang_int)


    a_.append(ang_int)

all_ang_int = pd.concat(a_)

#%
for pvar in plot_vars:
    all_ang_int['{}_abs'.format(pvar)] = np.abs(all_ang_int[pvar])
# %
#plot_ = all_ang_int[(all_ang_int['targ_ang_vel_smoothed']>-5) 
#                   & (all_ang_int['targ_ang_vel_smoothed']<5)]

#%%
plot_ = all_ang_int.copy()

# xvars = ['theta_error_smoothed', 'theta_error_dt_smoothed']
xvars = ['theta_error', 'theta_error_dt', 'ang_vel']
yvar = 'beta' #'interception_angle'
yvar = 'bearing_angle_dt'
import libs.regplot as rpl

for xvar in xvars:
    fig, ax = pl.subplots()
    #sns.regplot(data=plot_, x=xvar, y=yvar, ax=ax, color=bg_color)

    sns.scatterplot(data=plot_, x=xvar, y=yvar, ax=ax, color=bg_color)
    res = rpl.regplot(data=plot_, ax=ax, x=xvar, y=yvar,
                color=bg_color, scatter=False) #, ax=ax)
    # res.params: [intercept, slope]
    ax.set_box_aspect(1)
    fit_str = 'OLS: y = {:.2f}x + {:.2f}'.format(res.params[1], res.params[0])
    print(fit_str) #lope, intercept)
    ax.text(0.05, 0.85, fit_str, fontsize=8, transform=ax.transAxes)

    ax.set_box_aspect(1)

    putil.annotate_regr(plot_, ax, x=xvar, y=yvar, fontsize=8)

#%%
#c_frames = np.arange(post_frame-win, post_frame)
c_frames = np.arange(pre_frame-win, pre_frame+win)

fig, ax = pl.subplots()
plot_heading_vs_travelingdir_frames(c_frames, f1, f2, ax=ax, cap=cap,
                                          var1='ori', var2='traveling_dir')                                       
ax.set_xticks([0, 200, 400, 600, 800])


# %%
pl.figure()
pl.plot(f1['theta_error_dt'])
pl.plot(f1['theta_error_dt_smoothed'])
# %%
pl.figure()
pl.plot(f1['theta_error'])
pl.plot(f1['theta_error_smoothed'])



# %%
fig, ax =pl.subplots(figsize=(10,5)) #subplot_kw={'projection': 'polar'})

for acq_prefix, boutdf in actions[actions['action']=='interception'].groupby('acquisition'):
    f1 = flydf[flydf['acquisition']==acq_prefix].copy()

    for i, b_ in boutdf[boutdf['action']=='interception'].iterrows():
        start_, end_ = b_[['start', 'end']]

        s = start_ - (0.5*fps)
        e = end_    
        range_vec_dirs = np.unwrap( f1.loc[s:e]['abs_ang_between'] )
        #range_vec_dirs = f1.loc[s:e]['abs_ang_between'] - f1.loc[s:e]['abs_ang_between'].min()
        range_vec_diffs = np.diff(range_vec_dirs)
        xvals = f1.loc[s+1:e]['frame'] - f1.loc[start_]['frame']
        # xvals = xvals[1:] 
        
        ax.plot( xvals, range_vec_diffs, color=bg_color, lw=0.5)
        ax.axvline(0, color=bg_color, linestyle=':')

ax.set_xlim([-20, 50])
# %%
fig, ax =pl.subplots() #subplot_kw={'projection': 'polar'})
ax.plot(range_vec_dirs)
# %%

ft = ftjaaba[ftjaaba['acquisition']==acq_prefix].copy()

#flydf = f1.copy() #df[df['id']==0].copy()

ft.shape, f1.shape

# combine ft and f1 dataframes based on index:
currdf = pd.merge(f1, ft[['chasing', 'courting', 'boutdur', 'stim_hz', 'good_frames']], left_index=True, right_index=True)

#%%
currdf['action'] = None
for i, b_ in boutdf.iterrows():
    # Assign action based on start and end frame
    start_, end_ = b_[['start', 'end']]
    incl_ = currdf[currdf['frame'].isin(np.arange(start_, end_+1))].copy()
    incl_['action'] = b_['action']
    currdf.loc[incl_.index, 'action'] = b_['action']


#%% =====================================================
# subdivide --------------------------------------------------------
# split into small bouts
# --------------------------------------------------------
bout_dur = 0.20
min_boutdur = 0.25
min_dist_to_other = 2
#%
filtdf = currdf[(currdf['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (currdf['dist_to_other']>=min_dist_to_other)
                & (currdf['boutdur']>=min_boutdur)
                #& (df['good_frames']==1)
                #& (df['led_level']>0)
                & (currdf['action'].notnull())
                ].copy() #.reset_index(drop=True)

# subdivide into smaller boutsa
# bout_dur = 0.5
filtdf = util.subdivide_into_bouts(filtdf, bout_dur=bout_dur)

#%%

# Get mean value of small bouts
if 'strain' in filtdf.columns:
    filtdf = filtdf.drop(columns=['strain'])

interceptions = filtdf[filtdf['action']=='interception'].copy()
pursuits = filtdf[filtdf['action']=='pursuit'].copy()

pursuits = pursuits.drop(columns=['action'])
interceptions = interceptions.drop(columns=['action'])
#%
meanbouts = pursuits.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
meanbouts.head()

#%%
cmap='viridis'
stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

# find the closest matching value to one of the keys in stimhz_palette:
meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))   

#%%
vmin = min(list(stimhz_palette.keys()))
vmax = max(list(stimhz_palette.keys()))
hue_var = 'stim_hz'
hue_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

markersize=5
huevar='stim_hz'
cmap='viridis'
plot_com=True

xvar= 'facing_angle'
yvar = 'targ_pos_radius'


p_ = filtdf[(filtdf['action']=='interception')
            | (filtdf['chasing']>0)].copy()

fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True,
                            subplot_kw={'projection': 'polar'})
pp.plot_allo_vs_egocentric_pos(p_, axn=axn, xvar=xvar, yvar=yvar, huevar=huevar,
                            palette_dict=stimhz_palette, hue_norm=hue_norm, markersize=5,
                            com_markersize=40, com_lw=1)
for ax in axn:
    yl = ax.get_yticklabels()
    ax.set_yticklabels([v if i%2==0 else '' for i, v in enumerate(yl)])
pl.subplots_adjust(wspace=0.6, top=0.8, right=0.8)

putil.label_figure(fig, figid)

fig.suptitle('{}: {} {}, where min fract of bout >= {:.2f}'.format(sp, behav, data_type, min_frac_bout))

figname = 'allo-v-ego-{}-{}_{}-v-{}_min-frac-bout-{}_{}'.format(behav, data_type, xvar, yvar, min_frac_bout, sp)
#pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))


# %%
