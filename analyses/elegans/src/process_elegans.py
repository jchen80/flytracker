#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 29, 16:04:00 2024

@filename: test_elegans.py
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

#from relative_metrics import load_processed_data
import libs.utils as util
import libs.plotting as putil
import transform_data.relative_metrics as rel

# %%

# Set plotting
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=24)
bg_color = [0.7]*3 if plot_style=='dark' else 'w'

#% plotting settings
curr_species = ['Dele', 'Dmau', 'Dmel', 'Dsant', 'Dyak']
species_cmap = sns.color_palette('colorblind', n_colors=len(curr_species))
print(curr_species)
species_palette = dict((sp, col) for sp, col in zip(curr_species, species_cmap))

# %%

assay = '2d-projector' # '38mm-dyad'
create_new = False

minerva_base = '/Volumes/Julie'

srcdir = os.path.join(minerva_base, '2d-projector-analysis/FlyTracker/processed_mats') #relative_metrics'

# local savedir for giant pkl
localdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/{}/FlyTracker'.format(assay)

# get local file for aggregated data
out_fpath_local = glob.glob(os.path.join(localdir, 'processed.pkl'))[0]
print("Loading processed data from:", out_fpath_local)

# set figdir
figdir = os.path.join(os.path.split(srcdir)[0], 'predictive_coding', 'theta_error')
if not os.path.exists(figdir):
    os.makedirs(figdir)
print("Saving figs to: ", figdir)

#% Set fig id
figid = srcdir  
# %%

df0 = pd.read_pickle(out_fpath_local)
df0.head()

df0['acquisition'] = ['_'.join([f for f in f.split('_')[0:-1] if 'led' not in f]) for f in df0['acquisition']]

# %%

# extract size of stimulus based on format _szNxN in filename using regex
#df0['stim_size'] = [int(f.split('_sz')[-1].split('x')[0]) for f in df0['filename']]

# tmp -- add species
#df0['species'] = 'Dele'

# 
# summary of what we've got
print(df0[['species', 'acquisition']].drop_duplicates().groupby('species').count())

# %%
d_list = []
for acq, df_ in df0.groupby('acquisition'):
    f1 = df_[df_['id']==0].copy().reset_index(drop=True)
    f2 = df_[df_['id']==1].copy().reset_index(drop=True)
    if 'sec_diff' not in f1.columns:
        f1['sec_diff'] = f1['sec'].diff()
        f2['sec_diff'] = f2['sec'].diff()
    # calculate theta error
    f1 = rel.calculate_theta_error(f1, f2)
    f1 = rel.calculate_theta_error_from_traveling_direction(f1, f2)
    f2 = rel.calculate_theta_error(f2, f1)
    f2 = rel.calculate_theta_error_from_traveling_direction(f2, f1)
    # add
    d_list.append(f1)
    d_list.append(f2)
df0 = pd.concat(d_list)

# %%
importlib.reload(the)
#%
df = rel.calculate_angle_metrics_focal_fly(df0, winsize=5)
df = util.shift_variables_by_lag(df, lag=2)

#%% JAABA
importlib.reload(util)

fname = 'projector_data_elegans_all_20240325_jaaba' #if assay=='38mm-dyad' else None
jaaba1 = util.load_jaaba(assay, fname=fname)
jaaba1.head()

fname2 = 'projector_data_mel_yak_20240330_jaaba'
jaaba2 = util.load_jaaba(assay, fname=fname2)

if 'clacking' in jaaba2.columns:
    jaaba2 = jaaba2.drop(columns=['clacking'])
if 'led_level' not in jaaba1.columns:
    jaaba1['led_level'] = 1

print(jaaba1.shape, jaaba2.shape)

#%%
jaaba = pd.concat([jaaba1, jaaba2], axis=0)
jaaba.shape

#%%
# merge jaaba --------------------------------------------------------
ftjaaba = util.combine_jaaba_and_processed_df(df, jaaba)

#%%
#% Rename variables
if 'courtship' in ftjaaba.columns:
    ftjaaba = ftjaaba.rename(columns={'courtship': 'courting'})

#% Add bouts
if 'fpath' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['fpath'])
if 'name' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['name'])

#% Split into bouts of courtship
d_list = []
for acq, df_ in ftjaaba.groupby('acquisition'):
    df_ = df_.reset_index(drop=True)
    df_ = util.mat_split_courtship_bouts(df_, bout_marker='courting')
    dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
                    timevar='sec')
    d_list.append(df_.merge(dur_, on=['boutnum']))
ftjaaba = pd.concat(d_list)


if 'good_frames' not in ftjaaba.columns:
    ftjaaba['good_frames'] = 1
#%
if 'filename' in ftjaaba.columns:
    ftjaaba = ftjaaba.drop(columns=['filename'])

if 'dovas' in ftjaaba.columns:
    ftjaaba['dovas_deg'] = np.rad2deg(ftjaaba['dovas'])

#%% get means by BOUT
groupcols = [ 'species', 'acquisition', 'boutnum']

min_pos_theta = np.deg2rad(-160)
max_pos_theta = np.deg2rad(160)
min_dist_to_other = 2

#%% ------------------------------
# JAABA THRESHOLDS
# ------------------------------
if ftjaaba['chasing'].max() == 1:
    jaaba_thresh_dict = {'orienting': 0, 
                        'chasing': 0,
                        'singing': 0}
else:
    jaaba_thresh_dict = {'orienting': 10,
                        'chasing': 10,
                        'singing': 5}
print(jaaba_thresh_dict)

#%%
# ftjaaba['ang_vel_abs'] = np.abs(ftjaaba['ang_vel'])
ftjaaba = util.shift_variables_by_lag(ftjaaba, lag=2)

#    ftjaaba['ang_vel_deg'] = np.rad2deg(ftjaaba['ang_vel'])
#    ftjaaba['ang_vel_abs_shifted'] = ftjaaba.groupby('acquisition')['ang_vel_abs'].shift(-2)
#    ftjaaba['ang_vel_fly_shifted'] = ftjaaba.groupby('acquisition')['ang_vel_fly'].shift(-2)
#
#    ftjaaba['vel_shifted'] = ftjaaba.groupby('acquisition')['vel'].shift(-2)
#    ftjaaba['vel_shifted_abs'] = np.abs(ftjaaba['vel_shifted']) 
#
#    ftjaaba['ang_vel_shifted'] = ftjaaba.groupby('acquisition')['ang_vel'].shift(-2)

# binarize behavs
ftjaaba = util.binarize_behaviors(ftjaaba, jaaba_thresh_dict=jaaba_thresh_dict)
#%%


# Save
outfile = os.path.join(localdir, 'ftjaaba.pkl')
ftjaaba.to_pickle(outfile)



#%%
# tmp

ftjaaba.loc[ftjaaba['species']=='Dele', 'led_level'] = 1

#%%

#ftjaaba['ang_vel_fly_shifted_abs'] = np.abs(ftjaaba['ang_vel_fly_shifted'])

#%% =====================================================
# subdivide --------------------------------------------------------
# split into small bouts
# --------------------------------------------------------
bout_dur = 0.20
min_boutdur = 0.25
min_dist_to_other = 2
#%
filtdf = ftjaaba[(ftjaaba['id']==0)
                #& (ftjaaba['targ_pos_theta']>=min_pos_theta) 
                #& (ftjaaba['targ_pos_theta']<=max_pos_theta)
                & (ftjaaba['dist_to_other']>=min_dist_to_other)
                & (ftjaaba['boutdur']>=min_boutdur)
                & (ftjaaba['good_frames']==1)
                & (ftjaaba['led_level']>0)
                ].copy() #.reset_index(drop=True)

#%%
importlib.reload(util)

# subdivide into smaller boutsa
# bout_dur = 0.5
filtdf = util.subdivide_into_bouts(filtdf, bout_dur=bout_dur)

#%%

# Get mean value of small bouts
if 'strain' in filtdf.columns:
    filtdf = filtdf.drop(columns=['strain'])
#%
meanbouts = filtdf.groupby(['species', 'acquisition', 'boutnum']).mean().reset_index()
meanbouts.head()

cmap='viridis'
stimhz_palette = putil.get_palette_dict(ftjaaba[ftjaaba['stim_hz']>=0], 'stim_hz', cmap=cmap)

# find the closest matching value to one of the keys in stimhz_palette:
meanbouts['stim_hz'] = meanbouts['stim_hz'].apply(lambda x: min(stimhz_palette.keys(), key=lambda y:abs(y-x)))   

#%% ------------------------------------------------
# ANG_VEL vs. THETA_ERROR
# -------------------------------------------------
#
plotdf = meanbouts.copy() #[meanbouts['species']=='Dele'].copy()
plotdf.head()

#%%
import libs.regplot as rpl

#%%
xvar = 'theta_error'
yvar = 'ang_vel_fly_shifted'
plot_hue= False
plot_grid = True

behav='chasing'
min_frac_bout = 0.5


nframes_lag = 2
n_species = plotdf['species'].nunique()


if 'shifted' in yvar:
    figtitle = '{} bouts, where min fract of bout >= {:.2f}\nshifted {} frames'.format(behav, min_frac_bout, nframes_lag)
else:
    figtitle = '{} bouts, where min fract of bout >= {:.2f}'.format(behav, min_frac_bout)

shift_str = 'SHIFT-{}frames_'.format(nframes_lag) if 'shifted' in yvar else ''
hue_str = 'stimhz' if plot_hue else 'no-hue'

chase_ = plotdf[plotdf['{}_binary'.format(behav)]>min_frac_bout].copy()

# plot
fig, axn = pl.subplots(1, n_species, sharex=True, sharey=True)
for ai, (sp, df_) in enumerate(chase_[chase_['stim_hz']>0].groupby('species')):
    ax = axn[ai]
    # plot scatter
    if plot_hue:
        sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax,
                    hue='stim_hz', palette=stimhz_palette, 
                    legend=0, edgecolor='none', alpha=0.7)
    else:
        sns.scatterplot(data=df_, x=xvar, y=yvar, ax=ax, color=bg_color,
                    legend=0, edgecolor='none', alpha=0.7, s=2)

    if plot_grid:
        ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
        ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)

    ax.set_title(sp)
    # do fit
    res = rpl.regplot(data=df_, ax=ax, x=xvar, y=yvar, 
                color='w', scatter=False) #, ax=ax)

    ax.set_ylim([-30, 30])
    # res.params: [intercept, slope]
    fit_str = 'OLS: y = {:.2f}x + {:.2f}'.format(res.params[1], res.params[0])
    #print() #lope, intercept)
    ax.text(0.05, 0.85, fit_str, fontsize=8, transform=ax.transAxes)
    ax.set_box_aspect(1)
    
    # set xlabel to be theta subscript E
    ax.set_xlabel(r'$\theta_{E}$')
    ax.set_ylabel('$\omega_{f}$ (deg/s)')
    #annotate_regr(df_, ax, x=xvar, y=yvar, fontsize=8)

fig.suptitle(figtitle, fontsize=12)

putil.label_figure(fig, figid)
figname = 'sct_{}_v_{}_{}{}_min-frac-bout-{}'.format(yvar, xvar, shift_str, hue_str, min_frac_bout)
pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))
pl.savefig(os.path.join(figdir, '{}.svg'.format(figname)))

print(figdir, figname)


# %%
    #%% 
    # ------------------------------------------------
    # Compare small vs. large theta -error
    # -------------------------------------------------

    err_palette={'small': 'r', 'large': 'b'}
    min_frac_bout = 0.
    use_bouts = True
    theta_error_small = np.deg2rad(10)
    theta_error_large = np.deg2rad(25)
    curr_species = 'Dele'

    var1 = 'vel_shifted'
    var2 = 'ang_vel_shifted'

    nframes_lag_plot = 2 if 'shifted' in var1 or 'shifted' in var2 else 0

    if use_bouts:
        chase_ = meanbouts[(meanbouts['{}_binary'.format(behav)]>min_frac_bout)
                    & (meanbouts['species']==curr_species)].copy().reset_index(drop=True)
    else:
        chase_ = filtdf[(filtdf['{}_binary'.format(behav)]>min_frac_bout)
                    & (filtdf['species']==curr_species)].copy().reset_index(drop=True)

    chase_['error_size'] = None
    chase_.loc[(chase_['theta_error'] < theta_error_small) \
            & (chase_['theta_error'] > -theta_error_small), 'error_size'] = 'small'
    chase_.loc[(chase_['theta_error'] > theta_error_large) \
            | (chase_['theta_error'] < -theta_error_large), 'error_size'] = 'large'

    # plot ------------------------------------------------
    fig, axn = pl.subplots(1, 2, figsize=(7,4))
    fig.text(0.1, 0.95, '{} bouts, frac. of bout > {:.1f}, lag {} frames'.format(behav, min_frac_bout, nframes_lag_plot), fontsize=12)

    ax=axn[0]
    sns.histplot(data=chase_, x='theta_error', ax=ax, color=bg_color, bins=50)
    for thetalim, col in zip([theta_error_small, theta_error_large], ['r', 'b']):
        ax.axvline(x=thetalim, color=col, linestyle='--')
        ax.axvline(x=-thetalim, color=col, linestyle='--')
    ax.set_box_aspect(1)

    ax=axn[1]
    sns.scatterplot(data=chase_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
                    hue='error_size', palette=err_palette, s=2)
    ax.set_aspect(1)
    ax.plot(0, 0, 'wo', markersize=3)
    ax.axis('off')
    sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)

    fig.text(0.1, 0.9, curr_species, fontsize=24)
    pl.subplots_adjust(wspace=0.4)

    figname = 'hist_big-small-theta-error_{}_{}_{}_min-frac-bout-{}'.format(curr_species, theta_error_small, theta_error_large, min_frac_bout)
    #pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

    #% plot HISTS ------------------------------------------------
    fig, axn =pl.subplots(1, 2, figsize=(7, 4), sharey=True)
    fig.text(0.1, 0.95, '{} bouts, frac. of bout > {:.1f}, lag {} frames'.format(behav, min_frac_bout, nframes_lag_plot), fontsize=12)

    ax=axn[0]
    sns.histplot(data=chase_, x=var1,  ax=ax, bins=50, 
                stat='probability', cumulative=False, element='step', fill=False,
                hue='error_size', palette=err_palette, common_norm=False, legend=0)
    ax.set_xlabel('forward vel')
    ax=axn[1]
    sns.histplot(data=chase_, x=var2, ax=ax, color='r', bins=50, 
                stat='probability', cumulative=False, element='step', fill=False,
                hue='error_size', palette=err_palette, common_norm=False)
    ax.set_xlabel('angular vel')
    sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', frameon=False)

    curr_ylim = np.round(axn[1].get_ylim()[-1], 2)*1.15
    for v, ax in zip([var1, var2], [axn[0], axn[1]]):
        med_ = chase_.groupby('error_size')[v].median()
        for mval, cval in err_palette.items():
            ax.plot(med_[mval], curr_ylim, color=cval, marker='v', markersize=10) #ax.axvline(x=m, color=c, linestyle='--')
        #print(med_)
        ax.set_box_aspect(1)
    pl.subplots_adjust(wspace=0.3)

    fig.text(0.1, 0.9, curr_species, fontsize=24)

    #figname = 'hist_big-small-theta-error_ang-v-fwd-vel_{}_{}_{}_min-frac-bout-{}'.format(curr_species, theta_error_small, theta_error_large, min_frac_bout)
    #pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))



 
# %%
    #%% ---------------------------------------------
    # PLOT ALLO vs. EGO

    import parallel_pursuit as pp
    importlib.reload(pp)

    behav = 'chasing'
    min_frac_bout = 0.2
    do_bouts=False

    markersize=5
    huevar='stim_hz'
    cmap='viridis'
    plot_com=True

    xvar= 'facing_angle'
    yvar = 'targ_pos_radius'

    is_flytracker=True
    sign = -1 if is_flytracker else 1

    chase_ = meanbouts[meanbouts['{}_binary'.format(behav)]>=min_frac_bout].copy()

    data_type = 'BOUTS' if do_bouts else 'FRAMES' 
    if not do_bouts:
        # plot frames
        palette_dict = putil.get_palette_dict(filtdf[filtdf['stim_hz']>0], huevar, cmap='viridis')
        plotdf = filtdf[( filtdf['chasing']>jaaba_thresh_dict['chasing'])
                & ( filtdf['stim_hz']>0 )].copy()
    else:
        palette_dict = stimhz_palette
        plotdf = chase_[chase_['stim_hz']>0].copy()

    plotdf['facing_angle'] = sign * plotdf['facing_angle']


    vmin = min(list(palette_dict.keys()))
    vmax = max(list(palette_dict.keys()))
    hue_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for sp, p_ in plotdf.groupby('species'):
        fig, axn = pl.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True,
                                    subplot_kw={'projection': 'polar'})
        pp.plot_allo_vs_egocentric_pos(p_, axn=axn, xvar=xvar, yvar=yvar, huevar=huevar,
                                    palette_dict=palette_dict, hue_norm=hue_norm, markersize=5,
                                    com_markersize=40, com_lw=1)
        for ax in axn:
            yl = ax.get_yticklabels()
            ax.set_yticklabels([v if i%2==0 else '' for i, v in enumerate(yl)])
        pl.subplots_adjust(wspace=0.6, top=0.8, right=0.8)

        putil.label_figure(fig, figid)

        fig.suptitle('{}: {} {}, where min fract of bout >= {:.2f}'.format(sp, behav, data_type, min_frac_bout))

        figname = 'allo-v-ego-{}-{}_{}-v-{}_min-frac-bout-{}_{}'.format(behav, data_type, xvar, yvar, min_frac_bout, sp)
        #pl.savefig(os.path.join(figdir, '{}.png'.format(figname)))

        print(figdir, figname)


# %%

#%% ------------------------------------------
# PURSUIT v. INTERCEPTIONS
# --------------------------------------------
# %% Get manually annotated actions -- annoted with FlyTracker
acq = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'

cap, viddir = util.get_video_cap_check_multidir(acq, assay=assay, return_viddir=True)
# get path to actions file for current acquisition
#viddir = acqdir
action_fpaths = glob.glob(os.path.join(viddir, 'fly-tracker', '{}*'.format(acq), '*actions.mat'))
action_fpath = action_fpaths[0]
print(action_fpath)

# load actions to df
boutdf = util.ft_actions_to_bout_df(action_fpath)