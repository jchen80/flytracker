#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:47:00 2020
@author: julianarhee
@email: juliana.rhee@gmail.com  
"""
#%%
import sys
import os
import glob
import cv2
import numpy as np
import pandas as pd
import pylab as pl  
import seaborn as sns
import matplotlib as mpl
import pickle as pkl
import argparse

import libs.utils as util


#%%
def plot_frame_check_affines(ix, fly1, fly2, cap, frame_width=None, frame_height=None):
    '''
    Plot frame and rotations with markers oriented to fly's heading. IX is FRAME NUMBER.

    Arguments:
        ix -- _description_
        fly1 -- _description_
        fly2 -- _description_
        cap -- _description_

    Keyword Arguments:
        frame_width -- _description_ (default: {None})
        frame_height -- _description_ (default: {None})

    Returns:
        _description_
    '''
    if frame_width is None:
        frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # set fly oris as arrows
    fly_marker = '$\u2192$' # https://en.wikipedia.org/wiki/Template:Unicode_chart_Arrows
    m_ori = np.rad2deg(fly1[fly1['frame']==ix]['rot_ori'])
    f_ori = np.rad2deg(fly2[fly2['frame']==ix]['rot_ori'])
    marker_m = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_m._transform = marker_m.get_transform().rotate_deg(m_ori)
    marker_f = mpl.markers.MarkerStyle(marker=fly_marker)
    marker_f._transform = marker_f.get_transform().rotate_deg(f_ori)
    #print(np.rad2deg(fly1.loc[ix]['ori'])) #m_ori)
    #print(f_ori)

    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    fig = pl.figure(figsize=(8,4))
    ax = fig.add_subplot(121) # axn = pl.subplots(1, 2)
    ax.imshow(im, cmap='gray')
    ax.set_title("Frame {}".format(ix), fontsize=8, loc='left')
    ax.invert_yaxis()

    #ax = fig.add_subplot(142) 
    ax.plot(fly1[fly1['frame']==ix]['pos_x'], fly1[fly1['frame']==ix]['pos_y'], 'r*')
    ax.plot(fly2[fly2['frame']==ix]['pos_x'], fly2[fly2['frame']==ix]['pos_y'], 'bo')
    ax.set_aspect(1)
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    #ax.invert_yaxis()

    ax = fig.add_subplot(122)
    ax.set_title('centered and rotated to focal (*)', fontsize=8, loc='left') 
    # make a markerstyle class instance and modify its transform prop
    ax.plot([0, float(fly1[fly1['frame']==ix]['rot_x'].iloc[0])], 
            [0, float(fly1[fly1['frame']==ix]['rot_y'].iloc[0])], 'r', 
            marker=marker_m, markerfacecolor='r', markersize=10) 
    ax.plot([fly2[fly2['frame']==ix]['rot_x']], [fly2[fly2['frame']==ix]['rot_y']], 'b',
            marker=marker_f, markerfacecolor='b', markersize=10) 
    ax.set_aspect(1)
    ax.set_xlim(0-frame_width, frame_width)
    ax.set_ylim(0-frame_height, frame_height)
    #ax.invert_yaxis()

    return fig

def check_rotation_transform(ix, trk_, cap, id_colors=['r', 'b']):
    '''
    Note that ix should be frame.

    Arguments:
        ix -- _description_
        trk_ -- _description_
        cap -- _description_

    Keyword Arguments:
        id_colors -- _description_ (default: {['r', 'b']})

    Returns:
        _description_
    '''
    # get image
    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    fig = pl.figure(figsize=(12,5)) #pl.subplots(1, 2)
    # plot frame
    ax = fig.add_subplot(131)
    ax.imshow(im, cmap='gray')
    ax.invert_yaxis()
    # plot positions
    for i, d_ in trk_.groupby('id'):
        print('pos:', i, d_[d_['frame']==ix]['pos_x'], d_[d_['frame']==ix]['pos_y'])
        ax.plot(d_[d_['frame']==ix]['pos_x'], d_[d_['frame']==ix]['pos_y'], 
                marker='o', color=id_colors[i], markersize=3)

    fly1 = trk_[trk_['id']==0].copy().reset_index(drop=True)
    fly2 = trk_[trk_['id']==1].copy().reset_index(drop=True)
    # plot rotated positions, male faces EAST on cartesian
    ax = fig.add_subplot(132) #, projection='polar')
    for i, d_ in enumerate([fly1, fly2]):
        #print('rot:', i, d_.iloc[ix]['rot_x'], d_.iloc[ix]['rot_y'])
        pt = np.squeeze(np.array(d_[d_['frame']==ix][['trans_x', 'trans_y']].values))

        print(pt.shape)
        #ang = rotation_angs[ix]        
        #rx, ry = rotate([0, 0], pt, ang)
        ang = -1*fly1[fly1['frame']==ix]['ori'] 
        rotmat = np.array([[np.cos(ang), -np.sin(ang)],
                            [np.sin(ang), np.cos(ang)]])
        #rx, ry = (rotmat @ pt.T).T
        rx, ry = util.rotate_point(pt, ang) #[0, 0], pt, ang)
        print('rot:', i, rx, ry)
        ax.plot(rx, ry,marker='o', color=id_colors[i], markersize=3)
    ax.set_aspect(1)
    ax.set_title('rot')

    # POLAR
    ax = fig.add_subplot(133, projection='polar')
    for i, d_ in enumerate([fly1, fly2]):
        if i==0:
            ax.plot(0, 0, 'r*')
        #ang = fly1.iloc[ix]['ori'] #* -1
        pt = [d_[d_['frame']==ix]['trans_x'], 
              d_[d_['frame']==ix]['trans_y']]
        #ang = rotation_angs[ix]  
        #rx, ry = rotate((0,0), pt, ang)      
        #rx, ry = rotate2(pt, ang) #[0, 0], pt, ang)
        rad, th = util.cart2pol(rx, ry)
        ax.plot(th, rad, marker='o', color=id_colors[i], markersize=3)
    ax.set_aspect(1)
    ax.set_title('polar')
    #ax.set_theta_direction(-1)
    #print(ang)
    fig.suptitle('{}, ang={:.2f}'.format(ix, np.rad2deg(float(ang))))

    return fig




def plot_frame_target_projection(ix, fly1, fly2, cap, xvar='pos_x', yvar='pos_y'):
    cap.set(1, ix)
    ret, im = cap.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)

    # get vector between male and female
    xi = fly2.loc[ix][xvar] - fly1.loc[ix][xvar] 
    yi = fly2.loc[ix][yvar] - fly1.loc[ix][yvar]

    # get vector orthogonal to male's vector to female
    ortho_ = [yi, -xi] #ortho_hat = ortho_ / np.linalg.norm(ortho_)

    # project female heading vec onto orthog. vec
    f_ori = fly2.loc[ix]['ori']
    f_len = fly2.loc[ix]['major_axis_len']
    fem_vec = util.get_heading_vector(f_ori, f_len) #np.array([x_, y_])
    #female_hat = fem_vec / np.linalg.norm(fem_vec)
    vproj_ = util.proj_a_onto_b(fem_vec, ortho_)

    # plot
    fig, axn =pl.subplots(1, 2, figsize=(8, 4))
    ax = axn[0]
    ax.imshow(im, cmap='gray')
    ax.set_title("Frame {}".format(ix)) 
    # plot original positions
    x0, y0 = fly1.loc[ix][[xvar, yvar]]
    x1, y1 = fly2.loc[ix][[xvar, yvar]]
    # plot vector between
    ax.plot([x0, x0+xi], [y0, y0+yi])
    # plot orthogonal
    ax.plot([x1, x1+ortho_[0]], [y1, y1+ortho_[1]], 'orange')
    ax.set_aspect(1)
    # plot female heading
    ax.plot([x1, x1+fem_vec[0]], [y1, y1+fem_vec[1]], 'magenta')
    # plot proj
    ax.plot([x1, x1+vproj_[0]], [y1, y1+vproj_[1]], 'cyan')
    ax.plot([x1, x1-vproj_[0]], [y1, y1-vproj_[1]], 'cyan')

    # plot the vectors only
    ax=axn[1]
    ax.plot([0, xi], [0, yi], 'b')
    ax.plot([0, fem_vec[0]], [0, fem_vec[1]], 'magenta')
    ax.plot([0, ortho_[0]], [0, ortho_[1]], 'orange')
    ax.invert_yaxis()
    ax.plot([0, vproj_[0]], [0, vproj_[1]], 'cyan')
    #ax.plot([0, proj_[0]], [0, proj_[1]], 'magenta')
    ax.set_aspect(1)

    # check
    #diff = (vproj_ - np.array([fem_vec[0], fem_vec[1]]))
    #print(np.dot(diff, ortho_)) #(diff[0] * ortho_hat[0] ) + (diff[1] * ortho_hat[1])

    #ppm = calib_['PPM']
    #print(np.sqrt(xi**2 + yi**2)/ppm)
    #print(feat_.loc[ix]['dist_to_other']) 

    # len of projected female
    fem_sz = np.sqrt(vproj_[0]**2 + vproj_[1]**2) * 2
    dist_to_other = np.sqrt(xi**2 + yi**2)
    fem_sz_deg = 2*np.arctan(fem_sz/(2*dist_to_other))
    ax.set_title('Targ is {:.2f} deg. vis. ang'.format(np.rad2deg(fem_sz_deg)))

    return fig

def get_target_sizes(fly1, fly2, xvar='pos_x', yvar='pos_y'):
    '''
    For provided df of tracks (FlyTracker), calculates the size of target in deg. 

    Arguments:
        fly1 -- df of tracks.mat for fly1 (male or focal fly) 
        fly2 -- df of tracks.mat for fly2 (female or target fly)

        assumes that fly2 will have columns 'rel_ori', 'major_axis_len', and 'minor_axis_len' (if not, will need to add these with add_pairwise_metrics)

    Keyword Arguments:
        xvar -- position var to use for calculating vectors (default: {'pos_x'})
        yvar -- same as xvar (default: {'pos_y'})

    Returns:
        fly1 -- returns fly1 with new columns 'targ_ang_size' and 'targ_ang_size_deg'
    '''
    assert 'rel_ori' in fly2.columns, "fly2 must have column 'rel_ori' for target orientation"
    assert 'major_axis_len' in fly2.columns, "fly2 must have column 'major_axis_len' for target length"
    assert 'minor_axis_len' in fly2.columns, "fly2 must have column 'minor_axis_len' for target length"
    target_sizes = []
    for ix in fly1.index.tolist():
        diff_xi = fly2.loc[ix][xvar] - fly1.loc[ix][xvar] 
        diff_yi = fly2.loc[ix][yvar] - fly1.loc[ix][yvar]
        target_ori = fly2.loc[ix]['ori'] # use absolute orientation here 
        target_len_maj = fly2.loc[ix]['major_axis_len']
        target_len_min = fly2.loc[ix]['minor_axis_len']

        # take into account major/minor axes of ellipse
        target_sz_deg_maj = util.calculate_target_size_deg(diff_xi, diff_yi, target_ori, target_len_maj)
        target_sz_deg_min = util.calculate_target_size_deg(diff_xi, diff_yi, target_ori, target_len_min)

        target_sz_deg = np.max([target_sz_deg_maj, target_sz_deg_min])
        target_sizes.append(target_sz_deg)

    fly1['targ_ang_size'] = target_sizes
    fly1['targ_ang_size_deg'] = np.rad2deg(fly1['targ_ang_size'])
    return fly1


def get_relative_velocity(df_, win=1, 
                          value_var='dist_to_other', time_var='sec'):
    '''
    Calculate relative velocity between two flies, relative metric (one fly).
    If using FlyTracker feat.mat, dist_to_other is in mm, and time is sec.

    Arguments:
        fly1 -- feat_ dataframe for fly1

    Keyword Argumentsprint(figdir, figname)

:
        value_var -- relative dist variable to calculate position diff (default: {'dist_to_other'})
        time_var -- time variable to calculate time diff (default: {'sec'})
    '''
    # fill nan of 1st value with 0
    df_['{}_diff'.format(value_var)] = df_[value_var].interpolate().diff().fillna(0)# if dist incr, will be pos, if distance decr, will be neg
    df_['{}_diff'.format(time_var)] = df_[time_var].interpolate().diff().fillna(0) # if dist incr, will be pos, if distance decr, will be neg

    df_['rel_vel'] = df_['{}_diff'.format(value_var)] / (win*df_['{}_diff'.format(time_var)].mean())
    df_['rel_vel_abs'] = df_['{}_diff'.format(value_var)].abs() / (win*df_['{}_diff'.format(time_var)].mean())

    return df_


#%%
def get_copulation_ix(acq, 
    local_dir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/meta'):
    #local_dir='/Users/julianarhee/Documents/rutalab/projects/courtship/data/38mm-dyad/MF'):
    cop_ele = {
        '20231213-1103_fly1_eleWT_5do_sh_eleWT_5do_gh': 52267,
        '20231213-1154_fly3_eleWT_6do_sh_eleWT_5do_gh': 17243,
        '20231214-1051_fly2_eleWT_3do_sh_eleWT_3do_gh': 61512,
        '20231223-1117_fly1_eleWT_5do_sh_eleWT_5do_gh': 55582,
        '20231226-1137_fly2_eleWT_4do_sh_eleWT_4do_gh': 13740,
        '20240105-1007_fly1_eleWT_3do_sh_eleWT_3do_gh': 5051, 
        '20240109-1039_fly1_eleWT_4do_sh_eleWT_4do_gh': 177100,
        '20240322-1001_f1_eleWT_4do_gh': 5474,
        '20240322-1045_f4_eleWT_4do_gh': 54842,
        '20240322-1143_f6_eelWT_4do_gh': 1356,
        '20240322-1146_f7_eleWT_4do_gh': 12190,
        '20240322-1152_f8_eleWT_4do_gh': 7620,
        '20240322-1156_f9_eleWT_4do_gh': 92680
    }

    fname = 'courtship-free-behavior-GG (Responses) - Form Responses 1.csv'
    meta_fpath = os.path.join(local_dir, fname)
    meta = pd.read_csv(meta_fpath)

    if 'ele' in acq:
        match_ = [v for v in cop_ele.keys() if v.startswith(acq)]
        if len(match_)==0:
            print("No match: {}".format(acq))
            cop_ix = np.nan
        else:
            cop_ix = cop_ele[match_[0]]
    else:
        match_ = [v for v in meta['logfile'] if v.startswith(acq)]
        if len(match_)==0: #,  "{} not found".format(acq)
            print("NO match: {}".format(acq))
            cop_ix = np.nan
        else:
            #cop_ix = meta[meta['logfile']==match_[0]]['FlyTracker: copulation index']
            cop_ix = float(meta.loc[meta['logfile']==match_[0], 'FlyTracker: copulation index'])

    return cop_ix

def get_video_cap(acqdir, ftname=None, movie_fmt='avi', verbose=False):
    if ftname is None:
        vids = util.get_videos(acqdir, vid_type=movie_fmt)
    else:
        vids = util.get_video_by_ft_name(viddir, ftname, vid_type=movie_fmt)

    alt_movie_fmt = 'mp4' if movie_fmt=='avi' else 'avi'

    # try alt movie fmt
    try:
        assert len(vids)>0, "Found no video in directory: {}".format(vids)
        vids = [vids[-1]]
    except AssertionError as e:
        if ftname is None:
            vids = util.get_videos(acqdir, vid_type=alt_movie_fmt)
        else:
            vids = util.get_video_by_ft_name(acqdir, ftname, vid_type=alt_movie_fmt)
        #vids = util.get_videos(acqdir, vid_type=alt_movie_fmt)
        assert len(vids)==1, "Found more than one video in directory: {}".format(vids)  

    vidpath = vids[0]
    if verbose:
        print("Found video:", vidpath)

    cap = cv2.VideoCapture(vidpath)
    return cap

#% Calculation functions
def calculate_theta_error(f1, f2, xvar='pos_x', yvar='pos_y', fps=60.):
    vec_between = f2[[xvar, yvar]] - f1[[xvar, yvar]]
    abs_ang = np.arctan2(vec_between[yvar], vec_between[xvar])
    th_err = util.circular_distance(abs_ang, f1['ori']) # already bw -np.pi, pi

    f1['abs_ang_between'] = abs_ang # this is the line-of-sigh, line btween pursuer and target
    f1['theta_error'] = th_err
    
    f1['theta_error_dt'] = pd.Series(np.unwrap(f1['theta_error'].interpolate().ffill().bfill())).diff() / f1['sec'].diff().mean()
    f1['theta_error_deg'] = np.rad2deg(f1['theta_error'])

    return f1

def calculate_theta_error_from_traveling_direction(f1, f2, xvar='pos_x', yvar='pos_y'):
    if 'traveling' not in f1.columns:
        f1 = calculate_traveling_direction(f1)
        f2 = calculate_traveling_direction(f2)

    vec_between = f2[[xvar, yvar]] - f1[[xvar, yvar]]
    abs_ang = np.arctan2(vec_between[yvar], vec_between[xvar])
    th_err = util.circular_distance(abs_ang, f1['traveling']) # already bw -np.pi, pi

    f1['theta_error_traveling'] = th_err
    f1['theta_error_traveling_dt'] = pd.Series(np.unwrap(f1['theta_error'].interpolate().ffill().bfill())).diff() / f1['sec'].diff().mean()
    f1['theta_error_traveling_deg'] = np.rad2deg(f1['theta_error'])

    return f1

def calculate_traveling_direction(df_, winsize=5):
    #% smooth x, y, 
    df_['pos_x_smoothed'] = df_.groupby('id')['pos_x'].transform(lambda x: x.rolling(winsize, 1).mean())
    df_['pos_y_smoothed'] = 1*df_.groupby('id')['pos_y'].transform(lambda x: x.rolling(winsize, 1).mean())  # for FlyTracker, flipud

    # calculate traveling direction
    df_['traveling'] = np.arctan2(df_['pos_y_smoothed'].diff(), df_['pos_x_smoothed'].diff())
    df_['traveling_deg'] = np.rad2deg(df_['traveling'])

    return df_

def additional_angle_metrics_single(df_, winsize=5, fps=60, filter_funky=True):
    '''
    Calculate additional angle metrics for a single dataframe of a focal fly.
    Arguments:
        df_ -- dataframe with focal fly data
        winsize -- smoothing window size (default: 5)
        fps -- frames per second (default: 60)
        filter_funky -- whether to filter out funky values (default: True)
    '''
    # Calculate target angular vel
    df_ = util.smooth_and_calculate_velocity_circvar(df_, 
                            smooth_var='targ_pos_theta', vel_var='targ_ang_vel',
                            time_var='sec', winsize=winsize)

    # Calculate facing angle vel
    df_ = util.smooth_and_calculate_velocity_circvar(df_, 
                            smooth_var='facing_angle', vel_var='facing_angle_vel',)

    # Calculate "theta-error"
    # Pass a copy so the bad_ixs wipe inside smooth_and_calculate_velocity_circvar
    # cannot corrupt ori/sec and other source columns in df_ when theta_error is all NaN.
    try:
        df_tmp = util.smooth_and_calculate_velocity_circvar(df_.copy(), 
                                smooth_var='theta_error', vel_var='theta_error_dt', 
                                time_var='sec', winsize=winsize)
        #assert all(np.isnan(df_tmp['theta_error']))==False, "theta error is nan"
        df_['theta_error_dt'] = df_tmp['theta_error_dt']
        df_['theta_error_smoothed'] = df_tmp['theta_error_smoothed']
    #except Exception as e:
        #print("AssertionError: {}".format(e))
        #df_['theta_error_dt'] = np.nan
    except Exception as e:
        print("Error calculating theta error: {}".format(e))
        df_['theta_error_dt'] = np.nan

    # TMP splitiom to get vel into mm/s
    df_['vel_fly'] = np.concatenate(
                        (np.zeros(1), 
                        np.sqrt(np.sum(np.square(np.diff(df_[['pos_x_smoothed', 'pos_y_smoothed']], axis=0)), 
                        axis=1)))) / (winsize*4/fps)# /4 
    df_.loc[df_['vel_fly']>50, 'vel_fly'] = np.nan 
    df_['vel_fly'] = df_['vel_fly'].interpolate().ffill().bfill()
        
    df_ = util.smooth_and_calculate_velocity_circvar(df_, smooth_var='ori', 
                                                        vel_var='ang_vel_fly', 
                                                        time_var='sec', winsize=winsize)            
    # Calculate difference in ori between consecutive rows 
    df_['turn_size'] = df_.groupby('id')['ori'].transform(lambda x: x.diff())

    # Calculate relative vel
    df_['rel_vel'] = df_['dist_to_other'].interpolate().diff() / df_['sec'].diff().mean()

    df_['ang_acc'] = df_['ang_vel'].diff() / df_['sec_diff'].mean()
    df_['ang_acc_smoothed'] = util.smooth_orientations_pandas(df_['ang_acc'], winsize=3) 

    df_['facing_angle_acc'] = df_['facing_angle_vel'].diff() / df_['sec_diff'].mean()

    df_['ang_acc_fly'] = df_['ang_vel_fly'].diff() / df_['sec_diff'].mean()
    df_['ang_acc_fly_smoothed'] = df_['ang_vel_fly_smoothed'].diff() / df_['sec_diff'].mean()

    df_['theta_error_acc'] = df_['theta_error_dt'].diff() / df_['sec_diff'].mean()

    if filter_funky:    
        df_.loc[(df_['ang_vel_fly']>80) | (df_['ang_vel_fly']<-80), 'ang_vel_fly'] = np.nan

    return df_
 
def calculate_angle_metrics_focal_fly(df0, winsize=5, grouper='acquisition',
                                      fps=60, filter_funky=True):
    '''
    Loop through all the acquisitions (or specified by grouper) to calculate additional angle metrics for a focal fly.
    Arguments:
        df0 -- dataframe with focal fly data
        winsize -- smoothing window size (default: 5)
        grouper -- column to group by (default: 'acquisition')
        fps -- frames per second (default: 60)
        filter_funky -- whether to filter out funky values of ang vel (default: True)
    '''
    assert df0['id'].nunique()==1, "Too many fly IDs, specify 1"

    # df_ = df[df['acquisition']==acq].copy()
    d_list = []
    for acq, df_ in df0.groupby(grouper):
        #print(acq)
       df_ = additional_angle_metrics_single(df_, winsize=winsize, fps=fps, filter_funky=filter_funky) 
       d_list.append(df_)

    df = pd.concat(d_list)
    #%
    #df.loc[(df['ang_vel_fly']>80) | (df['ang_vel_fly']<-80), 'ang_vel_fly'] = np.nan

    #% and get abs values
    df['targ_pos_theta_deg'] = np.rad2deg(df['targ_pos_theta'])
    df['facing_angle_deg'] = np.rad2deg(df['facing_angle'])
    #df['rel_vel'] = np.nan

    df['rel_vel_abs'] = np.abs(df['rel_vel']) 
    df['targ_ang_vel_abs'] = np.abs(df['targ_ang_vel'])
    df['targ_pos_theta_abs'] = np.abs(df['targ_pos_theta'])
    if 'targ_ang_size' in df.columns: #has_size:
        df['targ_ang_size_deg'] = np.rad2deg(df['targ_ang_size'])
    df['targ_ang_vel_deg'] = np.rad2deg(df['targ_ang_vel'])
    df['targ_ang_vel_deg_abs'] = np.abs(df['targ_ang_vel_deg'])
    df['facing_angle_deg'] = np.rad2deg(df['facing_angle'])
    df['ang_vel_fly_abs'] = np.abs(df['ang_vel_fly'])

    df['facing_angle_vel_abs'] = np.abs(df['facing_angle_vel'])
    df['facing_angle_vel_deg'] = np.rad2deg(df['facing_angle_vel'])
    df['facing_angle_vel_deg_abs'] = np.abs(df['facing_angle_vel_deg'])

    df['theta_error_deg'] = np.rad2deg(df['theta_error'])
    df['theta_error_dt_deg'] = np.rad2deg(df['theta_error_dt'])
    df['theta_error_abs'] = np.abs(df['theta_error'])

    df['ang_vel_abs'] = np.abs(df['ang_vel'])
    df['ang_vel_deg'] = np.rad2deg(df['ang_vel'])

    # ftjaaba['ang_vel_abs'] = np.abs(ftjaaba['ang_vel'])
    df['ang_vel_deg'] = np.rad2deg(df['ang_vel'])

    df['ang_vel_fly_abs'] = np.abs(df['ang_vel_fly'])
    df['ang_vel_fly_deg'] = np.rad2deg(df['ang_vel_fly'])
    df['ang_vel_fly_abs_deg'] = np.rad2deg(df['ang_vel_fly_abs'])

    return df


def do_transformations_on_df(trk_, centroid_x, centroid_y,
                             feat_=None,
                             cop_ix=None, flyid1=0, flyid2=1,
                             verbose=False, get_relative_sizes=True, 
                             winsize=5, fps=60):
    '''
    Transform coordinates to focal fly (do it for both fly1 and fly2).
    Also calculate additional metrics based on these relative positions.
    Includes calculate_thet_error()

    Arguments:
        trk_ -- df with basic track data, positions, orientations, etc. 
        frame_width -- width of video frame, generally from calibration 
        frame_height -- height of video frame, generally from calibration

    Keyword Arguments:
        feat_ (pd.DataFrame, None): df with features, including dist_to_other (default: {None})
            - can be None if feat_ concatenated within trk_
        cop_ix (int): Frame index of copulation start (default: {None})
        flyid1 (int): id of male (default: {0})
        flyid2 (int): id of female (default: {1})
        verbose (bool): describe each step (default: {False})
    Returns:
        _description_
    '''
    if feat_ is None:
        assert 'dist_to_other' in trk_.columns, "No feat df provided. Need dist_to_other."

    if 'sec' not in trk_.columns:
        trk_['sec'] = trk_['frame'] / fps
    if 'sec' not in feat_.columns:
        feat_['sec'] = feat_['frame'] / fps

    # center x- and y-coordinates, generate new columns ctr_x and ctr_y which report position relative to center of frame 
    if verbose:
        print("Doing transformations:") #.format(acq)) 
        print("... centering coordinates")
    trk_ = util.center_coordinates(trk_, centroid_x, centroid_y) 

    # extract copies of trk and feat dfs for each fly 
    fly1 = trk_[trk_['id']==flyid1].copy().reset_index(drop=True)
    fly2 = trk_[trk_['id']==flyid2].copy().reset_index(drop=True)

    fly1_feat = feat_[feat_['id']==flyid1].copy().reset_index(drop=True)
    fly2_feat = feat_[feat_['id']==flyid2].copy().reset_index(drop=True)
    
    # FIRST, do fly1: -------------------------------------------------
    # translate coordinates so that focal fly is at origin
    # Creates trans_x, trans_y
    if verbose:
        print("... translating to focal fly1")
    fly1, fly2 = util.translate_coordinates_to_focal_fly(fly1, fly2)
    fly1['targ_centered_x'] = fly2['ctr_x']
    fly1['targ_centered_y'] = fly2['ctr_y']

    fly1['targ_centered_to_focal_x'] = fly2['trans_x']
    fly1['targ_centered_to_focal_y'] = fly2['trans_y']

    # add vel, both of which are position agnostic 
    fly1['target_vel'] = fly2_feat['vel']
    fly1['target_ang_vel'] = fly2_feat['ang_vel']

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    if verbose:
        print("... rotating to focal fly1")
    fly1, fly2 = util.rotate_coordinates_to_focal_fly(fly1, fly2)

    # add polar conversion
    polarcoords = util.cart2pol(fly2['rot_x'], fly2['rot_y']) 
    fly1['targ_pos_radius'] = polarcoords[0]
    fly1['targ_pos_theta'] = polarcoords[1]
    fly1['targ_rel_pos_x'] = fly2['rot_x']
    fly1['targ_rel_pos_y'] = fly2['rot_y']
    fly1['targ_rel_ori'] = fly2['rot_ori']

    # NOW, do fly2: ----------------------------------------------------
    # translate coordinates so that focal fly is at origin
    if verbose:
        print("... now repeating for fly2")
    fly2, fly1 = util.translate_coordinates_to_focal_fly(fly2, fly1)
    fly2['targ_centered_x'] = fly1['ctr_x']
    fly2['targ_centered_y'] = fly1['ctr_y']

    fly2['targ_centered_to_focal_x'] = fly1['trans_x']
    fly2['targ_centered_to_focal_y'] = fly1['trans_y']

    fly2['target_vel'] = fly1_feat['vel']
    fly2['target_ang_vel'] = fly1_feat['ang_vel']

    # rotate coordinates so that fly1 is facing 0 degrees (East)
    # Assumes fly1 ORI goes from 0 to pi CCW, with y-axis NOT-inverted.
    # if using FlyTracker, trk_['ori'] = -1*trk_['ori']
    fly2, fly1 = util.rotate_coordinates_to_focal_fly(fly2, fly1)

    # add polar conversion
    polarcoords = util.cart2pol(fly1['rot_x'], fly1['rot_y']) 
    fly2['targ_pos_radius'] = polarcoords[0]
    fly2['targ_pos_theta'] = polarcoords[1]
    fly2['targ_rel_pos_x'] = fly1['rot_x']
    fly2['targ_rel_pos_y'] = fly1['rot_y']
    fly2['targ_rel_ori'] = fly1['rot_ori']

    df = pd.concat([fly1, fly2], axis=0).reset_index(drop=True) #.so
    return df

    #% copulation index - TMP: fix this!
    if cop_ix is None or np.isnan(cop_ix):
        cop_ix = len(fly1)
        copulation = False
    else:
        copulation = True
    cop_ix = int(cop_ix)

    if get_relative_sizes:
        if verbose:
            print("... getting target sizes")
        #% Get all sizes and aggregate trk df
        fly1 = get_target_sizes(fly1, fly2, xvar='pos_x', yvar='pos_y')
        # Repeat for fly2:
        fly2 = get_target_sizes(fly2, fly1, xvar='pos_x', yvar='pos_y')

    if verbose:
        print("... calculating theta error")
    # calculate theta error, which is the angle between either the fly's orientation (heading) or traveling direction
    # and the line between the fly and the target (the other fly)
    fly1 = calculate_theta_error(fly1, fly2)
    fly1 = calculate_theta_error_from_traveling_direction(fly1, fly2)
    fly2 = calculate_theta_error(fly2, fly1)
    fly2 = calculate_theta_error_from_traveling_direction(fly2, fly1)

    # recombine trk df
    trk = pd.concat([fly1.iloc[:cop_ix], fly2.iloc[:cop_ix]], axis=0).reset_index(drop=True)#.sort_index()
    trk['copulation'] = copulation

    # Get relative velocity and aggregate feat df
    if verbose:
        print("... calculating relative velocity")
    if feat_ is not None:
        if verbose:
            print("... calculating relative velocity from feat df")
        f_list = []
        for _, df_ in feat_.groupby('id'):
            df_ = get_relative_velocity(df_, win=1, 
                                value_var='dist_to_other', time_var='sec')
            f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
        feat = pd.concat(f_list, axis=0).reset_index(drop=True) #.sort_index()
        feat['copulation'] = copulation
        #print(trk.iloc[-1].name, feat.iloc[-1].name)
        df = pd.concat([trk, 
                feat.drop(columns=[c for c in feat.columns if c in trk.columns])], axis=1)
        assert df.shape[0]==trk.shape[0], "Bad merge: {}, {}".format(feat.shape, trk.shape)
        if verbose:
            print("... df shape:", df.shape)
    else:
        if verbose:
            print("(no feat df provided, using trk df)")
        # feat_ and trk_ are already combined 
        f_list = []
        assert 'dist_to_other' in trk.columns, "No feat df provided. Need dist_to_other."
        for fi, df_ in trk.groupby('id'):
            df_ = get_relative_velocity(df_, win=1, 
                                value_var='dist_to_other', time_var='sec')
            f_list.append(df_.reset_index(drop=True).iloc[:cop_ix])
        df = pd.concat(f_list, axis=0).reset_index(drop=True) #.sort_index()


    # Add addtional angular metrics (like ang_vel_fly and smoothing)
    if verbose:
        print("... calculating additional angle metrics")
        print(df['id'].unique())
        f1_slice = df[df['id']==flyid1]
        print("theta_error NaN:", f1_slice['theta_error'].isna().sum(), "/", len(f1_slice))
        print("theta_error sample:", f1_slice['theta_error'].dropna().head())

    f1 = additional_angle_metrics_single(df[df['id']==flyid1].copy(), winsize=winsize, fps=fps, filter_funky=True)
    f2 = additional_angle_metrics_single(df[df['id']==flyid2].copy(), winsize=winsize, fps=fps, filter_funky=True)
    # Recombine f1 and f2
    df = pd.concat([f1, f2], axis=0).reset_index(drop=True) #.so

    #acq = os.path.split(acqdir)[-1]

    return df


def load_processed_df(processedmat_dir, acqdir=None, acq=None,
                    create_new=False):
    '''
    Load a processed DataFrame from the processed-mats directory.

    Checks for parquet first, then falls back to pickle (migrating to parquet).
    Returns None if ``create_new`` is True or no file is found.

    Arguments:
        acqdir (str): Path to the acquisition video directory (basename used as acq name).

    Keyword Arguments:
        savedir (str or None): Directory containing processed files.
            Defaults to *acqdir* when None. (default: {None})
        create_new (bool): If True skip loading and return None so the
            caller can regenerate. (default: {False})

    Returns:
        pd.DataFrame or None
    '''
    if acqdir is None:
        assert acq is not None, "Either acqdir or acq must be provided"
        acqdir = os.path.join(processedmat_dir, acq)
    elif acq is None:
        assert acqdir is not None, "Either acqdir or acq must be provided"
        acq = os.path.split(acqdir)[-1]

    #acq = os.path.split(acqdir)[-1]
    parquet_fpath = os.path.join(processedmat_dir, '{}_df.parquet'.format(acq))
    pkl_fpath = os.path.join(processedmat_dir, '{}_df.pkl'.format(acq))

    if create_new:
        return None

    if os.path.exists(parquet_fpath):
        print('Loaded: {}'.format(parquet_fpath))
        return pd.read_parquet(parquet_fpath)

    if os.path.exists(pkl_fpath):
        df_, _ = util.replace_pkl_with_parquet(pkl_fpath)
        return df_

    print('No processed df found for {}'.format(acq))
    return None


def get_metrics_relative_to_focal_fly(acqdir, mov_is_upstream=False, fps=60, cop_ix=None,
                                      movie_fmt='avi', flyid1=0, flyid2=1,
                                      default_frame_width=None, default_frame_height=None,
                                      plot_checks=False, get_relative_sizes=True,
                                      save=False, savedir=None, create_new=False):
    '''
    Load -feat.mat and -trk.mat, do some processing, save processed df to savedir.

    Arguments:
        acqdir -- _description_

    Keyword Arguments:
        get_relative_sizes (bool): Calculate target size based on distance and angle, SLOW (default: {True})
        fps (float): Acquisition rate (default: {60})
        cop_ix: Copulation frame or None (default: {None})
        movie_fmt (str): (default: {'avi'})
        flyid1 (int): flytracker male index (default: {0})
        flyid2 (int): flytracker female index (default: {1})
        plot_checks (bool): do some plotting to check correct transformations (default: {False})
        savedir (str, None): Save processed _df.pkl (default: {None})
    '''
    # check output dir
    if savedir is None:
        if save is True:
            print("No save directory provided. Saving to acquisition directory.")
        savedir = acqdir

    # try loading existing processed df
    cached = load_processed_df(savedir, acqdir=acqdir, create_new=create_new)
    if cached is not None:
        return cached

    acq = os.path.split(acqdir)[-1]
    parquet_fpath = os.path.join(savedir, '{}_df.parquet'.format(acq))
    pkl_fpath = os.path.join(savedir, '{}_df.pkl'.format(acq))

    print("Not loading old, creating new: {}".format('{}_df.parquet'.format(acq)))
    # load flyracker data
    if mov_is_upstream:
        subfolder = ''
    else:
        subfolder = '*'
    calib_, trk_, feat_ = util.load_flytracker_data(acqdir, fps=fps, 
                                                    calib_is_upstream=mov_is_upstream,
                                                    subfolder=subfolder,
                                                    filter_ori=True)

    # get video file for plotting/sanity checks
#    vids = util.get_videos(acqdir, vid_type=movie_fmt)
#    alt_movie_fmt = 'mp4' if movie_fmt=='avi' else 'avi'
#    try:
#        assert len(vids)>0, "Found no video in directory: {}".format(vids)
#        vids = [vids[-1]]
#    except AssertionError as e:
#        vids = util.get_videos(acqdir, vid_type=alt_movie_fmt)
#        assert len(vids)==1, "Found more than one video in directory: {}".format(vids)  
#
#    vidpath = vids[0]
#    cap = cv2.VideoCapture(vidpath)
    if mov_is_upstream:
        parentdir, ftname = os.path.split(acqdir)
        viddir = os.path.split(parentdir)[0]
        cap = get_video_cap(viddir, ftname=ftname, movie_fmt=movie_fmt) 
    else:
        cap = get_video_cap(acqdir, movie_fmt=movie_fmt)

    # N frames should equal size of DCL df
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if n_frames == 0:
        print("No frames found in video: {}".format(acqdir))
        print('{}: using default, {}x{}'.format(acq, default_frame_width, default_frame_height))
        frame_width = default_frame_width
        frame_height = default_frame_height
    else:
        frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    #print(frame_width, frame_height) # array columns x array rows
    # Check that we have 2 flies
    if trk_['id'].nunique()==1:
        print("ERROR: Only 1 fly found ({})".format(acq) )
        return None

    # switch ORI
    trk_['ori'] = -1*trk_['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
    df_ = do_transformations_on_df(trk_, frame_width, frame_height, 
                                   feat_=feat_, cop_ix=cop_ix,
                                   flyid1=flyid1, flyid2=flyid2, 
                                   get_relative_sizes=get_relative_sizes)

    # save
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        df_.to_parquet(parquet_fpath, engine='pyarrow', compression='snappy')
        print('Saved: {}'.format(parquet_fpath))
        # remove any stale pkl
        if os.path.exists(pkl_fpath):
            os.remove(pkl_fpath)

    #% plot - sanity checks
    if plot_checks:
        fly1 = df_[df_['id']==flyid1]
        fly2 = df_[df_['id']==flyid2]
        # check affine transformations for centering and rotating male
        ix = 6500 #5000 #2500 #590
        fig = plot_frame_check_affines(ix, fly1, fly2, cap, frame_width, frame_height)
        fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)

        # check projections for calculating size based on distance and angle
        ix = 100 #213527 #5000 #2500 #590
        for ix in [100, 3000, 5000]:
            fig = plot_frame_target_projection(ix, fly1, fly2, cap, 
                                            xvar='pos_x', yvar='pos_y')
            fig.text(0.1, 0.95, os.path.split(acqdir)[-1], fontsize=4)

    return df_



def load_processed_data(acqdir, savedir=None, load=True):
    '''
    Backward-compatible wrapper around :func:`load_processed_df`.

    When *load* is True, returns the DataFrame (or None).
    When *load* is False, returns True/False indicating whether a file exists.
    '''
    if savedir is None:
        savedir = acqdir
    df_ = load_processed_df(savedir, acqdir=acqdir)
    if load:
        return df_
    return df_ is not None
    


#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process FlyTracker data for relative metrics.')
    parser.add_argument('--savedir', type=str, help='Directory to save processed data.')    
    parser.add_argument('--movie_fmt', type=str, default='avi', help='Movie format (default: avi).')
    parser.add_argument('--flyid1', type=int, default=0, help='ID of focal fly (default: 0).')  
    parser.add_argument('--flyid2', type=int, default=1, help='ID of target fly (default: 1).') 
    parser.add_argument('--plot_checks', type=bool, default=False, help='Plot checks (default: False).')    
    parser.add_argument('--viddir', type=str, default='/Volumes/Julie/38mm_dyad/courtship-videos/38mm_dyad', help='Root directory of videos (default: /Volumes/Julie/38mm_dyad/courtship-videos/38mm_dyad).')   
    parser.add_argument('--new', type=bool, default=False, help='Create new processed data (default: False).')
    parser.add_argument('--subdir', type=str, default=None, help='subdir of tracked folders, e.g., fly-tracker (default: None).')
    parser.add_argument('--single', type=bool, default=False, help='Set True if just running one acquisition, provide path to acquisition dir for viddir rather than parent (default: False).')
    parser.add_argument('--default_frame_width', type=int, default=None, help='Default frame width (default: None).')
    parser.add_argument('--default_frame_height', type=int, default=None, help='Default frame height (default: None).')

    args = parser.parse_args()
    # 
    viddir = args.viddir 
    savedir = args.savedir
    movie_fmt = args.movie_fmt
    flyid1 = args.flyid1
    flyid2 = args.flyid2
    subdir = args.subdir
    create_new = args.new
    single = args.single

#%% #Hardcoded parameter values for running in interactive mode
    interactive = False

    #viddir = '/Volumes/Giacomo/free_behavior_data'
    #savedir = '/Volumes/Julie/free-behavior-analysis/38mm_dyad/MF/FlyTracker/processed_mats'

    #viddir = '/Volumes/Giacomo/JAABA_classifiers/projector/changing_dot_size_speed'
    #savedir = '/Volumes/Julie/2d-projector-analysis/FlyTracker/processed_mats'

    if interactive:
        viddir = '/Volumes/Juliana/2d-projector'
        savedir = '/Volumes/Juliana/2d-projector-analysis/circle_diffspeeds/FlyTracker/processed_mats'
        subdir = 'fly-tracker'
        flyid1 = 0
        flyid2 = 1
        movie_fmt = '.avi'
        create_new=True
        default_frame_width = None
        default_frame_height = None

#%%
    if single:
        if subdir is not None:
            found_mats = glob.glob(os.path.join(viddir, '*{}*'.format(subdir), '*', '*feat.mat'))
        else:
            found_mats = glob.glob(os.path.join(viddir, '*', '*feat.mat'))
    else:
        if subdir is not None:
            found_mats = glob.glob(os.path.join(viddir,  '20*', '*{}*'.format(subdir), '*', '*feat.mat'))
        else:
            found_mats = glob.glob(os.path.join(viddir,  '20*', '*', '*feat.mat'))
    print('Found {} processed videos.'.format(len(found_mats)))

    #%% For each found acquisition (video), calculate relative metrics
    for fp in found_mats:
        if (subdir is not None) or (single is True):
            # FT output dir is parent dir
            ftdir = os.path.split(fp)[0]
            # video dir is upstream
            if subdir is not None:
                viddir = os.path.split(ftdir)[0]    
            elif single is True:
                viddir = os.path.split(os.path.split(ftdir)[0])[0]
            # acq name is FT name
            acq = os.path.split(ftdir)[-1]
        else:
            acq = os.path.split(os.path.split(fp.split(viddir+'/')[-1])[0])[0]

        if "BADTRACKING" in fp: # Giacomo added this to skip bad tracking
            continue
        
        acqdir = os.path.join(viddir, acq)
        print(acq)
        print(acqdir)
        # Try loading data
        if not create_new:
            df_ = load_processed_data(acqdir, load=False, savedir=savedir)
            if df_ is False: 
                create_new=True #assert ft is True, "No feat df found, creating now."
            else:
                print("processed mat exists, skipping")        
        # Create a new processed_mat file by calculating relative metrics
        if create_new:
            print("Creaing new procssed metrics")
            if 'projector' not in viddir:
                cop_ix = get_copulation_ix(acq)
            else:
                cop_ix = None
            df_ = get_metrics_relative_to_focal_fly(acqdir,
                                        savedir=savedir,
                                        movie_fmt=movie_fmt, 
                                        mov_is_upstream=subdir is not None,
                                        flyid1=flyid1, flyid2=flyid2,
                                        default_frame_width=default_frame_width,
                                        default_frame_height=default_frame_height,
                                        plot_checks=False,
                                        create_new=create_new,
                                        save=create_new)

