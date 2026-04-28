#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/01/28 13:17:48
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

'''
import os
import re
import glob
import scipy.io
import mat73
import cv2

from itertools import groupby
from operator import itemgetter

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# General
# ---------------------------------------------------------------------
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

def flatten(t):
    return [item for sublist in t for item in sublist]

def euclidean_dist(df1, df2, cols=['x_coord','y_coord']):
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)

def check_nan(wingR):
    if any(np.isnan(wingR)):
        wingR_ = pd.Series(wingR)
        wingR = wingR_.interpolate()
    return wingR

def set_angle_range_to_neg_pos_pi(ang):
    from math import remainder, tau

#    if ang>np.pi:
#        ang = ang - 2*np.pi
#    elif ang<-np.pi:
#        ang = ang + 2*np.pi
    return remainder(ang, tau)

def CoM(df, xvar='pos_x', yvar='pos_y'):
    '''
    Calculate center of mass for x, y coordinates in df.
    
    Arguments:
        df -- _description_

    Returns:
        _description_
    '''
    x = df[xvar].values
    y = df[yvar].values
    m = np.ones(df[xvar].shape)
    cgx = np.sum(x*m) / np.sum(m)
    cgy = np.sum(y*m) / np.sum(m)

    return cgx, cgy

def circular_distance(ang1, ang2):
    # efficiently computes the circular distance between two angles (Tom/Rufei)

    circdist = np.angle(np.exp(1j * ang1) / np.exp(1j * ang2))

    return circdist

def wrap_to_2pi(lambda_):
    # Wrap angles in lambda to the interval [0, 2*pi]
    positive_input = (lambda_ > 0)
    lambda_ = np.mod(lambda_, 2 * np.pi)
    lambda_[(lambda_ == 0) & positive_input] = 2 * np.pi
    return lambda_

def circ_dist(x, y):
    # Check input dimensions
    if x.shape != y.shape and y.size != 1:
        raise ValueError('Input dimensions do not match.')
    
    # Compute pairwise circular difference
    r = np.angle(np.exp(1j * x) / np.exp(1j * y))
    return r

def wrap_pi(a):  # (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

def circ_dist2(x, y):
    '''
    % r = circ_dist(alpha, beta)
    %   All pairwise difference x_i-y_j around the circle computed efficiently.
    %
    %   Input:
    %     alpha       sample of linear random variable
    %     beta       sample of linear random variable
    %
    %   Output:
    %     r       matrix with pairwise differences
    %
    % References:
    %     Biostatistical Analysis, J. H. Zar, p. 651
    %
    % PHB 3/19/2009
    %
    % Circular Statistics Toolbox for Matlab

    % By Philipp Berens, 2009
    % berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

    np.angle:  Calculates the phase angle of a complex number or 
               array of complex numbers, returning the angle in radians by default
               Uses the atan2 function internally to compute the angle between 
               the positive real axis and the complex number on the complex plane
               Returns in range (-pi, pi].
    ''' 

    #np.tile(A, (m, n)) like repmat(A, m, n) in MATLAB

    at = np.angle( np.tile( np.exp(1j*x), (1, np.size(y)) ) / np.tile( np.exp(1j*y), (np.size(x), 1) ) )

    if len(at) == 1:
        return float(at)
    else:
        return at #np.exp(1j*x) / np.exp(1j*y) )



# Function to keep value if all are the same, else return NaN
def only_if_unique(x):
    ''' 
    Use in df.groupby().mean() type of functions to return the same value if all values are unique and non-numeric
    
    # Apply custom aggregation
    result = df.groupby('group').agg(
        {col: 'mean' if pd.api.types.is_numeric_dtype(dtype)
        else only_if_unique
        for col, dtype in df.dtypes.items()}
    ).reset_index()

    '''
    if x.nunique() == 1:
        return x.iloc[0]
    return pd.NA

def groupby_aggr_if_numeric(f1, groupby_cols=[], 
                            aggr_type='mean'):
    '''
    Group by specified columns and aggregate numeric columns with the specified aggregation type.
    Non-numeric columns are aggregated using util.only_if_unique.
    
    Parameters:
    - f1: DataFrame to group and aggregate.
    - groupby_cols: List of columns to group by. 
    - aggr_type: Aggregation type for numeric columns (default is 'mean').
    
    Returns:
    - meanbouts: DataFrame with aggregated values.
    '''
    meanbouts = f1.groupby(groupby_cols, as_index=False).agg(
                    {col: aggr_type if pd.api.types.is_numeric_dtype(dtype)
                    else only_if_unique
                    for col, dtype in f1.dtypes.items()}
                ).reset_index()
    return meanbouts


def convert_time_to_seconds(time_str):
    '''
    Convert time string to seconds.
    Assumes that time_str is in MM:SS format.
    
    Arguments:
        time_str -- time string in MM:SS format

    Returns:
        seconds -- time in seconds
    
    Example:
        >>> convert_time_to_seconds('00:01:00')
        60
        >>> convert_time_to_seconds('01:00:00')
        3600
    '''
    # if only 2 elements, assume it's MM:SS
    if len(time_str.split(':')) == 2:
        time_str = '00:' + time_str
        
    h, m, s = time_str.split(':')
    return int(h)*60 + int(m)*60 + int(s)

# ---------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------
def get_species_from_acquisition_name(acq: str) -> str:
    """
    Extract 4-letter species code starting with 'D' from acquisition name.
    Example: '..._Dele-HK_...' -> 'Dele'. Falls back to known substrings.
    """
    m = re.search(r'_(D[a-zA-Z]{3})', acq)
    if m:
        return m.group(1)
    if 'mel' in acq:
        return 'Dmel'
    if 'yak' in acq:
        return 'Dyak'
    if 'ele' in acq:
        return 'Dele'
    return 'Unknown'


#%% 
# ---------------------------------------------------------------------
# Bout parsing
# ---------------------------------------------------------------------
def get_indices_of_consecutive_rows(passdf):
    '''
    Find start and stop indices of consecutive rows in a dataframe.

    Arguments:
        passdf -- frames which pass boolean condition(s)

    Returns:
        Series of tuples, each containing start and stop indices of consecutive rows
        Also updates passdf with "diff" (can ignore) and "group" columns, the latter contains bout nums
    '''
    passdf['diff'] = passdf.index.to_series().diff().fillna(1)
    passdf['diff'] = passdf['diff'].apply(lambda x: 1 if x>1 else 0)
    passdf['group'] = passdf['diff'].cumsum()

    return passdf.groupby('group').apply(lambda x: (x.index[0], x.index[-1]))


def filter_bouts_by_frame_duration(consec_bouts, min_bout_len, fps=60, return_indices=False):
    min_bout_len_frames = min_bout_len*fps # corresponds to 0.25s at 60Hz
    incl_bouts = [c for i, c in enumerate(consec_bouts) if c[1]-c[0]>=min_bout_len_frames]
    incl_ixs = [i for i, c in enumerate(consec_bouts) if c[1]-c[0]>=min_bout_len_frames]
    #print("{} of {} bouts pass min dur {}sec".format(len(incl_bouts), len(consec_bouts), min_bout_len))

    if return_indices:
        return incl_ixs, incl_bouts
    else:
        return incl_bouts

def subdivide_into_subbouts(ftjaaba, bout_dur=0.2, grouper='acquisition'):
    d_list = []
    for acq, df_ in ftjaaba.groupby(grouper):
        sec_min, sec_max = df_['sec'].min(), df_['sec'].max()

        group_dur_sec = sec_max - sec_min #df_['sec'].max() - df_['sec'].min()
        n_mini_bouts = int(group_dur_sec / bout_dur)
        #print(n_mini_bouts)
        bins = np.linspace(sec_min, sec_max, n_mini_bouts, endpoint=False)
        bin_labels = np.arange(0, len(bins)-1)
        #print(bin_labels)
        df_['subboutnum'] = pd.cut(df_['sec'], bins=bins, labels=bin_labels)
        d_list.append(df_)

    ftjaaba = pd.concat(d_list)

    return ftjaaba

def subdivide_bouts_into_subbouts(filtdf, bout_dur=1.0):
    '''
    Subivide chunks of consecutive frames into smaller bouts of bout_dur length.
    Takes larger BOUTS (courting or not courting) 
    Arguments:
        filtdf -- filtered/selected df, must contain original indices (frames) for grouping

    Keyword Arguments:
        bout_dur -- duration of each chunk in sec (default: {1.0})

    Returns:
        filtdf with columns "group" (consecutive chunks) and "boutnum" (mini bouts) added 
    '''
    # assign grouping based on row index -- filtdf should have the original indices (frames) as ftjaaba
    consec_bouts = get_indices_of_consecutive_rows(filtdf)

    b_list = []
    boutnum = 0
    for g, df_ in filtdf.groupby('group'):
        group_dur_sec = df_.iloc[-1]['sec'] - df_.iloc[0]['sec']
        #print(group_dur_sec)
        if group_dur_sec / bout_dur < 2: #bout_dur: #2:
            df_['boutnum'] = boutnum
            #filtdf.loc[filtdf['group'] == g, 'boutnum'] = boutnum
            bin_labels = [boutnum]
        else:
            # subdivide into mini-bouts of bout_dur length
            group_dur_sec = df_.iloc[-1]['sec'] - df_.iloc[0]['sec']
            #t0 = df_.iloc[0]['sec']
            n_mini_bouts = int(group_dur_sec / bout_dur)
            #t1_values = np.linspace(t0 + bout_dur, t0 + group_dur_sec, n_mini_bouts, endpoint=False, )

            bins = np.linspace(df_.iloc[0]['sec'], df_.iloc[-1]['sec'], n_mini_bouts, endpoint=False)
            bin_labels = np.arange(boutnum, boutnum + len(bins)-1)
            #print(bin_labels)
            df_['boutnum'] = pd.cut(df_['sec'], bins=bins, labels=bin_labels)

            #filtdf.loc[filtdf['group']==g, 'boutnum'] = df_['bin_sec']

        boutnum += len(bin_labels)
        b_list.append(df_)
    filtdf = pd.concat(b_list)

    return filtdf



# processing

def circmedian(angs):
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
    pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    pdists = np.abs(pdists).sum(1)
    return angs[np.argmin(pdists)]

def smooth_orientations(y, winsize=5):
    yy = np.concatenate((y, y))
    smoothed = np.convolve(np.array([1] * winsize), yy)[winsize: len(y) + winsize]
    return smoothed #% (2 * np.pi)

def smooth_orientations_pandas(x, winsize=3): 
    # 'unwrap' the angles so there is no wrap around
    x1 = pd.Series(np.rad2deg(np.unwrap(x)))
    # smooth the data with a moving average
    # note: this is pandas 17.1, the api changed for version 18
    x2 = x1.rolling(winsize, min_periods=1).mean() #pd.rolling_mean(x1, window=3)
    # convert back to wrapped data if desired
    x3 = x2 % 360
    return np.deg2rad(x3)

def smooth_and_calculate_velocity_circvar(df, smooth_var='ori', vel_var='ang_vel',
                                  time_var='sec', winsize=3):
    '''
    Smooth circular var and then calculate velocity. Takes care of NaNs.
    Assumes 'id' is in df.

    Arguments:
        df -- _description_

    Keyword Arguments:
        smooth_var -- _description_ (default: {'ori'})
        vel_var -- _description_ (default: {'ang_vel'})
        time_var -- _description_ (default: {'sec'})
        winsize -- _description_ (default: {3})

    Returns:
        _description_
    '''
    df[vel_var] = np.nan
    df['{}_smoothed'.format(smooth_var)] = np.nan
    for i, df_ in df.groupby('id'): 
        # unwrap for continuous angles, then interpolate NaNs
        #nans = df_[df_[smooth_var].isna()].index
        unwrapped = pd.Series(np.unwrap(df_[smooth_var].interpolate().ffill().bfill()),
                            index=df_.index) #.interpolate().values))

        # interpolate over nans now that the values are unwrapped
        oris = unwrapped.interpolate() 

        # smooth with rolling() - this sets range from 0, 2pi
        smoothed = smooth_orientations_pandas(oris, winsize=winsize) #smoothed = smooth_orientations(df_['ori'], winsize=3)

        # revert back to -pi, pi
        smoothed_wrap = pd.Series([set_angle_range_to_neg_pos_pi(i) for i in smoothed])
        df.loc[df['id']==i, '{}_smoothed'.format(smooth_var)] = smoothed_wrap.values

        # unwrap again to take difference between oris -- should look similar to ORIS
        #smoothed_unwrap = pd.Series(np.unwrap(smoothed), index=df_.index)

        # take difference and set range -pi, pi
        smoothed_diff = [set_angle_range_to_neg_pos_pi(i) for i in smoothed_wrap.diff().bfill()]
        # get smoothed angular velocity by dividing by timme
        ang_vel_smoothed = smoothed_diff / df_[time_var].diff().mean()

        #smoothed_diff = np.concatenate([[0], smoothed_diff])
        ori_diff = [set_angle_range_to_neg_pos_pi(i) for i in oris.diff().bfill()]
        ang_vel = ori_diff / df_[time_var].diff().mean() 

        df.loc[df['id']==i, vel_var] = ang_vel
        df.loc[df['id']==i, '{}_diff'.format(smooth_var)] = ori_diff #.values

        df.loc[df['id']==i, '{}_smoothed'.format(vel_var)] = ang_vel_smoothed #.values
        #df.loc[df['id']==i, '{}_smoothed_range'.format(smooth_var)] = [set_angle_range_to_neg_pos_pi(i) for i in smoothed_wrap]

    # Only wipe columns that this call produced, not all columns.
    # Wiping all columns here destroys source data (ori, theta_error, sec, etc.)
    # whenever ANY input variable has NaN frames.
    bad_ixs = df[df[smooth_var].isna()]['frame'].dropna().index.tolist()
    produced_cols = [vel_var, '{}_smoothed'.format(smooth_var),
                     '{}_diff'.format(smooth_var), '{}_smoothed'.format(vel_var)]
    wipe_cols = [c for c in produced_cols if c in df.columns]
    df.loc[bad_ixs, wipe_cols] = np.nan

    df['{}_abs'.format(vel_var)] = np.abs(df[vel_var])

    return df

# ---------------------------------------------------------------------
# Some vector calcs 
# ---------------------------------------------------------------------
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_heading_vector(f_ori, f_len):
    # female ori and length
    # get female heading as a vector
    th = (np.pi - f_ori) % np.pi
    y_ = f_len/2 * np.sin(th)
    x_ = f_len/2 * np.cos(th)
    return np.array([x_, y_])

def compute_dist_to_other(pos_x, pos_y, other_pos_x, other_pos_y,
                          pix_per_mm=1, smooth=True):
    '''Compute Euclidean distance between two flies, matching FlyTracker's
    feat_compute.m (lines ~140-146).

    FlyTracker computes the raw pixel distance, smooths with a [1 2 1]/4
    kernel, then converts to mm via pix_per_mm.  Both flies receive the
    same symmetric distance value.

    Reference
    ---------
    kristinbranson/FlyTracker  tracking/feat_compute.m  lines 140-146
    https://github.com/kristinbranson/FlyTracker/blob/main/tracking/feat_compute.m

    Args
    ----
    pos_x, pos_y :             (np.ndarray) position of focal fly (pixels)
    other_pos_x, other_pos_y : (np.ndarray) position of other fly (pixels)
    pix_per_mm :               (float) pixels per mm from calibration
                               (default 1, returns pixels)
    smooth :                   (bool) apply [1,2,1]/4 smoothing kernel
                               (default True, matching FlyTracker)

    Returns
    -------
    dist : (np.ndarray) distance in mm (or pixels if pix_per_mm=1)
    '''
    dist = np.sqrt((pos_x - other_pos_x)**2 + (pos_y - other_pos_y)**2)
    if smooth and len(dist) >= 3:
        kernel = np.array([1, 2, 1]) / 4.0
        dist[1:-1] = np.convolve(dist, kernel, mode='valid')
    return dist / pix_per_mm


def compute_facing_angle(ori, pos_x, pos_y, other_pos_x, other_pos_y):
    '''Compute the facing angle between a fly's heading and the direction
    toward another fly, matching FlyTracker's feat_compute.m (lines ~157-162).

    The heading vector is [cos(ori), -sin(ori)] (image coordinates where
    y increases downward).  The facing angle is the unsigned angle between
    this heading and the unit vector pointing from the focal fly to the
    other fly:  facing_angle = acos(dot(heading, direction_to_other)).

    Result is in radians, range [0, pi].  0 = pointing at the other fly,
    pi = facing away.

    Reference
    ---------
    kristinbranson/FlyTracker  tracking/feat_compute.m  lines 149-162
    https://github.com/kristinbranson/FlyTracker/blob/main/tracking/feat_compute.m

    Args
    ----
    ori :          (np.ndarray) orientation of focal fly (radians)
    pos_x, pos_y : (np.ndarray) position of focal fly
    other_pos_x, other_pos_y : (np.ndarray) position of other fly

    Returns
    -------
    facing_angle : (np.ndarray) unsigned angle in [0, pi] radians
    '''
    heading = np.column_stack([np.cos(ori), -np.sin(ori)])
    vec_to_other = np.column_stack([other_pos_x - pos_x,
                                    other_pos_y - pos_y])
    norm = np.linalg.norm(vec_to_other, axis=1, keepdims=True)
    norm[norm == 0] = np.nan
    vec_to_other = vec_to_other / norm
    dot = np.clip(np.sum(heading * vec_to_other, axis=1), -1, 1)
    return np.arccos(dot)


def proj_a_onto_b(a, b):
    #np.sqrt(female_hat[0]**2 + female_hat[1]**2)
    #sign = -1 if f_ori < 0 else 1
    #vproj_ = sign * np.array(ortho_hat) * (np.dot(f_vec, ortho_) / np.linalg.norm(ortho_)**2) #, ortho_)
    #vproj_ = np.array(ortho_) * (np.dot(f_vec, ortho_) / np.linalg.norm(ortho_)**2) #, ortho_)
    vproj_ = np.array(b) * (np.dot(a, b) / np.linalg.norm(b)**2) #, ortho_)
    #ang_bw = angle_between(female_hat, ortho_)
    #proj_ = ortho_hat* np.sqrt(x_**2 + y_**2) * np.cos(ang_bw)
    return vproj_

def calculate_female_size_deg(xi, yi, f_ori, f_len):
    '''
    Calculate size of target (defined by f_ori, f_len) in degrees of visual angle.
    Finds vector orthogonal to focal and target flies. Calculates heading of target using f_ori and f_len. Then, projects target heading onto orthogonal vector.
    Size is calculated as 2*arctan(fem_sz/(2*dist_to_other)).
    Note: make sure units are consistent (e.g., pixels for f_len, xi, yi).

    Arguments:
        xi -- x coordinate of vector between focal and target flies
        yi -- y coordinate of vector between focal and target flies
        f_ori -- orientation of target fly (from FlyTracker, -180 to 180; 0 faces east, positive is CCW)
        f_len -- length of target fly (from FlyTracker, in pixels)

    Returns:
        Returns calculated size in deg for provided inputs.
    '''
    # get vector between male and female
    #xi = fly2.loc[ix][xvar] - fly1.loc[ix][xvar] 
    #yi = fly2.loc[ix][yvar] - fly1.loc[ix][yvar]

    # get vector orthogonal to male's vector to female
    ortho_ = [yi, -xi] #ortho_hat = ortho_ / np.linalg.norm(ortho_)

    # project female heading vec onto orthog. vec
    #f_ori = fly2.loc[ix]['ori']
    #f_len = fly2.loc[ix]['major_axis_len']
    fem_vec = get_heading_vector(f_ori, f_len) #np.array([x_, y_])
    #female_hat = fem_vec / np.linalg.norm(fem_vec)
    vproj_ = proj_a_onto_b(fem_vec, ortho_)

    # calculate detg vis angle
    fem_sz = np.sqrt(vproj_[0]**2 + vproj_[1]**2) * 2
    dist_to_other = np.sqrt(xi**2 + yi**2)
    fem_sz_deg = 2*np.arctan(fem_sz/(2*dist_to_other))

    return fem_sz_deg


def center_coordinates(df, frame_width, frame_height, 
                       xvar='pos_x', yvar='pos_y', ctrx='ctr_x', ctry='ctr_y'):
    '''
    _summary_

    Arguments:
        df -- pd.DataFrame with columns xvar and yvar
        frame_width -- height of frame, corresponds to fly x-pos
        frame_height -- width of frame, corresponds to fly y-pos

    Keyword Arguments:
        xvar -- _description_ (default: {'pos_x'})
        yvar -- _description_ (default: {'pos_y'})
        ctrx_x -- _description_ (default: {'ctr_x'})
        ctrx_y -- _description_ (default: {'ctr_y'})

    Returns:
        df -- pd.DataFrame with new columns ctr_x and ctr_y
    '''
    df[ctrx] = df[xvar] - frame_width/2
    df[ctry] = df[yvar] - frame_height/2

    return df

def translate_coordinates_to_focal_fly(fly1, fly2):
    '''
    Translate coords so that x, y of focal fly (fly1) is (0, 0). 
    Assumes coordsinates have been centered already (ctr_x, ctr_y).

    Arguments:
        fly1 -- _description_
        fly2 -- _description_

    Returns:
        fly1, fly2 with columns 'trans_x' and 'trans_y'. fly1 is 0.
    '''
    assert 'ctr_x' in fly1.columns, "No 'ctr_x' column in fly1 df"
    fly1['trans_x'] = fly1['ctr_x'] - fly1['ctr_x']
    fly1['trans_y'] = fly1['ctr_y'] - fly1['ctr_y']
    fly2['trans_x'] = fly2['ctr_x'] - fly1['ctr_x']
    fly2['trans_y'] = fly2['ctr_y'] - fly1['ctr_y']

    return fly1, fly2

def rotate_point(p, angle, origin=(0, 0)): #degrees=0):
    '''
    Calculate rotation matrix R and perform R.dot(p.T) to get rotated coords.

    Returns:
        _description_
    '''
    #angle = np.deg2rad(degrees)
    R = np.squeeze(np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]]))
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rotate_coordinates_to_focal_fly(fly1, fly2):
    '''
    Apply rotation to fly2 so that fly1 is at 0 heading.
    Assumes 'ori' is a column in fly1 and fly2. (from FlyTracker)

    Arguments:
        fly1 -- _description_
        fly2 -- _description_

    Returns:
        fly1, fly2 with columns 'rot_x' and 'rot_y'. fly1 is 0.

    '''
    assert 'trans_x' in fly1.columns, "trans_x not found in fly1 DF"
    ori_vals = fly1['ori'].values # -pi to pi
    ori = -1*ori_vals + np.deg2rad(0) # ori - ori is 0 heading 

    fly2[['rot_x', 'rot_y']]= np.nan
    fly1[['rot_x', 'rot_y']] = 0

    #rotmats = np.array([np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])\
    #            for theta in ori] )

    #xys = fly2[['trans_x', 'trans_y']].values
    #fly2[['rot_x', 'rot_y']] = [rot.dot(xy) for xy, rot in zip(xys, rotmats)]

    fly2[['rot_x', 'rot_y']] = [rotate_point(pt, ang) for pt, ang in zip(fly2[['trans_x', 'trans_y']].values, ori)]

    fly2['rot_ori'] = fly2['ori'] + ori                 
    fly1['rot_ori'] = fly1['ori'] + ori # should be 0

    return fly1, fly2


def cart2pol(x, y):
    '''
    Returns radius * theta in radians

    Arguments:
        x -- _description_
        y -- _description_
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


# ---------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------
def replace_pkl_with_parquet(pkl_fpath):
    '''
    Load a pickle file, save it as parquet alongside, and delete the pkl.

    Arguments:
        pkl_fpath -- path to the .pkl file to migrate

    Returns:
        df -- loaded DataFrame
        parquet_fpath -- path to the written .parquet file
    '''
    parquet_fpath = pkl_fpath.replace('.pkl', '.parquet')
    df = pd.read_pickle(pkl_fpath)
    df.to_parquet(parquet_fpath, engine='pyarrow', compression='snappy')
    os.remove(pkl_fpath)
    print('Migrated {} -> {}'.format(os.path.basename(pkl_fpath),
                                     os.path.basename(parquet_fpath)))
    return df, parquet_fpath


def load_ft_actions(found_actions_paths, split_end=False):
    a_ = []
    for fp in found_actions_paths:
        actions_ = ft_actions_to_bout_df(fp)
        if split_end:
            basename = '_'.join(os.path.split(fp)[-1].split('_')[0:-1])
        else:
            basename = os.path.split(fp)[-1].split('-actions')[0]
        print(basename)
        actions_['acquisition'] = basename
        #actions_['ction_num'] = actions_.index.tolist()
        actions_['species'] = actions_['acquisition'].str.extract(r'(D\w{3})')
        a_.append(actions_)

    actions_df = pd.concat(a_)

    return actions_df

def assign_action_frames_to_df(df, actions):
    '''
    Assign action frames to df.

    Arguments:
        df -- dataframe with frame column
        actions -- dataframe with actions, boutnums, and start/end frames (Created by FlyTracker annotations)

    Returns:
        df -- dataframe with action columns and boutnums
    '''
    for (action_name, bout_num), a_df in actions.groupby(['action', 'boutnum']):
        #print(action_name, bout_num)
        if action_name not in df.columns:
            print("Adding new action column: ", action_name)
            df[action_name] = 0
            df[f'{action_name}_boutnum'] = None
        # Assign all frames between start and end in df
        start, end = a_df['start'].item(), a_df['end'].item()
        frame_range = np.arange(start, end + 1)
        df.loc[df['frame'].isin(frame_range), action_name] = 1
        df.loc[df['frame'].isin(frame_range), f'{action_name}_boutnum'] = bout_num

    return df


def combine_jaaba_and_processed_df(df, jaaba):
    '''
    Combines processed df (containing feat, trk data, or equivalent) with jaaba data.

    Arguments:
        df -- _description_
        jaaba -- _description_

    Returns:
        _description_
    '''
    c_list = []
    no_dfs = []
    for acq, ja_ in jaaba.groupby('acquisition'):
        df_ = df[(df['acquisition']==acq) & (df['id']==0)].reset_index(drop=True)
        try:
            if len(df_)>0:
                if ja_.shape[0] < df_.shape[0]:
                    last_frame = ja_['frame'].max()
                    df_ = df_[df_['frame']<=last_frame]
                else:
                    assert ja_.shape[0] == df_.shape[0], "Mismatch in number of flies between jaaba {} and processed data {}.".format(ja_.shape, df_.shape) 
                drop_cols = [c for c in ja_.columns if c in df_.columns]
                combined_ = pd.concat([df_, ja_.drop(columns=drop_cols)], axis=1)
                assert combined_.shape[0] == df_.shape[0], "Bad merge: {}".format(acq)
                c_list.append(combined_)
            else:
                no_dfs.append(acq)
        except Exception as e:
            print(acq)
            print(e)
            continue

    ftjaaba = pd.concat(c_list, axis=0).reset_index(drop=True)

    return ftjaaba

def split_condition_from_acquisition_name(df0):
    '''
    Sometimes condition (led level, stimulus size, etc.) is saved in acquisition filename. 
    Extract condition, save as column in df0, and update acquisition name to standard.
    NOTE: currently only set to check for ledXX in filename (for projector assay tests)
    Arguments:
        df0 -- pd.DataFrame
            Can be aggregate or 1 dataframe, will add flynum, date_str, then group by flynum, datestr, species

    Returns:
        newdf -- pd.DataFrame
            Subset of data, with 1 acquisition per date-fly combo.
    '''
    # find the pattern 'flyX' where X is a number in ft_acq
    #ft_acq = [s for s in ft_acq if re.search(r'fly\d', s)]
    df0['flynum'] = [int(re.search('_fly\d_', a)[0][4:-1]) if 'fly' in a else 
                     int(re.search('_f\d_', a)[0][2:-1]) for a in df0['acquisition']]
    df0['date_str'] = [int(v[0:8]) for v in df0['acquisition']]
    # check that we only take 1 file per date/fly combo
    df_list = []
    #df0['led'] = 0
    for (flynum, datestr, sp), d_ in df0.groupby(['flynum', 'date_str', 'species']):
        #if len(d_['acquisition'].unique())>1 and 
        curr_acqs = d_['acquisition'].unique()
        if any(['led' in a for a in curr_acqs]):
            #print(flynum, datestr, curr_acqs)
            if len([a for a in curr_acqs if a.endswith('led20')]) > 0:
                keep_acq = [a for a in curr_acqs if a.endswith('led20')][0]
                led_level = 20
            elif len([a for a in curr_acqs if not a.endswith('led00')]) > 0:
                led_str = [a.split('led')[-1] for a in curr_acqs if not a.endswith('led00')]
                nonzero_led_ix = np.argmax([int(v) for v in led_str])
                led_level = int(led_str[nonzero_led_ix])
                keep_acq = [a for a in curr_acqs if a.endswith('led{}'.format(led_str[nonzero_led_ix]))][0]
            # updated df
            keep_df = d_[d_['acquisition']==keep_acq].copy()
            keep_df['led'] = led_level
            updated_acq = keep_acq.split('_led')[0]
            keep_df['acquisition'] = updated_acq
            df_list.append(keep_df)
        else:
            df_list.append(d_)

    newdf = pd.concat(df_list)

    return newdf



def load_aggregate_data_pkl(savedir, mat_type='df', included_species=None):
    '''
    Find all *feat.pkl (or *trk.pkl) files in savedir and load them into a single dataframe.
    These files are created by relative_metrics.py and contain processed data from FlyTracker.

    Arguments:
        savedir -- Full path to dir containing processed *.pkl files, e.g., _feat.pkl, _df.pkl, etc.

    Keyword Arguments:
        mat_type -- feat or trk (default: {'feat'})

    Returns:
        feat -- pandas dataframe containing all processed data.
    '''
    found_fns = glob.glob(os.path.join(savedir, '*{}.pkl'.format(mat_type)))
    print("Found {} processed *_{}.pkl files".format(len(found_fns), mat_type))
    f_list=[]
    for fp in found_fns:
        if 'BADTRACKING' in fp:
            continue

        acq = os.path.split(fp)[1].split('_{}'.format(mat_type))[0] 
        # format acq
        year_str = acq[0:4]
        if len(re.findall(year_str, acq))>1:
            acq = acq.split('_'+year_str)[0]

        if 'yak' in acq:
            sp = 'Dyak'
        elif 'mel' in acq:
            sp = 'Dmel'
        elif 'ele' in acq:
            sp = 'Dele'
        if included_species is not None:
            if sp not in included_species:
                continue

        #if 'ele' in fp: # ignore ele for now
        #    continue
        #fp = found_fns[0]
        #acq = os.path.split(acq_viddir)[0]
        print(os.path.split(fp)[-1])
        #with open(fp, 'rb') as f:
        #    feat_ = pkl.load(f)
        feat_ = pd.read_pickle(fp)
        feat_['acquisition'] = acq 
        feat_['species'] = sp

        f_list.append(feat_)

    feat = pd.concat(f_list, axis=0).reset_index(drop=True) 

    return feat

def ft_actions_to_bout_df(action_fpath):
    '''
    Take manually annoted bouts from FlyTracker -actions.mat file and convert to a pandas df

    Arguments:
        action_fpath -- _description_

    Returns:
        _description_
    '''
    # mat['bouts'] is (n_flies, action_types)
    # mat['bouts'][0, 10] gets bout start/end/likelihood for fly1, action #10
    # mat['behs'] are the behavior names

    # load mat
    mat = scipy.io.loadmat(action_fpath)

    # behavior names
    behs = [i[0][0] for i in mat['behs']]

    # aggregate into list
    b_list = []
    for i, beh in enumerate(behs):
        # get male action's 
        if mat['bouts'][0, i].shape[1]==3:
            b = mat['bouts'][0, i]
            b_df = pd.DataFrame(data=b, columns=['start', 'end', 'likelihood'])
            b_df['action'] = beh
            b_df['id'] = 0
            b_list.append(b_df)

    boutdf = pd.concat(b_list)
    boutdf['boutnum'] = boutdf.index.tolist()

    return boutdf

def load_jaaba(assay='2d-projector', experiment='circle_diffspeeds', fname=None):
    '''
    Assay can be '2d-projector' or '38mm-dyad' -- uses hard-coded local paths for faster loading.

    Arguments:
        assay -- _description_

    Returns:
        _description_
    '''
    #local_basedir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data'
    local_basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee'
    if assay=='2d-projector':
        srcdir =  os.path.join(local_basedir, '2d_projector', assay, experiment, 'FlyTracker')
        #% Load jaaba-traansformed data
        #jaaba_fpath = os.path.join(srcdir, 'jaaba_transformed_data_transf.pkl')
        if fname is None:
            jaaba_fpath = os.path.join(srcdir, 'projector_data_mel_yak_20240330_jaaba.pkl')
        else:
            if 'jaaba' in fname:
                jaaba_fpath = os.path.join(srcdir, '{}.pkl'.format(fname))
            else:
                jaaba_fpath = glob.glob(os.path.join(srcdir, '*{}*jaaba.pkl'.format(fname)))[0]

        assert os.path.exists(jaaba_fpath), "File not found: {}".format(jaaba_fpath)
        jaaba = pd.read_pickle(jaaba_fpath)   
        print(jaaba['species'].unique())
    elif assay=='38mm_dyad':
        #jaaba_file = '/Volumes/Julie/free-behavior-analysis/38mm-dyad/jaaba.pkl'
        srcdir = os.path.join(local_basedir, 'free_behavior', assay, experiment, 'FlyTracker') #'38mm-dyad-ft-jaaba'
        if fname is not None:
            jaaba_fpath = os.path.join(srcdir, '{}.pkl'.format(fname))
        else:
            jaaba_fpath = os.path.join(srcdir, 'jaaba.pkl')
        print("loading: {}".format(jaaba_fpath))
        jaaba = pd.read_pickle(jaaba_fpath)
        #with open(jaaba_fpath, 'rb') as f:
        #    jaaba = pkl.load(f)
        #jaaba.head()
    print("Loaded: {}".format(jaaba_fpath))

    return jaaba

def binarize_behaviors(df, jaaba_thresh_dict, courtvar='courting'):
    '''
    Assign binary labels to behaviors based on jaaba_thresh_dict.

    Arguments:
        df -- _description_
        jaaba_thresh_dict -- _description_

    Keyword Arguments:
        courtvar -- _description_ (default: {'courting'})

    Returns:
        _description_
    '''

    for behav, thresh in jaaba_thresh_dict.items():
        df['{}_binary'.format(behav)] = 0
        df.loc[df[behav]>thresh, '{}_binary'.format(behav)] = 1

    return df

def assign_jaaba_behaviors(plotdf, courtvar='courting', jaaba_thresh_dict=None, min_thresh=5):
    if jaaba_thresh_dict is None:
        jaaba_thresh_dict = {'orienting': min_thresh, 'chasing': min_thresh, 'singing': min_thresh} 
    plotdf.loc[plotdf[courtvar]==0, 'behavior'] = 'disengaged'
    for b, thr in jaaba_thresh_dict.items():
        plotdf.loc[plotdf[b]>thr, 'behavior'] = b
    #plotdf.loc[plotdf['chasing']>, 'behavior'] = 'chasing'
    #plotdf.loc[plotdf['singing']>0, 'behavior'] = 'singing'
    #plotdf.loc[((plotdf['chasing']>0) & (plotdf['singing']==0)), 'behavior'] = 'chasing only'
    return plotdf


def get_video_cap_check_multidir(acq, assay='2d-projector', return_viddir=False,
                                minerva_base='/Volumes/Juliana'):
    '''
    Specific issue where multiple vid sources are possible (e.g., Minerva and Giacomo's drives).

    Arguments:
        acq -- _description_

    Returns:
        _description_
    '''
    #minerva_base='/Volumes/Juliana'

    session = acq.split('-')[0]
    viddir = os.path.join(minerva_base, assay, session)

    # videos are in parent dir of FT folder
    #cap = pp.get_video_cap(viddir) #acqdir)
    try:
        vids = glob.glob(os.path.join(minerva_base, assay, '20*', '{}*.avi'.format(acq)))
        video_fpath = vids[0]
    except IndexError:
        if '2x2' in acq:
            minerva_base2 = '/Volumes/Giacomo/JAABA_classifiers/projector/changing_dot_size_speed/2x2'
        else:
            minerva_base2 = '/Volumes/Giacomo/JAABA_classifiers/projector/changing_dot_size_speed'

        found_vids = glob.glob(os.path.join(minerva_base2, '{}*'.format(acq), 'movie.avi'))
        assert len(found_vids)>0, "No video found for acq {}\nChecked in: {}".format(acq, minerva_base2)
        video_fpath = found_vids[0]
        print(video_fpath)

    cap = cv2.VideoCapture(video_fpath)

    if return_viddir:
        return cap, viddir
    else:
        return cap

def get_acq_from_ftpath(fp, viddir):     
    '''
    Return acq name (subdir of viddir that contains video file)

    Arguments:
        fp -- _description_
        viddir -- _description_

    Returns:
        _description_
    '''
    acq = os.path.split(os.path.split(fp.split(viddir+'/')[-1])[0])[0]
    return acq

def get_video_by_ft_name(viddir, ftname, vid_type='.avi'):

    found_vidpaths = glob.glob(os.path.join(viddir, '*{}*{}'.format(ftname, vid_type)))

    return found_vidpaths


def get_videos(folder, vid_type='.avi'):
    '''
    _summary_

    Arguments:
        folder -- parent dir containing video files

    Keyword Arguments:
        vid_type -- _description_ (default: {'.avi'
    })

    Returns:
        returns list of video file paths
    '''
    found_vidpaths = glob.glob(os.path.join(folder, '*{}'.format(vid_type)))

    return found_vidpaths

def get_acq_dir(sessionid, assay_prefix='single_20mm*', rootdir='/mnt/sda/Videos'):
    '''
    Args:
    -----
    sessionid: (str)
        YYYYMMDD-HHMM prefix for acquisition id
    '''
    acquisition_dirs = sorted([f for f in glob.glob(os.path.join(rootdir, '%s*' % assay_prefix, 
                            '%s*' % sessionid)) if os.path.isdir(f)], key=natsort)
    #print("Found %i acquisitions from %s" % (len(acquisition_dirs), sessionid))
    assert len(acquisition_dirs)==1, "Unable to find unique acq. from session ID: %s" % sessionid
    acq_dir = acquisition_dirs[0]
    acquisition = os.path.split(acq_dir)[-1]

    return acq_dir

def get_movie_metadata(curr_movie_path):
    '''
    Get metadata for specified movie.

    Args:
    -----
    curr_movie_path: (str)
        Path to .avi file
        
    Returns
    -------
    minfo: dict
    '''
    vidcap = cv2.VideoCapture(curr_movie_path)
    success, image = vidcap.read()
    framerate = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    minfo = {'framerate': framerate,
             'width': width,
             'height': height,
             'n_frames': n_frames,
             'movie_path': curr_movie_path
    }

    vidcap.release()

    return minfo


# matlab analysis output to python (AO)
def mat_data_names_to_df(currmat):
    '''
    Specific func for data output stores in struct with 'data' and names' as dict.

    Returns: df, where data is 'data' and 'names' are columns
    '''
    df = pd.DataFrame(data=currmat['data'], columns=currmat['names']).fillna(0)

    return df


def mat_combine_binary_behaviors(curr_acq_mat):
    '''
    Take as input dict of dicts (from a given acquisition) from custom
    output analysis (quick_ethograms.m):

    Arguments:
        curr_acq_mat (dict): mat[species][acquisition_index] 
                             NOTE: (keys should include 'behavior', 'wings')

    Returns:
        B_df (pd.DataFrame): all aggregated behaviors that were binarized in Matlab function (GetBinaryBehaviors_v4.m and wings.m -- outputs of quick_ethograms.m)
 
    '''
    binary_behaviors = mat_data_names_to_df(curr_acq_mat['behavior'])
    binary_behaviors = binary_behaviors.rename(columns={'All Wing Ext': 'All Wing Extensions'})
    # tseries = pd.DataFrame(data=mat[sp][acq_ix]['wings']['tseries']['data'],
    #                        columns=mat[sp][acq_ix]['wings']['tseries']['names']).fillna(0)
    binary_wings = mat_data_names_to_df(curr_acq_mat['wings']['tseries'])
    assert binary_wings['All Wing Extensions'].equals(binary_behaviors['All Wing Extensions']), "ERROR: 'wings' tseries W.E. not the same as 'behavior' dict"
    B_df = pd.merge(binary_behaviors, binary_wings,  how='left', 
             left_on=['All Wing Extensions', 'Time Vector (s)'], right_on=['All Wing Extensions', 'Time Vector (s)']) # right_on = ['B_c1','c2'])
    return B_df



def mat_get_bout_indices(currmat):
    '''
    Fromat separate bout index (start/end) arrays from custom matlab analysis into one dataframe. NOTE: This is specific to WING events.
    
    Arguments:
        currmat (dict): mat[species][acquisition_index]['wings']

    Returns:
        bouts (pd.DataFrame): dataframe with start/end indices of bouts and wings

    '''

    # Get main BOUTS df
    bouts = pd.DataFrame(data=currmat['wings']['bouts']['data'],
                         columns=currmat['wings']['bouts']['names'])

    # boutname = 'boutsL'
    for boutname, colname in zip(['boutsL', 'boutsR', 'boutsB'], ['Left', 'Right', 'Bilateral']):
        if 'names' not in currmat['wings'][boutname].keys():
            currmat['wings'][boutname]['names'] = np.array(['{} Ext Start Indices'.format(colname), '{} Ext End Indices'.format(colname)])

        if currmat['wings'][boutname]['data'].shape[0]==0:
            continue 

        if len(currmat['wings'][boutname]['data'].shape)==1:
            bouts_ = pd.DataFrame(data=currmat['wings'][boutname]['data'],
                             index=currmat['wings'][boutname]['names']).T
        else:
            bouts_ = pd.DataFrame(data=currmat['wings'][boutname]['data'],
                             columns=currmat['wings'][boutname]['names'])
        for bi, brow in bouts_.iterrows():
            s_ix, e_ix = brow[['{} Ext Start Indices'.format(colname), '{} Ext End Indices'.format(colname)]]
            # Note, sometimes bout indices not exact match for Bilateral case
            bouts.loc[(bouts['All Ext Start Indices']<=s_ix) & (bouts['All Ext End Indices']>=e_ix), 'wing'] = colname 
            
        
    return bouts


def mat_split_courtship_bouts(bin_, bout_marker='Disengaged'):
    '''
    Use binary Disengaged 1 or 0 to find bout starts, assign from 0
    '''
    diff_ = bin_[bout_marker].diff()
    bout_starts = np.where(diff_!=0)[0] # each of these index values is the START ix of a bout
    for i, v in enumerate(bout_starts):
        if i == len(bout_starts)-1:
            bin_.loc[v:, 'boutnum'] = i
        else:
            v2 = bout_starts[i+1]
            bin_.loc[v:v2, 'boutnum'] = i
    return bin_


def get_bout_durs(df, bout_varname='boutnum', return_as_df=False,
                    timevar='Time Vector (s)'):
    '''
    Get duration of parsed bouts. 
    Parse with parse_bouts(count_varname='instrip', bout_varname='boutnum').

    Arguments:
        df -- behavior dataframe, must have 'boutnum' as column (run parse_inout_bouts())  

    Returns:
        dict, keys=boutnum, vals=boutdur (in sec)
    '''
    assert 'boutnum' in df.columns, "Bouts not parse. Run:  df=parse_inout_bouts(df)"

    boutdurs={}
    grouper = ['boutnum']
    for boutnum, df_ in df.groupby(bout_varname):
        boutdur = df_.sort_values(by=timevar).iloc[-1][timevar] - df_.iloc[0][timevar]
        boutdurs.update({boutnum: boutdur})

    if return_as_df:
        durs_ = pd.DataFrame.from_dict(boutdurs, orient='index').reset_index()
        durs_.columns = [bout_varname, 'boutdur']

        return durs_

    return boutdurs



# ---------------------------------------------------------------------
# FlyTracker functions
# ---------------------------------------------------------------------

def add_frame_nums(trackdf, fps=None):
    '''Add frame index and sec to dataframes
    '''
    frame_ixs = trackdf[trackdf['id']==0].index.tolist()
    trackdf['frame'] = np.nan
    for i, g in trackdf.groupby('id'):
        trackdf.loc[g.index, 'frame'] = frame_ixs
    trackdf['frame'] = trackdf['frame'].astype(int)

    if fps is not None:
        trackdf['sec'] = trackdf['frame'] / float(fps)
    
    return trackdf

def get_mat_paths_for_all_vids(acquisition_dir, subfolder='*', ftype='track'):
    '''
    Get all .mat files associated with a given acquisition (experiment)
    
    Args:
    -----
    acquisition_dir: (str) 
        Dir containing all input movies (and FlyTracker results).
    ftype: (str)
        Type of .mat to load. Must be: -bg, -feat, -seg, or -track.
         
    Returns:
    --------
    Sorted list of all .mat files. 
    ''' 
    paths_to_matfiles = sorted(glob.glob(os.path.join(acquisition_dir, subfolder, '*%s.mat' % ftype)), 
           key=natsort)
      
    return paths_to_matfiles


def get_feature_units(mat_fpath):
    '''
    Load -feat.mat and get units for each of the var. names
    '''
    try:
        mat = scipy.io.loadmat(mat_fpath)
    except NotImplementedError as e:
        mat = mat73.loadmat(mat_fpath) #scipy.io.loadmat(mat_fpath)
    
    mdata = mat.get('feat')
    mdtype = mdata.dtype
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}

    columns = [n[0].replace(' ', '_') for n in ndata['names'][0]]
    units = [n[0] if len(n)>0 else 'NA' for n in ndata['units'][0]]
    
    unit_lut = dict((k, v) for k, v in zip(columns, units))
    
    return unit_lut

def load_calibration(curr_acq, calib_is_upstream=False):
    '''
    Load calibration.mat from FlyTracker

    Load calibration.mat from FlyTracker. 
    1. Searches for the calibration.mat file in the parent directory of curr_acq (if calib_is_upstream=True) or in curr_acq (if False).
    2. Loads the .mat file and extracts the struct containing calibration data.
    3. Parses the struct fields and values into a Python dictionary, extracting the relevant fields andconverting types as needed

    fieldnames
    - rois - (n_chambers, 4) array of [y, x, height, width] for each chamber. coordinates of extreme points of chamber. 
        Can use to estimate w/h if missing.
    - centroids - (n_chambers, 2) array of [x, y] for each chamber. centroid coordinates of each chamber.
        Can use to estimate w/h if missing.
    - w, h - frame width and height. If missing, will try to estimate from rois.
    - FPS - frames per second
    - PPM - pixels per mm
    - n_chambers, n_rows, n_cols - chamber layout info
    - cop_ind - index of frames where copulation occurs

    Args:
    -----
    curr_acq: (str)
        Dir containing all .avi files for specific acquisition (and calibration.mat files).
        
    Returns:
    --------
    
    '''
    fieldnames = ['n_chambers', 'n_rows', 'n_cols', 'FPS', 'PPM', 'w', 'h', 
                  'centroids', 'rois', 'n_flies', 'cop_ind']
    if calib_is_upstream:
        # assumes flyracker folders were saved in a subdir, and that calibration.mat is in the parent dir, same level as subdir
        basedir = os.path.split(os.path.split(curr_acq)[0])[0]
    else:
        basedir = curr_acq
    calib_fpath = os.path.join(basedir, 'calibration.mat')

    assert os.path.exists(calib_fpath), "No calibration found: %s" % curr_acq

    try:
        mat = scipy.io.loadmat(calib_fpath)
    except NotImplementedError as e:
        mat = mat73.loadmat(calib_fpath) #scipy.io.loadmat(mat_fpath)

    struct_name = [k for k in mat.keys() if not k.startswith('__')]
    assert len(struct_name)==1, "Did not find unique struct name: %s" % str(struct_name)
    
    mdata = mat.get(struct_name[0])

    # Use fields to create dict
    # 'names' (1, 35) 
    # 'data' (n_flies, n_frames, n_fields)
    # 'flags' (possible switches, check with flytracker/visualizer)
    mdtype = mdata.dtype
    if mdata.ndim == 1:
        ndata = {n: mdata[n][0] for n in mdtype.names}
    else:
        ndata = {n: mdata[n][0, 0] for n in mdtype.names}

    all_fields = dict((k, v[0]) if len(v)>0 else (k, v) for k, v in ndata.items())
    calib = {}
    for k, v in all_fields.items():
        if k not in fieldnames:
            continue
        if len(v)==1:
            if v.dtype=='object':
                calib[k] = np.array(v[0][0])
            
            else:
                calib[k] = int(v) if v.dtype in ['uint8', 'uint16'] else float(v)
        else:
            calib[k] = np.array(v)

    # DEBUG: show parsed calib before w/h fallback
    print("DEBUG load_calibration parsed: {}".format(
        {k: (v, type(v).__name__, v.shape if isinstance(v, np.ndarray) else '') 
         for k, v in calib.items()}))

    # Estimate w/h if missing (empty arrays from some calibration files)
    w_missing = 'w' not in calib or (isinstance(calib['w'], np.ndarray) and calib['w'].size == 0)
    h_missing = 'h' not in calib or (isinstance(calib['h'], np.ndarray) and calib['h'].size == 0)
    if w_missing or h_missing:
        source = None
        # `rois` in calib is (n_chambers, 4) array of [y, x, height, width] for each chamber. Can use to estimate w/h if missing.
        if 'rois' in calib and isinstance(calib['rois'], np.ndarray) and calib['rois'].size > 0:
            roi = calib['rois']
            if roi.ndim == 1:
                # single chamber: [y, x, height, width]
                if w_missing:
                    calib['w'] = float(roi[3])
                if h_missing:
                    calib['h'] = float(roi[2])
            else:
                # multiple chambers: (n_chambers, 4)
                if w_missing:
                    calib['w'] = float((roi[:, 3]).max())
                if h_missing:
                    calib['h'] = float((roi[:, 2]).max())
            source = 'rois'
        if source:
            print("WARNING: estimated frame size from {}: w={}, h={}".format(
                source, calib.get('w', 'MISSING'), calib.get('h', 'MISSING'))) # 2nd arg is the value given if value is missing for the key

    return calib

def load_feat(curr_acq, subfolder='*'):
    mat_fpaths = get_mat_paths_for_all_vids(curr_acq, subfolder=subfolder, ftype='feat')
    feat = load_mat(mat_fpaths)
    return feat

def load_tracks(curr_acq, subfolder='*'):
    mat_fpaths = get_mat_paths_for_all_vids(curr_acq, subfolder=subfolder, ftype='track')
    trk = load_mat(mat_fpaths)
    return trk

def load_mat_frames_and_var(mat_fpath):
    try:
        mat = scipy.io.loadmat(mat_fpath)
    except NotImplementedError as e:
        mat = mat73.loadmat(mat_fpath) #scipy.io.loadmat(mat_fpath)

    struct_name = [k for k in mat.keys() if not k.startswith('__')]
    assert len(struct_name)==1, "Did not find unique struct name: %s" % str(struct_name) 
    mdata = mat.get(struct_name[0])

    # Use fields to create dict
    # 'names' (1, 18) 
    # 'data' (n_frames, n_vars)
    # 'units' (units of variables) 
    mdtype = mdata.dtype
    ndata = {n: mdata[n][0][0] for n in mdtype.names}

    columns = [n[0] for n in ndata['names'][0]]
    n_frames, n_vars = ndata['data'].shape
    # turn into dataframe
    df = pd.DataFrame(data=ndata['data'], columns=columns)

    return df
 
 
def load_mat(mat_fpaths): #results_dir):
    '''
    Load track.mat and parse into dataframe. Assumes data is nflies, nframes, nflags.

    Args:
    -----
    mat_fpaths: list
        List of path(s) to -track.mat file (from FlyTracker) 

    Returns:
    -------
    df: (pd.DataFrame)
        Dataframe of all the extracted data from FlyTracker.
        Rows are frames, columns are features (including fly ID, 'id')
    '''
    #ft_outfile = glob.glob(os.path.join(results_dir, '*', '*track.mat'))[0]
    #print(ft_outfile)
    all_dfs=[]
    for mat_fpath in sorted(mat_fpaths, key=natsort):
        try:
            mat = scipy.io.loadmat(mat_fpath)
            struct_name = [k for k in mat.keys() if not k.startswith('__')]
            assert len(struct_name)==1, "Did not find unique struct name: %s" % str(struct_name) 
            mdata = mat.get(struct_name[0])

            # Use fields to create dict
            # 'names' (1, 35) 
            # 'data' (n_flies, n_frames, n_fields)
            # 'flags' (possible switches, check with flytracker/visualizer)
            mdtype = mdata.dtype
            ndata = {n: mdata[n][0][0] for n in mdtype.names}
            columns = [n[0].replace(' ', '_') for n in ndata['names'][0]]
        except NotImplementedError as e:
            mat = mat73.loadmat(mat_fpath) #scipy.io.loadmat(mat_fpath)
            struct_name = [k for k in mat.keys() if not k.startswith('__')]
            # is already dict
            ndata = mat[struct_name[0]] 
            columns = [n.replace(' ', '_') for n in ndata['names']]

        n_flies, n_frames, n_flags = ndata['data'].shape
        d_list=[]
        for fly_ix in range(n_flies):
            tmpdf = pd.DataFrame(data=ndata['data'][fly_ix, :], columns=columns)
            tmpdf['id'] = fly_ix
            d_list.append(tmpdf)
        df_ = pd.concat(d_list, axis=0, ignore_index=True)
        df_['fpath'] = mat_fpath  
        all_dfs.append(df_) 
        
    df = pd.concat(all_dfs, axis=0, ignore_index=True)

    return df

def load_binary_evs_from_mat(matlab_src, feat=None, sex='m',
                behavior_names=['Bilateral Wing Extensions', 'Unilateral Wing Extensions', 'Putative Tap Events', 'Chasing', 'Licking/Proboscis Ext', 'Copulation Attempts', 'Orienting']):

    '''
    Specific to output from matlab using AO's binarization of behaviors for ethograms.
    matlab_src: path to Ddata.mat (output of quick_ethograms.m)

    '''
    try:
        mat = scipy.io.loadmat(matlab_src, simplify_cells=True)
    except NotImplementedError as e:
        mat = mat73.loadmat(matlab_src) #scipy.io.loadmat(mat_fpath)

    species_list = [k for k in mat.keys() if not k.startswith('__')]
    nonorienting_names = [b for b in behavior_names if b!='Orienting']

    binevs_list=[]
    for sp in species_list:
        if len(mat[sp])==0:
            continue
        if not isinstance(mat[sp], list):
            mat[sp] = [mat[sp]]
        for acq_ix, acq_mat in enumerate(mat[sp]):
            acq = acq_mat['acquisition']
            if feat is not None and acq not in feat['acquisition'].unique():
                print("skipping: {}".format(acq))
                continue

            #if acq in ['20240109-1039_fly1_eleWT_4do_sh_eleWT_4do_gh']:
            #    continue
            print(sp, acq)
            bin_ = mat_combine_binary_behaviors(acq_mat) #mat[sp][acq_ix])
            # Get bout starts
            bin_ = mat_split_courtship_bouts(bin_)
            # wing bouts?
            bin_['Unilateral Wing Extensions'] = [1 if (l==1 or r==1) and (l!=r) else 0 for (l, r) \
                                      in bin_[['Left Wing Extensions', 'Right Wing Extensions']].values]
            ori_only = bin_[(bin_[nonorienting_names].eq(0).all(1)) & (bin_['Orienting'])]
            bin_['Orienting Only'] = 0
            bin_['Orienting Only'].loc[ori_only.index] = 1

            #bouts_ =.mat_get_bout_indices(acq_mat) #mat[sp][acq_ix])
            # get features mat
            if feat is not None:
                feat_ = feat[(feat['acquisition']==acq) & (feat['sex']==sex)].copy().reset_index(drop=True)
                bin_ = bin_.loc[0:feat_['frame'].iloc[-1]] # Only grab til copulation index
                assert bin_.shape[0]==feat_.shape[0], "Incorrect shapes for merging: binary evs {} and feat {}".format(bin_.shape, feat_.shape)
                evs_ = pd.merge(bin_, feat_, left_index=True, right_index=True)
            else:
                evs_ = bin_.copy()
            binevs_list.append(evs_)
    events = pd.concat(binevs_list).reset_index()

#    for aq, df_ in events.groupby('acquisition'):
#        dur_dict = get_bout_durs(df_)
#        events.loc[df_.index, 'boutdur'] = [dur_dict[v] for v in df_['boutnum']]

    return events

def add_bout_durations(df):
    # add bout durations
    for aq, df_ in df.groupby('acquisition'):
        dur_dict = get_bout_durs(df_)
        df.loc[df_.index, 'boutdur'] = [dur_dict[v] for v in df_['boutnum']]

    return df

def load_flytracker_data(acq_dir, calib_is_upstream=False, fps=60, 
                         subfolder='*', filter_ori=True):
    '''
    Get calibration info, -feat.mat and -track.mat as DFs.
    If calib_is_upstream, subfolder should be '' -- load_feat and load_trk looks into os.path.join(acq_dir, subfolder, *.mat)
    NOTE: filter_ori NaNs ori if wing position is missing.
    
    Args:
    -----
    acq_dir: (str) directory containing FlyTracker output
    calib_is_upstream: (bool) calibration.mat file is upstream of flytracker output directory (default should prob be true)
    subfolder: (str) subfolder where -feat.mat and -track.mat are stored if NOT in default level of where video .avi file is
    filter_ori: (bool) if True, set ORI to NaN where wing info is missing
    fps: (int) frames per second of video
    
    Returns:
    --------
    calib: 
    trackdf: raw tracking data (e.g., position, orientation, left wing ang)
    featdf: features derived from tracking data (e.g., velocity, dist to x)
    '''
    #% Get corresponding calibration file
    calib=None; trackdf=None; featdf=None;
    try:
        calib = load_calibration(acq_dir, calib_is_upstream=calib_is_upstream)
    except Exception as e:
        print("No calibration! Error: {}".format(e))
        calib={}
        calib['FPS'] = fps

    #% Load feature mat
    try:
        feat_ = load_feat(acq_dir, subfolder=subfolder)
        trk_ = load_tracks(acq_dir, subfolder=subfolder)

        trackdf = add_frame_nums(trk_, fps=calib['FPS'])
        featdf = add_frame_nums(feat_, fps=calib['FPS'])

        if filter_ori:
            # find locs where ORI info can't be trusted
            no_wing_info = trk_[trk_[['wing_l_x', 'wing_l_y', 'wing_r_x', 'wing_r_y']].isna().sum(axis=1) == 4 ].index
            trk_.loc[no_wing_info, 'ori'] = np.nan
    except Exception as e:
        print("ERROR: No TRACKS for {}".format(acq_dir))

    # Estimate w/h from tracking data if missing from calib
    w_ok = 'w' in calib and np.isscalar(calib['w'])
    h_ok = 'h' in calib and np.isscalar(calib['h'])
    if (not w_ok or not h_ok) and trk_ is not None and 'pos_x' in trk_.columns:
        if not w_ok:
            calib['w'] = float(np.nanmax(trk_['pos_x']))
        if not h_ok:
            calib['h'] = float(np.nanmax(trk_['pos_y']))
        print("WARNING: estimated frame size from tracking data: w={}, h={}".format(
            calib['w'], calib['h']))

    return calib, trk_, feat_


def combine_flytracker_data(acq, viddir, subfolder='*', fps=60):
    '''From specified acquisition folder, load_flytracker_data(), combine feat and trk
    
    Args:
    -----
    acq: (str) acquisition name, relevant if more than 1 acquisition in viddir
    viddir: (str) parent dir containing all video files
    subfolder: (str) subfolder where -feat.mat and -track.mat are stored if NOT in default level of where video .avi file is (for ex., 'fly-tracker/*')
    fps: (int) frames per second of video
    
    '''
    # load flytracker .mat as df
    calib_, trk_, feat_ = load_flytracker_data(viddir, 
                                    subfolder=subfolder,
                                    fps=fps)
    # TODO:  fix frame numbering in util.
    featpath = [f for f in feat_['fpath'].unique() if acq in f][0] 
    trkpath = [f for f in trk_['fpath'].unique() if acq in f][0]
    trk_cols = [c for c in trk_.columns if c not in feat_.columns]
    trk_ = trk_[trk_['fpath']==trkpath]
    feat_ = feat_[feat_['fpath']==featpath]
    for i, t_ in trk_.groupby('id'):
        trk_.loc[t_.index, 'frame'] = np.arange(0, len(t_))
    for i, f_ in feat_.groupby('id'):
        feat_.loc[f_.index, 'frame'] = np.arange(0, len(f_))

    # find where we have no wing info, bec ori can't be trusted
    # find where any of the wing columns are NaN:
    no_wing_info = trk_[trk_[['wing_l_x', 'wing_l_y', 'wing_r_x', 'wing_r_y']].isna().sum(axis=1) == 4 ].index
    trk_.loc[no_wing_info, 'ori'] = np.nan

    df_ = pd.concat([trk_[trk_cols], feat_], axis=1).reset_index(drop=True)

    return df_

## aggregate funcs
def aggr_load_feat(savedir, found_sessionpaths=[], create_new=False):

    if not create_new:
        try:
            # try loading saved feat.mat
            pkl_fpath = os.path.join(savedir, 'feat.pkl')
            feat = pd.read_pickle(pkl_fpath)
        except Exception as e:
            traceback.print_exc()
            print("Error loading feat. Creating new.")
            create_new = True
        
    if create_new:
        assert len(found_sessionpaths)>1, "No session paths provided."
        feat = aggr_feat_mats(found_sessionpaths) 

    return feat

def aggr_feat_mats(found_sessionpaths):
    '''
    Cycle through found session paths analyzed with FlyTracker, add some additional meta info, then save as feat DF. Tested and some manual corrections included.

    Args:
    -----
    List of fullpaths to parent dirs of experiments with -feat.mat found.

    Returns:
    --------
    feat (pd.DataFrame)
    '''
    f_list = []; t_list=[];
    for i, acq_dir in enumerate(found_sessionpaths[::-1]):
        acq = os.path.split(acq_dir)[-1]

        if acq == '20240109-1039_fly1_eleWT_4do_sh_eleWT_4do_gh':
            # RERUN, unequal feat and trk sizes
            continue
        elif 'BADTRACKING' in acq:
            continue

        try:
            calib_, trk_, feat_ = load_flytracker_data(acq_dir) #os.path.join(videodir, acq))
        except Exception as e:
            print("ERROR: %s" % acq)
            print(e)
            continue
            
        if 'cop_ind' not in calib_.keys():
            print(acq, 'No cop')
        else:
            print(acq, calib_['cop_ind'])
            
        # get species
        if 'mel' in acq:
            species_abbr = 'mel'
            species_strain = 'wt'
        elif 'suz' in acq:
            species_abbr = 'suz'
            species_strain = 'wt'
        elif 'ele' in acq:
            species_abbr = 'ele'
            species_strain = 'wt'
        elif 'yak' in acq:
            species_abbr = 'yak'
            species_strain = 'wt'
        else:
            if '_fly' in acq:
                species_abbr = acq.split('_')[2]
            else:
                species_abbr = acq.split('_')[1]
            species_strain = 'wt'
            if species_abbr.startswith('mau'):
                species_strain = species_abbr[3:]
                species_abbr = species_abbr[0:3]
            elif species_abbr in ('Canton-S', 'ctns'):
                species_abbr = 'mel'
                species_strain = 'cantons'
        # get age
        age = int(re.findall('(\d{1}do)', acq)[0][0])
#        if '_fly' in acq:
#            age = int(re.sub('\D', '', acq.split('_')[3]))
#        elif len(acq.split('_'))<2:
#            age = None
#        else:
#            age = int(re.sub('\D', '', acq.split('_')[2]))

        # get sex
        cop_ix = calib_['cop_ind'] if calib_['cop_ind']>=1 else feat_['frame'].iloc[-1]
        #ix_male = trk_.groupby('id').mean()['body_area'].idxmax() #.unique()
        #ix_female = trk_.groupby('id')['body_area'].mean().idxmax()
        #if acq in ['20231222-1149_fly2_eleWT_4do_sh_eleWT_4do_gh', '20231223-1212_fly3_eleWT_5do_sh_eleWT_5do_gh']:
        #    ix_male = 1
        if float(trk_[trk_['frame']<=cop_ix].groupby('id')['body_area'].mean().round(1).diff().abs().dropna()) == 0: 
            #float(feat_[feat_['frame']<=cop_ix].groupby('id')['max_wing_ang'].mean().round(1).diff().abs().dropna()) == 0: # difference is super tiny
            #ix_male = trk_[trk_['frame']<=cop_ix].groupby('id')['body_area'].mean().idxmin() # use body size
            ix_male = feat_[feat_['frame']<=cop_ix].groupby('id')['max_wing_ang'].mean().idxmax()
        else:
            #ix_male = feat_[feat_['frame']<=cop_ix].groupby('id')['max_wing_ang'].mean().idxmax()
            ix_male = trk_[trk_['frame']<=cop_ix].groupby('id')['body_area'].mean().idxmin() # use body size
            
    #     if trk_.groupby('id')['minor_axis_len'].mean().round().diff().max() <= 1:
    #         ix_female = trk_.groupby('id')['body_area'].mean().idxmax()
    #     else:
    #         ix_female = trk_.groupby('id')['minor_axis_len'].mean().idxmax() # this seems most reliable
        #assert len(ix_max)==1, "Ambiguous sex based on size: {}".format(trk_.groupby('id').mean()[['major_axis_len', 'body_area', 'minor_axis_len']].idxmax())
        #ix_male = feat_.groupby('id')['max_wing_ang'].mean().idxmax()
        feat_.loc[feat_['id']==ix_male, 'sex'] = 'm'
        feat_.loc[feat_['id']!=ix_male, 'sex'] = 'f'
        
        trk_.loc[trk_['id']==ix_male, 'sex'] = 'm'
        trk_.loc[trk_['id']!=ix_male, 'sex'] = 'f'
       
        print('--', species_abbr, age, feat_['sex'].unique(), 'male ID: {}'.format(feat_[feat_['sex']=='m']['id'].unique()[0]))
        try:
            #male_wg = feat_[(feat_['frame']<=cop_ix) & (feat_['sex']=='m')]['max_wing_ang'].median()
            #female_wg = feat_[(feat_['frame']<=cop_ix) & (feat_['sex']=='f')]['max_wing_ang'].median()
            male_wg = feat_[(feat_['frame']<cop_ix) & (feat_['max_wing_ang']>1.5) & (feat_['sex']=='m')].count()[0]
            female_wg = feat_[(feat_['frame']<cop_ix) & (feat_['max_wing_ang']>1.5) & (feat_['sex']=='f')].count()[0]
            assert male_wg-female_wg > 100, "Male wing is not >> than female. Check: %s" % acq
        except AssertionError as e: # AssertionError as e:
            print(e)   
            
        # update
        feat_['species'] = species_abbr
        feat_['strain'] = species_strain
        feat_['age'] = age
        feat_['acquisition'] = acq
        feat_['copulation_index'] = calib_['cop_ind']
        feat_['copulation'] = calib_['cop_ind']>0
        grab_index = calib_['cop_ind']-1 if calib_['cop_ind']>1 else feat_.iloc[-1].name
        f_ = feat_[feat_['frame']<=grab_index].reset_index(drop=True)
        f_list.append(f_)
        # add trk, too
        trk_['acquisition'] = acq
        t_ = trk_[trk_['frame']<=grab_index].reset_index(drop=True)
        t_list.append(t_)
        #print(f_.shape, t_.shape)
        assert f_.shape[0]==t_.shape[0], "Unequal"
    feat = pd.concat(f_list) #.reset_index(drop=True)
    trk = pd.concat(t_list)

    return feat, trk


# ---------------------------------------------------------------------
# Calculate courtship metrics
# ---------------------------------------------------------------------

def threshold_courtship_bouts(feat0, max_dist_to_other=5, max_facing_angle=30):
    '''
    Set thresholds for identifying courtship bouts using extracted features
    from FlyTracker. Sets thresholds on 'facing_angle' and 'dist_to_other'.
    
    Args:
    -----
    feat: (pd.DataFrame)
        Calculated features from -feat.mat  *assumes 1 fly id only*
    
    max_dist: (float)
        Inter-fly distance at which interaction is considered a bout
        
    max_angle: (float)
        Angle (in degs) that flies are oriented to e/o
    
    Returns:
    --------
    df: (pd.DataFrame)
        Updated dataframe with courtship (bool) as column
                
    '''
    feat = feat0.copy()
    # Convert to deg to make my life easier
    feat['courtship'] = False
    all_nans=[]
    #feat.shape
        
    feat['facing_angle_deg'] = np.rad2deg(feat['facing_angle'])

    # Identify bad flips, where wings flip so fly seems to face opposite dir
    #bad_flips = feat[feat['facing_angle'].diff().abs()>0.5].index.tolist()
    #feat.loc[bad_flips] = np.nan

#        # find all non-consecutive nan indices
#        found_nans = feat[feat['facing_angle'].isna()].index.tolist()
#        non_consecs = np.where(np.diff(found_nans)>1)[0] # each pair of values represents 1 chunk   
#        non_consecs 
#        for i, ix in enumerate(non_consecs[0::2]):
#            curr_ix = list(non_consecs).index(ix)        
#            s_ix = found_nans[ix]         
#            next_ix = non_consecs[curr_ix+1]
#            e_ix = found_nans[next_ix]
#            print(s_ix, e_ix)
#            feat.loc[s_ix:e_ix]=np.nan     
# 
#        # Get all nan indices, and block out in-between frames, too
#        nan_ixs = feat.isna().index.tolist()
#        chunks = []
#        for k, g in groupby(enumerate(nan_ixs), lambda ix: ix[0]-ix[1]):
#            chunks.append(list(map(itemgetter(1), g)))
#        for chunk in chunks:
#            feat.loc[chunk] = np.nan
#
    # Find true facing frames    
    feat.loc[(feat['dist_to_other'] < max_dist_to_other) & (
        feat['facing_angle_deg'] < max_facing_angle), 'courtship'] = True
             
    return feat

def get_true_bouts(feat0, calib, ibi_min_sec=0.5):
    '''
    Group frames that pass threshold for "courtship" into actual bouts.
    Bouts specified arbitrarily.
    
    Args:
    -----
    df: (pd.DataFrame)
        Thresholded dataframe from -feat.mat (output of threshold_courtship_bouts())
    
    calib: (dict)
        From calib.mat output (saved as dict)
    
    ibi_min_sec: (float)
        Min. duration (in sec) to be considered a separate bout.
    '''
    feat = feat0.copy()
    # Get list of all bout chunks
    courtship_ixs = feat[feat['courtship']].index.tolist()
    bouts = []
    for k, g in groupby(enumerate(courtship_ixs), lambda ix: ix[0]-ix[1]):
        bouts.append(list(map(itemgetter(1), g)))
    fps = calib['FPS']

    # Identify likely bout-stop false-alarms (i.e., next "bout" starts immediately after...)
    curr_bout_ix = 0
    combine_these=[curr_bout_ix]
    bouts_to_combine={}
    for i, b in enumerate(bouts[0:-1]):
        ibi = (bouts[i+1][0] - b[-1] ) / fps
        # print(i, ibi)

        if np.round(ibi) <= ibi_min_sec: # check dur of next bout after current one
            if len(combine_these)==0:
                combine_these=[i]
            combine_these.append(i+1)
        else:
            if len(combine_these)==0:
                combine_these=[i]
            bouts_to_combine.update({curr_bout_ix: combine_these})
            curr_bout_ix+=1
            combine_these=[]
#     interbout_sec = np.array([(bouts[i+1][0] - b[-1])/fps for i, b in enumerate(bouts[0:-1])])

#     # identify where interbout is "too short"
#     #ibi_min_sec = 0.5
#     ibi_too_short = np.where(interbout_sec < ibi_min_sec)[0] # indexes into bouts
    
#     # get starting indices for true bouts
#     gap_starts = np.where(np.diff(ibi_too_short)>1)[0]
#     gap_ixs = [ibi_too_short[0]]
#     gap_ixs.extend([ibi_too_short[i+1] for i in gap_starts])

#     # concatenate too-short bouts into full bouts
#     true_bouts=[]
#     for i in gap_ixs:
#         curr_ = bouts[i]
#         curr_.extend(bouts[i+1])
#         true_bouts.append(curr_)
#     interbout_sec = np.array([(true_bouts[i+1][0] - b[-1])/fps for i, b in enumerate(true_bouts[0:-1])])
#     ibi_too_short = np.where(interbout_sec < ibi_min_sec)[0] 
#     assert len(ibi_too_short)==0, "Bad bout concatenation. Found %i too short" % len(ibi_too_short)
    bout_dict={}
    for bout_num, bout_ixs in bouts_to_combine.items():
        curr_ixs = flatten([bouts[i] for i in bout_ixs])
        bout_dict.update({bout_num: curr_ixs})

    # reassign courtship bouts:
    for bout_num, bout_ixs in bout_dict.items():
        start, end = bout_ixs[0], bout_ixs[-1]
        feat.loc[start:end, 'courtship']=True
            
    return feat, bout_dict


# projector functions
from numpy.fft import fft, ifft, fftfreq
def get_fft(df_, fft_var='pos_y', time_var='sec', return_pos=True):

    x_fly = df_[fft_var] - df_[fft_var].mean()
    t_fly =  df_[time_var] #np.arange(0,1,ts)
    fft_fly, all_freqs = calculate_fft(x_fly, t_fly)
    if return_pos:
        freq = all_freqs[np.where(all_freqs >= 0)] 
        amp_fly = calculate_fft_amplitude(fft_fly[np.where(all_freqs >= 0)], t_fly)

    else:
        freq = all_freqs
        amp_fly = calculate_fft_amplitude(fft_fly, t_fly)

    return amp_fly, freq


def calculate_fft(x, t, sr=60):

    # sampling interval
    ts = np.mean(np.diff(t)) #1.0/sr
    npnts =  len(t)
    
    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/sr
#     freq = n/T  # freq mirrorin above nyquist
    freq = fftfreq(npnts, ts)
    
    return X, freq #X, freq

def calculate_fft_amplitude(X, t):

    pwr = 2*np.abs(X)/len(t)   
    return pwr


def smooth_timecourse(in_trace, win_size=42):
    '''
    Don't Use this one
    '''
    #smooth trace
    win_half = int(round(win_size/2))
    trace_pad = np.pad(in_trace, ((win_half, win_half)), 'reflect') # 'symmetric') #'edge')

    smooth_trace = np.convolve(trace_pad, np.ones((win_size,))*(1/float(win_size)),'valid')
     
    return smooth_trace[int(win_half/2):-int(win_half/2)]

###
import scipy.stats as spstats
def groupby_circmeans(d_, circ_vars=['min_wing_ang', 'max_wing_ang',  'angle_between', 'facing_angle']):
    return pd.concat([pd.DataFrame({varname: spstats.circmean(d_[varname], np.pi, 0)}, index=[0]) for varname in circ_vars], axis=1)

def binary_events_to_bouts(events,
        groups = ['species', 'acquisition', 'Disengaged', 'copulation']):
    # Get SUM/MEAN of each binary event per bout
    #groups = ['species', 'acquisition', 'Disengaged', 'copulation']
    bouts = events.groupby(groups, group_keys=True)\
                     .apply(get_bout_durs, return_as_df=True).reset_index()
    grab_cols =  ['All Wing Extensions', 'Putative Tap Events', 
                  'Chasing', 'Licking/Proboscis Ext', 'Copulation Attempts', 'Orienting',
                 'Orienting Only', 'Unilateral Wing Extensions', 'Bilateral Wing Extensions']
    #grab_cols.extend(['species', 'acquisition', 'boutnum'])
    #counts = events.groupby(['acquisition', 'species', 'boutnum']).mean().reset_index()[grab_cols] ## should it be counts? maybe a fraction (fraction of bout)
    counts = events.groupby(['acquisition', 'species', 'boutnum'])[grab_cols].mean().reset_index() ## should it be counts? maybe a fraction (fraction of bout)
    bouts = bouts.merge(counts, on=['acquisition', 'species', 'boutnum'])
    # Merge with FEAT averages 
    feat_vars = ['vel', 'ang_vel', 'mean_wing_length',
               'axis_ratio', 'fg_body_ratio', 'contrast', 'dist_to_wall',
               'dist_to_other', 'leg_dist']
    circ_vars = ['min_wing_ang', 'max_wing_ang',  'angle_between', 'facing_angle']
    # Get average per bout for the other values
    feat_means = events.groupby(['acquisition', 'species', 'boutnum'])[feat_vars].mean().reset_index()
    feat_means_circ = events.groupby(['acquisition', 'species', 'boutnum']).apply(groupby_circmeans).reset_index()
    feat_means = feat_means.merge(feat_means_circ, on=['acquisition', 'species', 'boutnum'])
    #feat_means
    bouts = bouts.merge(feat_means, on=['acquisition', 'species', 'boutnum'])

    return bouts

def custom_filter_events(events):
    # manual
    # acq = '20220130-1143_mauR_4do_sh'
    events.loc[(events['acquisition']=='20220130-1143_mauR_4do_sh')
               & (events['index']>=2304) & (events['index']<=2308), 'Licking Proboscis Ext'] = 1
    events.loc[(events['acquisition']=='20220130-1143_mauR_4do_sh') 
               & (events['index']>=2309) & (events['index']<=2313), 'Copulation Attempts'] = 1

    # acq = '20220128-1516_mauR4_4do_gh'
    # 16464, '20220128-1516_mauR4_4do_gh', uwe, but is grooming
    #16464, '20220128-1516_mauR4_4do_gh', uwe, but is grooming
    # 22590, def tap
    # 23090+ abdomen vibrating?
    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']<=22095), 'Unilateral Wing Extensions'] = 0
    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']<=22095), 'All Wing Extensions'] = 0

    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']>=22630) & (events['index']<=22640), 'Licking/Proboscis Ext'] = 1
    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']>=22685) & (events['index']<=22704), 'Licking/Proboscis Ext'] = 1
    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']>=22720) & (events['index']<=22722), 'Licking/Proboscis Ext'] = 1

    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']>=22640) & (events['index']<=22656), 'Copulation Attempts'] = 1
    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']>=22704) & (events['index']<=22706), 'Copulation Attempts'] = 1

    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']>=22800) & (events['index']<=22905), 'Orienting'] = 1
    events.loc[(events['acquisition']=='20220128-1516_mauR4_4do_gh') 
            & (events['index']>=22095) & (events['index']<=22932), 'Chasing'] = 1

    return events

# %%
def add_stim_hz(df0, n_frames=24000, n_epochs=10, file_grouper='file_name'):
    '''
    Assigns a 'stim_hz' column to the DataFrame df0 based on the number of frames (known epoch lengths based on stimulus)
    
    Args:
    -----
    df0: (pd.DataFrame)
        DataFrame with a 'frame' column, which is used to assign stimulus Hz values.
    n_frames: (int)
        Total number of frames in the dataset.
    n_epochs: (int)
        Number of epochs in the dataset, which is used to calculate the number of frames per epoch.
    file_grouper: (str)
        Column name used to group the DataFrame, typically the file name or acquisition identifier.    
    '''
    n_frames_per_epoch = n_frames/n_epochs
    epoch_starts = np.arange(0, n_frames, n_frames_per_epoch)

    # This is very temporary... 
    # also see dlc.get_step_indices, which attempts to use velocity of dot
    # NOTE: check stimulus Hz values in stimulus generation, there might be 
    # an extra 0 frame in between epochs due to how the rotation is wrapped around
    stimhz_vals = np.array([0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.625, 0.8, 1])
    stimhz_dict = dict((k, v) for k, v in enumerate(stimhz_vals))
    
    #d_['epoch'] = 0
    wstim= []
    for fn, df_ in df0.groupby(file_grouper):
        df_['stim_hz']=np.nan
        for i, v in enumerate(epoch_starts):
            df_.loc[(df_['frame']>=v) & (df_['frame']<v+n_frames_per_epoch), 'stim_hz'] = stimhz_dict[i]
        wstim.append(df_)
    df0 = pd.concat(wstim)

    return df0


def find_video(video_dir, exts=('.avi', '.mp4', '.mov')):
    """Search for a video file in a directory (or as a direct path)."""
    import glob
    for ext in exts:
        matches = glob.glob(os.path.join(video_dir, '*' + ext))
        if matches:
            return matches[0]
    for ext in exts:
        if os.path.isfile(video_dir + ext):
            return video_dir + ext
    return None


def recompute_ang_vel(ori_series, fps=60):
    """Recompute angular velocity from orientation with head-tail flip correction.

    Uses np.unwrap with discont=pi/2 to aggressively correct head-tail flips,
    then applies the FlyTracker-style [1, 2, 1]/4 smoothing kernel.

    Args:
        ori_series: pd.Series of orientation values in radians.
        fps: frames per second (used to scale gradient to rad/s).

    Returns:
        pd.Series of angular velocity in rad/s, same index as input.
    """
    unwrapped = np.unwrap(ori_series.interpolate().ffill().bfill().values,
                          discont=np.pi/2)
    ang_vel = np.gradient(unwrapped) * fps
    ft_kernel = np.array([1, 2, 1]) / 4.0
    ang_vel = np.convolve(ang_vel, ft_kernel, mode='same')
    return pd.Series(ang_vel, index=ori_series.index)


def find_action_snippet(flydf, action='chasing', fps=60, snippet_dur_sec=3):
    """Find the longest contiguous bout of an action and return a snippet.

    Args:
        flydf: DataFrame with 'frame' and the action column.
        action: name of the binary (0/1) column to search for bouts.
        fps: frames per second.
        snippet_dur_sec: max duration of returned snippet in seconds.

    Returns:
        stretch: DataFrame slice of the snippet (with 'sec' column added).
        f_start, f_end: start and end frame numbers.
    """
    flydf = flydf.sort_values('frame').copy()
    flydf['sec'] = flydf['frame'] / fps
    block_col = f'__{action}_block'
    flydf[block_col] = (flydf[action] != flydf[action].shift()).cumsum()
    action_blocks = (flydf[flydf[action] == 1]
                     .groupby(block_col)['frame']
                     .agg(['min', 'max', 'count'])
                     .sort_values('count', ascending=False))
    best = action_blocks.iloc[0]
    f_start = int(best['min'])
    f_end = min(f_start + int(snippet_dur_sec * fps), int(best['max']))
    stretch = flydf[(flydf['frame'] >= f_start) & (flydf['frame'] <= f_end)].copy()
    stretch.drop(columns=[block_col], inplace=True, errors='ignore')
    return stretch, f_start, f_end


def cross_correlation_lag(x, y, fps=60):
    """Cross-correlation between two signals x and y.

    Returns the normalized cross-correlation, lags, lag in frames, and lag in
    seconds.  Lag is defined as how much Y lags relative to X.

    Args:
        x: first signal (array-like).
        y: second signal (array-like).
        fps: frames per second (default 60).

    Returns:
        correlation: normalized cross-correlation array.
        lags: displacement indices for the cross-correlation.
        lag_frames: displacement in frames at peak correlation.
        t_lag: displacement in seconds.
    """
    import scipy.signal as signal
    correlation = signal.correlate(x - np.mean(x), y - np.mean(y), mode="full")
    npoints = len(x)
    ccorr = correlation / (npoints * y.std() * x.std())
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    lag_frames = lags[np.argmax(correlation)]
    t_lag = lag_frames / fps
    return ccorr, lags, lag_frames, t_lag


def shift_variables_by_lag(df, file_grouper='acquisition', lag=2):
    """Shift fly response variables forward by *lag* frames within each group.

    After shifting, the value at time t corresponds to the variable's value
    at t + lag (i.e., the future response aligned to the current stimulus).

    Args:
        df: DataFrame with columns 'ang_vel_fly', 'vel_fly', 'vel', 'ang_vel'.
        file_grouper: column name to group by (default 'acquisition').
        lag: number of frames to shift forward (default 2).

    Returns:
        df with new columns: ang_vel_fly_shifted, vel_fly_shifted,
        ang_vel_abs_shifted, vel_shifted, vel_shifted_abs, ang_vel_shifted.
    """
    df['ang_vel_fly_shifted'] = df.groupby(file_grouper)['ang_vel_fly'].shift(-lag)
    df['vel_fly_shifted'] = df.groupby(file_grouper)['vel_fly'].shift(-lag)
    df['ang_vel_abs_shifted'] = np.abs(df['ang_vel_fly_shifted'])
    df['vel_shifted'] = df.groupby(file_grouper)['vel'].shift(-lag)
    df['vel_shifted_abs'] = np.abs(df['vel_shifted'])
    df['ang_vel_shifted'] = df.groupby(file_grouper)['ang_vel'].shift(-lag)
    return df


def split_theta_error(chase_, theta_error_small=np.deg2rad(10),
                      theta_error_large=np.deg2rad(25)):
    """Label each row's theta_error as 'small' or 'large' based on thresholds.

    Adds an 'error_size' column: 'small' if |theta_error| < *theta_error_small*,
    'large' if |theta_error| > *theta_error_large*, else None.
    """
    chase_['error_size'] = None
    chase_.loc[
        (chase_['theta_error'] < theta_error_small)
        & (chase_['theta_error'] > -theta_error_small),
        'error_size',
    ] = 'small'
    chase_.loc[
        (chase_['theta_error'] > theta_error_large)
        | (chase_['theta_error'] < -theta_error_large),
        'error_size',
    ] = 'large'
    return chase_


def count_chasing_frames(ftjaaba, grouper=['species', 'acquisition'],
                         chase_var='chasing_binary'):
    """Count chasing and total frames per group and compute the fraction.

    Returns:
        chase_counts DataFrame with n_frames_chasing, n_frames_total,
        and frac_frames_chasing columns.
    """
    n_frames_chasing = (
        ftjaaba[ftjaaba[chase_var] > 0]
        .groupby(grouper)['frame'].count()
        .reset_index()
    )
    n_frames_chasing.rename(columns={'frame': 'n_frames_chasing'}, inplace=True)
    n_frames_total = (
        ftjaaba.groupby(grouper)['frame'].count().reset_index()
    )
    n_frames_total.rename(columns={'frame': 'n_frames_total'}, inplace=True)
    chase_counts = n_frames_chasing.merge(n_frames_total, on=grouper, how='left')
    chase_counts['frac_frames_chasing'] = (
        chase_counts['n_frames_chasing'] / chase_counts['n_frames_total']
    )
    return chase_counts
