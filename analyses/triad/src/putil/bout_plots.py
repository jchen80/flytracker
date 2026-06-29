import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import cv2
from adjustText import adjust_text
import glob
import numpy as np
import os
import re
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns


from analyses.triad.src import util as tutil


def _plot_bout_signals_on_axes(axes, actor_rows, candidate_ids, id_colors, lookback_frames, fps):
    '''
    Plot all signals on provided axes. Shared by plot_bout_signals and plot_switch_signals.

    Arguments:
        axes          -- list of 4 axes [te, dte_smooth, dist, relvel_smooth]
        actor_rows    -- prepped actor rows df with _cand_id, _t, and signal columns
        candidate_ids -- sorted list of candidate fly ids
        id_colors     -- dict mapping fly id to color
        lookback_frames -- smoothing window in frames
        fps           -- frames per second
    '''
    actor_rows = actor_rows.sort_values(['_cand_id', 'frame'])

    # compute signals
    actor_rows['_abs_theta'] = actor_rows['theta_error'].abs()
    actor_rows['_abs_theta_dt'] = (actor_rows
                                    .groupby('_cand_id')['_abs_theta']
                                    .transform(lambda x: x.diff() / (1/fps)))
    actor_rows['_abs_theta_dt_smoothed'] = (actor_rows
                                             .groupby('_cand_id')['_abs_theta_dt']
                                             .transform(lambda x: x.rolling(
                                                 window=lookback_frames, min_periods=1).mean()))
    actor_rows['_rel_vel_smoothed'] = (actor_rows
                                        .groupby('_cand_id')['rel_vel']
                                        .transform(lambda x: x.rolling(
                                            window=lookback_frames, min_periods=1).mean()))

    for cand_id in candidate_ids:
        cand_rows = actor_rows[actor_rows['_cand_id'] == cand_id].sort_values('frame')
        color = id_colors.get(cand_id, 'white')
        t = cand_rows['_t'].values
        label = f'fly {cand_id}'

        axes[0].plot(t, np.rad2deg(cand_rows['_abs_theta']),
                     color=color, lw=1.5, label=label)
        axes[1].plot(t, np.rad2deg(cand_rows['_abs_theta_dt_smoothed']),
                     color=color, lw=1.5, label=label)
        axes[2].plot(t, cand_rows['dist_to_other'],
                     color=color, lw=1.5, label=label)
        axes[3].plot(t, cand_rows['_rel_vel_smoothed'],
                     color=color, lw=1.5, label=label)

    axes[0].set_ylabel('|θ error| (°)')
    axes[1].set_ylabel(f'smoothed d|θ error|/dt\n(°/s)')
    axes[2].set_ylabel('dist to other (px)')
    axes[3].set_ylabel(f'smoothed rel vel\n(px/s)')
    axes[0].legend(fontsize=9, loc='upper right')

    return actor_rows  # return with computed signal columns


def _mark_switch_frames(axes, switch_times_t, color='yellow', label='auto switch', ls='--'):
    '''
    Draw a vertical line at each switch time on all axes.

    Arguments:
        axes            -- list of axes
        switch_times_t  -- list of float time values (seconds relative to bout start)

    Keyword Arguments:
        color  -- line color (default: 'yellow')
        label  -- legend label for the first line (default: 'auto switch')
        ls     -- line style (default: '--')
    '''
    for t in switch_times_t:
        for ax in axes:
            ax.axvline(t, color=color, lw=1.5, ls=ls, alpha=0.9, label=label)
        label = None  # only label first line


def plot_bout_signals(df, bout_num, action_col='courtship',
                      fps=60, lookback_sec=1.0,
                      mark_switches=True,
                      save_dir=None, acq=None,
                      id_colors=None, figsize=(14, 12)):
    '''
    Plot all signals for a full bout: |TE|, smoothed d|TE|/dt, dist_to_other, smoothed rel_vel.

    Arguments:
        df       -- single-acquisition transformed tracks df
        bout_num -- bout number to plot

    Keyword Arguments:
        action_col    -- name of action column (default: 'courtship')
        fps           -- frames per second (default: 60)
        lookback_sec  -- smoothing window in seconds (default: 1.0)
        mark_switches -- if True, draw vertical lines at switch frames from
                         '{action_col}_switch' column (default: True)
        save_dir      -- directory to save figure (default: None)
        acq           -- acquisition name for title (default: None)
        id_colors     -- dict mapping fly id to color (default: None)
        figsize       -- figure size (default: (14, 12))

    Returns:
        fig, axes
    '''
    if id_colors is None:
        id_colors = {0: 'dodgerblue', 1: 'tomato', 2: 'limegreen'}

    actor_rows, actor_id, bout_start, bout_end = _get_bout_actor_rows(
        df, bout_num, action_col, fps)
    if actor_rows is None:
        return None, None

    lookback_frames = int(lookback_sec * fps)
    candidate_ids = sorted(actor_rows['_cand_id'].unique())
    bout_start_t = 0
    bout_end_t = (bout_end - bout_start) / fps

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    _plot_bout_signals_on_axes(axes, actor_rows, candidate_ids,
                                id_colors, lookback_frames, fps)
    _shade_bout_axes(axes, bout_start_t, bout_end_t)

    if mark_switches:
        boutnum_col = f'{action_col}_boutnum'
        switch_col  = f'{action_col}_auto_switch'

        # Auto-detected switches (courtship_auto_switch)
        if switch_col in df.columns:
            switch_frames = df[(df['id'] == actor_id) &
                               (df[switch_col] == 1) &
                               (df[boutnum_col] == bout_num)]['frame'].unique()
            switch_times_t = sorted((f - bout_start) / fps for f in switch_frames)
            if switch_times_t:
                _mark_switch_frames(axes, switch_times_t,
                                    color='yellow', label='auto switch', ls='--')

        # Manually annotated switches (switching action column)
        if 'switching' in df.columns:
            manual_frames = df[(df['id'] == actor_id) &
                               (df['switching'] == actor_id) &
                               (df['frame'] >= bout_start) &
                               (df['frame'] <= bout_end)]['frame'].unique()
            manual_times_t = sorted((f - bout_start) / fps for f in manual_frames)
            if manual_times_t:
                _mark_switch_frames(axes, manual_times_t,
                                    color='cyan', label='manual switch', ls=':')

        if any(ax.get_legend_handles_labels()[0] for ax in axes):
            axes[0].legend(fontsize=7, loc='upper right')

    axes[3].set_xlabel('time relative to bout start (s)')
    fig.suptitle(_bout_title(acq, action_col, bout_num, actor_id, bout_start, bout_end, fps),
                 fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir,
                                f'{action_col}_bout{int(bout_num):03d}_signals.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, axes


def _get_bout_actor_rows(df, bout_num, action_col, fps):
    '''Extract and prep actor rows for a given bout. Used by plotting functions.'''
    boutnum_col = f'{action_col}_boutnum'
    bout_rows = df[df[boutnum_col] == bout_num]
    if len(bout_rows) == 0:
        print(f"No rows found for bout {bout_num}")
        return None, None, None, None

    actor_id = int(bout_rows[bout_rows[action_col] == bout_rows['id']].iloc[0]['id'])
    bout_start = int(bout_rows['frame'].min())
    bout_end = int(bout_rows['frame'].max())

    pad = int(fps)
    frame_min = max(0, bout_start - pad)
    frame_max = bout_end + pad

    actor_rows = df[(df['id'] == actor_id) &
                    (df['frame'] >= frame_min) &
                    (df['frame'] <= frame_max)].copy()

    actor_rows['_cand_id'] = (actor_rows['pair']
                               .str.split('_')
                               .apply(lambda x: [int(i) for i in x])
                               .apply(lambda x: [i for i in x if i != actor_id][0]))
    actor_rows['_t'] = (actor_rows['frame'] - bout_start) / fps

    return actor_rows, actor_id, bout_start, bout_end


def _shade_bout_axes(axes, bout_start_t, bout_end_t):
    '''Shade bout region and add reference lines. Used by plotting functions.'''
    for ax in axes:
        ax.axvspan(bout_start_t, bout_end_t, color='#999999', alpha=0.08)
        ax.axvline(bout_start_t, color='white', lw=1, ls='--', alpha=0.5)
        ax.axvline(bout_end_t, color='white', lw=1, ls='--', alpha=0.5)
        ax.axhline(0, color='white', lw=0.5, alpha=0.3)


def _bout_title(acq, action_col, bout_num, actor_id, bout_start, bout_end, fps):
    '''Generate consistent bout title string. Used by plotting functions.'''
    base = f'{acq} — ' if acq else ''
    return (f'{base}{action_col} bout {bout_num} | actor: fly {actor_id}\n'
            f'frames {bout_start}–{bout_end} ({(bout_end-bout_start)/fps:.1f}s)')
