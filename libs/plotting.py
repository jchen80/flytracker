#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : plotting.py
Created        : 2023/03/27 18:57:30
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
import numpy as np
import pandas as pd
import scipy.stats as spstats
import matplotlib as mpl
import pylab as pl
import seaborn as sns

import libs.utils as util
import matplotlib.pyplot as plt
## generic
# ----------------------------------------------------------------------
# Visualization 
# ----------------------------------------------------------------------
def label_figure(fig, fig_id, x=0.01, y=0.98, fontsize=8):
    fig.text(x, y, fig_id, fontsize=fontsize)

def set_sns_style(style='dark', min_fontsize=6):
    font_styles = {
                    'axes.labelsize': min_fontsize+1, # x and y labels
                    'axes.titlesize': min_fontsize+1, # axis title size
                    'figure.titlesize': min_fontsize+4,
                    'xtick.labelsize': min_fontsize, # fontsize of tick labels
                    'ytick.labelsize': min_fontsize,  
                    'legend.fontsize': min_fontsize,
                    'legend.title_fontsize': min_fontsize+1
        }
    for k, v in font_styles.items():
        pl.rcParams[k] = v

    pl.rcParams['axes.linewidth'] = 0.5

    if style=='dark':
        custom_style = {
                    'axes.labelcolor': 'white',
                    'axes.edgecolor': 'white',
                    'grid.color': 'gray',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'text.color': 'white',
                    'axes.facecolor': 'black',
                    'axes.grid': False,
                    'figure.facecolor': 'black',
                    'savefig.facecolor': 'black',
                    'savefig.edgecolor': 'none'}
        custom_style.update(font_styles)

#        pl.rcParams['figure.facecolor'] = 'black'
#        pl.rcParams['axes.facecolor'] = 'black'
        sns.set_style("dark", rc=custom_style)
    elif style == 'white':
        custom_style = {
                    'axes.labelcolor': 'black',
                    'axes.edgecolor': 'black',
                    'grid.color': 'gray',
                    'xtick.color': 'black',
                    'ytick.color': 'black',
                    'text.color': 'black',
                    'axes.facecolor': 'white',
                    'axes.grid': False,
                    'figure.facecolor': 'white'}
        custom_style.update(font_styles)
        sns.set_style('white', rc=custom_style)
    elif style in ('courtship'):
        # Ruta-lab courtship scheme: dark grey (#262626) background, white text/axes.
        custom_style = {
                    'axes.labelcolor': 'white',
                    'axes.edgecolor': 'white',
                    'grid.color': '#555555',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'text.color': 'white',
                    'axes.facecolor': COURTSHIP_BG,
                    'axes.grid': False,
                    'figure.facecolor': COURTSHIP_BG,
                    'savefig.facecolor': COURTSHIP_BG,
                    'savefig.edgecolor': 'none'}
        custom_style.update(font_styles)
        sns.set_style('dark', rc=custom_style)

    pl.rcParams['savefig.dpi'] = 400
    pl.rcParams['figure.figsize'] = [6,4]

    pl.rcParams['svg.fonttype'] = 'none'


# ──────────────────────────────────────────────────────────────────────────
# Ruta-lab courtship color scheme (projector + triad modules)
# ──────────────────────────────────────────────────────────────────────────
COURTSHIP_BG = '#262626'                  # figure/axes background for style='courtship'

# assay-type colors: species × chamber configuration (MMF = 1 male + 2 females
# of the focal pair context; MFF as labelled in the triad/projector pipelines)
ASSAY_TYPE_COLORS = {
    'Dmel_MMF': '#8258E0',
    'Dmel_MFF': '#197FE3',
    'Dyak_MMF': '#CA2661',
    'Dyak_MFF': '#FF6146',
}
# species colors when chamber configuration is not considered
SPECIES_COLORS = {'Dmel': '#197FE3', 'Dyak': '#FF6146'}

OLD_TARGET_COLOR = '#808080'              # medium grey: the old/previous switch target
FOCAL_COLOR      = '#3A1A5E'              # dark purple: the focal male in allo trajectories
NEUTRAL_GREY     = '#808080'              # generic grey (e.g. the 'all frames' baseline)


def _blend(color, target, amount):
    """Blend `color` toward `target` by `amount` in [0,1]; returns a hex string."""
    c = np.array(mpl.colors.to_rgb(color))
    t = np.array(mpl.colors.to_rgb(target))
    return mpl.colors.to_hex((1.0 - amount) * c + amount * t)


def darken(color, amount=0.15):
    """Darken a color by blending it toward black."""
    return _blend(color, 'black', amount)


def lighten(color, amount=0.15):
    """Lighten a color by blending it toward white."""
    return _blend(color, 'white', amount)


# retinal FOV motion, the requested base hues nudged a little darker for contrast
PROGRESSIVE_COLOR = darken('#04AE3D', 0.18)   # progressive (target image moving forward)
REGRESSIVE_COLOR  = darken('#E046D2', 0.18)   # regressive  (target image moving backward)


def species_color(species, default='#444444'):
    """Color for a species label ('Dmel'/'Dyak')."""
    return SPECIES_COLORS.get(str(species), default)


def assay_type_color(label, default=None):
    """Color for an exact assay-type label ('Dmel_MMF', etc.), or `default`."""
    return ASSAY_TYPE_COLORS.get(str(label), default)


def courtship_color(label, default='#444444'):
    """Resolve a group label to its scheme color.

    Tries the exact assay-type key first ('Dmel_MFF'); otherwise falls back to the
    species color if the label starts with a known species ('Dmel', 'Dmel_40',
    'Dyak_inner', ...). Returns `default` if neither matches.
    """
    s = str(label)
    if s in ASSAY_TYPE_COLORS:
        return ASSAY_TYPE_COLORS[s]
    for sp, c in SPECIES_COLORS.items():
        if s.startswith(sp):
            return c
    return default


def color_gradient(base, n, light=0.62, dark=0.34):
    """`n` shades along `base`, ordered low → high (light tint → dark shade).

    Use for an ordered numeric hue (LED intensity, stimulus speed) drawn in a
    single species/assay color. The low end is `base` lightened toward white, the
    high end `base` darkened toward black, passing through `base` near the middle.
    """
    if n <= 1:
        return [base]
    out = []
    for i in range(n):
        f = i / (n - 1)                  # 0 (low) .. 1 (high)
        if f <= 0.5:
            out.append(_blend(base, 'white', light * (1 - 2 * f)))
        else:
            out.append(_blend(base, 'black', dark * (2 * f - 1)))
    return out


def prog_regr_cmap():
    """Diverging colormap: regressive (magenta) → light → progressive (green)."""
    return mpl.colors.LinearSegmentedColormap.from_list(
        'prog_regr', [REGRESSIVE_COLOR, '#F2F2F2', PROGRESSIVE_COLOR])


def gradient_cmap(base, light=0.55, dark=0.34):
    """Sequential colormap light tint → `base` → dark shade (for an ordered hue
    such as LED intensity or stimulus speed, drawn in one species/assay color)."""
    return mpl.colors.LinearSegmentedColormap.from_list(
        'grad', [lighten(base, light), base, darken(base, dark)])


# ticks
def change_spine_color(ax, color, spine='left'):
    ax.spines[spine].set_color(color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis='y', colors=color)

def set_outward_spines(ax):
    ax.tick_params(which='both', axis='both', length=2, width=0.5, color='w',
               direction='out', left=True)

def remove_spines(ax, axes=['right', 'top']):
    for pos in axes:
       ax.spines[pos].set_visible(False)


def vertical_scalebar(ax, leg_xpos=0, leg_ypos=0, leg_scale=100, leg_unit='mm', 
                      color='w', lw=1,fontsize=6, offset=2):
    #leg_xpos=0; leg_ypos=round(df0.loc[odor_ix]['ft_posy']); leg_scale=100
    ax.plot([leg_xpos, leg_xpos], [leg_ypos, leg_ypos+leg_scale], color, lw=lw)
    
    ax.text(leg_xpos-offset, leg_ypos+(leg_scale/2), '{} {}'.format(leg_scale, leg_unit), 
            fontsize=fontsize, horizontalalignment='right',
            verticalalignment='center')
    #ax.text(leg_xpos, leg_ypos + (leg_scale/2), '{} {}'.format(leg_scale, leg_unit),
    #ax.axis('off')

def horizontal_scalebar(ax, leg_xpos=0, leg_ypos=0, leg_scale=100, 
                        color='w', lw=1,fontsize=6, leg_unit='mm', offset=2):
    #leg_xpos=0; leg_ypos=round(df0.loc[odor_ix]['ft_posy']); leg_scale=100
    ax.plot([leg_xpos, leg_xpos+leg_scale], [leg_ypos, leg_ypos], color, lw=lw)
    
    ax.text(leg_xpos + (leg_scale/2), leg_ypos - offset, 
            '{} {}'.format(leg_scale, leg_unit), 
            fontsize=fontsize, horizontalalignment='center')
    #ax.axis('off')


def custom_legend(labels, colors, use_line=True, lw=4, markersize=10):
    '''
    Returns legend handles

    Arguments:
        labels -- _description_
        colors -- _description_

    Keyword Arguments:
        use_line -- _description_ (default: {True})
        lw -- _description_ (default: {4})
        markersize -- _description_ (default: {10})

    Returns:
        _description_
    '''
    if use_line:
        legh = [mpl.lines.Line2D([0], [0], color=c, label=l, lw=lw) for c, l in zip(colors, labels)]
    else:
        legh = [mpl.lines.Line2D([0], [0], marker='o', color='w', label=l, lw=0,
                    markerfacecolor=c, markersize=markersize) for c, l in zip(colors, labels)]

    return legh

#%% Stats-related functions for plotting

def annotate_regr(data, ax, x='facing_angle', y='ang_vel', 
                  fontsize=8, xloc=.05, yloc=.8, **kws):
    '''
    Do pearson corr and annotate plot in upper left with p and r values.

    Argum:
        data -- _description_
        ax -- _description_

    Keyword Arguments:
        x -- _description_ (default: {'facing_angle'})
        y -- _description_ (default: {'ang_vel'})
        fontsize -- _description_ (default: {8})
    '''
    import scipy.stats as spstats

    r, p = spstats.pearsonr(data[x].interpolate().ffill().bfill(), 
                            data[y].interpolate().ffill().bfill())
    ax.text(xloc, yloc, 'pearson r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes, fontsize=fontsize)

    return (r, p)

#%%

def plot_array_of_trajectories(trajdf, sorted_eff=[], nr=5, nc=7, 
                            aspect_ratio=0.5, sharey=True,
                            bool_colors=['r'], bool_vars=['instrip'], title='filename',
                            notable=[]):

    if len(sorted_eff)==0:
        sorted_eff = sorted(trajdf['filename'].unique(), key=util.natsort)


    maxy = trajdf['ft_posy'].max() if not sharey else 1600
     
    fig, axn = pl.subplots(nr, nc, figsize=(15,8), sharex=True)
    for fi, fn in enumerate(sorted_eff): #(fn, df_) in enumerate(etdf.groupby('filename')):
        if fi >= nr*nc:
            break
        ax=axn.flat[fi]
        df_ = trajdf[trajdf['filename']==fn].copy()
        #eff_ix = float(mean_tortdf[mean_tortdf['filename']==fn]['efficiency_ix'].unique())
        # PLOT
        plot_zeroed_trajectory(df_, ax=ax, traj_lw=1.5, odor_lw=1.0,
                                     strip_width=50, #params[fn]['strip_width'],
                                     strip_sep=1000, #) #params[fn]['strip_sep'])
                                bool_colors=bool_colors,
                                bool_vars=bool_vars)
        # legend
        ax.axis('off')
        if fi==0:
            leg_xpos=-150; leg_ypos=0; leg_scale=100
            vertical_scalebar(ax, leg_xpos=leg_xpos, leg_ypos=leg_ypos)
        #ax.set_box_aspect(3)
        ax.set_xlim([-150, 150])
        ax.set_ylim([-100, maxy])
        ax.axis('off')
        ax.set_aspect(aspect_ratio)
        if title=='filename':
            ax_title = fn
        else:
            ax_title = fn.split('_')[0] # use datetime str 
        if fn in notable:
            ax.set_title('{}:\n*{}'.format(fi, ax_title), fontsize=6, loc='left')
        else:
            ax.set_title('{}:\n{}'.format(fi, ax_title), fontsize=6, loc='left')

    for ax in axn.flat[fi:]:
        ax.axis('off')
        #ax.set_aspect(0.5)
        
    pl.tight_layout()
    pl.subplots_adjust(top=0.85, hspace=0.4, wspace=0.5) #left=0.1, right=0.9, wspace=1, hspace=1, bottom=0.1, top=0.8)
    return fig


# circular stuff

def add_colorwheel(fig, cmap='hsv', axes=[0.8, 0.8, 0.1, 0.1], 
                   theta_range=[-np.pi, np.pi], deg2plot=None,
                    flip_theta=False, plot_ring=False, north=True):

    '''
    Assumes values go from 0-->180, -180-->0. (radians).
    ''' 
    display_axes = fig.add_axes(axes, projection='polar')
    #display_axes._direction = max(theta_range) #2*np.pi ## This is a nasty hack - using the hidden field to 
                                      ## multiply the values such that 1 become 2*pi
                                      ## this field is supposed to take values 1 or -1 only!!
    #norm = mpl.colors.Normalize(0.0, 2*np.pi)
    norm = mpl.colors.Normalize(theta_range[0], theta_range[1])

    # Plot the colorbar onto the polar axis
    # note - use orientation horizontal so that the gradient goes around
    # the wheel rather than centre out
    quant_steps = 2056
    cb = mpl.colorbar.ColorbarBase(display_axes, cmap=mpl.cm.get_cmap(cmap, quant_steps),
                                       norm=norm, orientation='horizontal')
    # aesthetics - get rid of border and axis labels                                   
    cb.outline.set_visible(False)                                 
    #display_axes.set_axis_off()
    #display_axes.set_rlim([-1,1])
    if deg2plot is not None:
        display_axes.plot([0, deg2plot], [0, 1], 'k')
   
    if north: 
        display_axes.set_theta_zero_location('N')
    if flip_theta:
        display_axes.set_theta_direction(-1)  # theta increasing clockwise
    if plot_ring:
        display_axes.set_rlim([-1,1])


    return display_axes

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, 
                    edgecolor='w', facecolor=[0.7]*3, alpha=0.7, lw=0.5,
                    theta_zero_location='N', theta_direction=-1):
    """
    Produce a circular histogram of angles on ax.
    From: https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    # x = (x+np.pi) % (2*np.pi) - np.pi
    # Force bins to partition entire circle
    if not gaps:
        #bins = np.linspace(-np.pi, np.pi, num=bins+1)
        bins = np.linspace(0, 2*np.pi, num=bins+1)
    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)
    # Compute width of each bin
    widths = np.diff(bins)
    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n
    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, width=widths, #align='edge', 
                     edgecolor=edgecolor, fill=True, linewidth=lw, facecolor=facecolor,
                    alpha=alpha)
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
        #ax.tick_params(which='both', axis='both', size=0)
    ax.set_theta_zero_location(theta_zero_location) #'N')
    ax.set_theta_direction(theta_direction)  # theta increasing clockwise
     
    return n, bins, patches

def add_colorbar(fig, ax, vmin, vmax, shrink=0.1, pad=0.2, 
                 cmap=mpl.cm.viridis, label=''):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #cmap = mpl.cm.Greys # 
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)


    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                shrink=shrink, label=label, pad=pad)
    return

def colorbar_from_mappable(ax, norm, cmap, hue_title='', axes=[0.85, 0.3, 0.01, 0.4],
                            fontsize=7, ticks=None, ticklabels=None): #pad=0.05):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = ax.figure
    #ax.legend_.remove()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=pad)
    cax = fig.add_axes(axes) 
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax) #ax=ax)
    cbar.ax.set_ylabel(hue_title, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    if ticks is not None:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    #pl.colorbar(im, cax=cax)

def get_palette_dict(df, huevar, cmap='viridis'):
    '''
    Create dict of color values for plotting huevar in df. Expects discrete values.

    Arguments:
        df -- _description_
        huevar -- _description_

    Keyword Arguments:
        cmap -- _description_ (default: {'viridis'})

    Returns:
        _description_
    '''
    bin_vals = sorted(df[huevar].unique())
    return dict((k, v) for k, v in zip(bin_vals, 
                        sns.color_palette(cmap, n_colors=len(bin_vals))) )


#%%
def plot_vector_path(ax, x, y, c, scale=1.5, width=0.005, headwidth=5, pivot='tail', 
                    colormap=mpl.cm.plasma, vmin=None, vmax=None, hue_title='',
                    axes=[0.8, 0.3, 0.01, 0.4]):
    if vmin is None:
        #vmin, vmax = b_[hue_param].min(), b_[hue_param].max()
        vmin, vmax = c.min(), c.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #colors=b_[hue_param]

#     uu = b_['ft_posx'].shift(periods=-1) - b_['ft_posx']
#     vv = b_['ft_posy'].shift(periods=-1) - b_['ft_posy']
#     ax.quiver(b_['ft_posx'].values, b_['ft_posy'].values, uu, vv, color=colormap(norm(colors)), 
#               angles='xy', scale_units='xy', scale=1.5)
    uu = np.roll(x, -1) - x # b_['ft_posx']
    vv = np.roll(y, -1) - y #b_['ft_posy'].shift(periods=-1) - b_['ft_posy']
    uu[-1]=np.nan
    vv[-1]=np.nan
    ax.quiver(x, y, uu, vv, color=colormap(norm(c)), 
              angles='xy', scale_units='xy', scale=scale, pivot=pivot,
              width=width, headwidth=headwidth)
    colorbar_from_mappable(ax, norm, cmap=colormap, hue_title=hue_title, axes=axes)
    return ax

# DEEPLABCUT plotting
def plot_bodyparts(df, frame_ix, bodyparts2plot, ax=None, pcutoff=0.9, 
                    color='r', alpha=1.0, markersize=6):
    if ax is None:
        fig, ax = pl.subplots()
    for bpindex, bp in enumerate(bodyparts2plot):
        prob = df.xs(
            (bp, "likelihood"), level=(-2, -1), axis=1
        ).values.squeeze()
        mask = prob < pcutoff # confident predictions have likelihood > pcutoff
        temp_x = np.ma.array(
            df.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask)
        temp_y = np.ma.array(
            df.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask)
        ax.plot(temp_x[frame_ix], temp_y[frame_ix], ".", color=color, alpha=alpha, markersize=markersize)


def plot_skeleton(coords, inds=None, ax=None, color='k', alpha=1, lw=1):
    if ax is None:
        fig, ax = pl.subplots()
    # abdomen lines
    segs = coords[tuple(zip(*tuple(inds))), :].swapaxes(0, 1) if inds else []
    coll = mpl.collections.LineCollection(segs, colors=color, alpha=alpha, lw=lw)
#         # og skeleton
#         segs0 = coords[tuple(zip(*tuple(inds_og))), :].swapaxes(0, 1) if inds_og else []
#         col0 = mpl.collections.LineCollection(segs0, colors=skeleton_color, alpha=alphavalue, lw=skeleton_lw)
    # plot
    ax.add_collection(coll)
#         ax.add_collection(col0)


def plot_ethogram(positions, ev, plot_behaviors=[], behavior_colors=[], nonbinary_behavs=[],
                bg_color=[0.7]*3, lw=0.05):
    import matplotlib.gridspec as gridspec

    if len(behavior_colors)==0:
        behavior_colors = sns.color_palette('cubehelix', 
                            n_colors=len(positions)+1, desat=1)[1:]
    if len(plot_behaviors)==0:
        plot_behaviors = np.arange(0, len(positions))
 
    fig = pl.figure(figsize=(8, len(nonbinary_behavs)*1.5)) #, constrained_layout=True)
    fig.patch.set_facecolor('k') #fig.patch.set_alpha(0)
    spec = gridspec.GridSpec(ncols=1, nrows=len(nonbinary_behavs)+2, figure=fig)
    ax0 = fig.add_subplot(spec[0:2, 0]) # video

    ax0.eventplot(positions, orientation='horizontal', 
                  colors=behavior_colors, linewidths=lw, lineoffsets=1.25)
    xlim = ax0.get_xlim()[-1]
    for ai, (label, color) in enumerate(zip(plot_behaviors, behavior_colors)):
        print(ai, label)
        ax0.text(xlim, ai*1.25, label, color=color, va='center')

    for ai, behav in enumerate(nonbinary_behavs):
        #print(ai, behav)
        ax_ = fig.add_subplot(spec[ai+2, 0]) #, sharex=ax0)  # 2p
        ax_.plot(ev['Time Vector (s)'], ev[behav], color=bg_color, lw=0.5)
        ax_.set_ylabel(behav, fontsize=8)
        if behav=='leg_dist':
            ax_.set_ylim([0, 18])
        elif behav=='vel':
            ax_.set_ylim([0, 75])
        elif 'angle' in behav:
            ax_.set_ylim([0, 3.2])
    for ai, ax in enumerate(fig.axes):
        if ai == 3:
            ax.set_xlabel('time (s)')
            ax.set_xticklabels(ax_.get_xticks()[0:])
        else: 
            ax.set_xlabel('')
            ax.set_xticklabels([])
    pl.subplots_adjust(left=0.1, right=0.8, top=0.9)
    return fig

def add_colored_lines(b_, ax, xvar='ft_posx', yvar='ft_posy', 
                    hue_var='heading', cmap='hsv', norm=None,
                    lw=1, alpha=1):
    '''
    Plot lines between x, y points in bout, b_, with specified colors.

    Arguments:
        b_ -- _description_
        ax -- _description_

    Keyword Arguments:
        xvar -- _description_ (default: {'ft_posx'})
        yvar -- _description_ (default: {'ft_posy'})
        hue_var -- _description_ (default: {'heading'})
        cmap -- _description_ (default: {'hsv'})
        norm -- _description_ (default: {None})

    Returns:
        ax
    '''
    #if norm is None:
    #    mpl.colors.Normalize(theta_range[0], theta_range[1])
    assert norm is not None, "Must provide norm"
    xy = b_[[xvar, yvar]].values
    xy = xy.reshape(-1, 1, 2)
    huev = b_[hue_var].values
    #print(huev.dtype)
    segments = np.hstack([xy[:-1], xy[1:]])
    coll = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm,
                                lw=lw, alpha=alpha) #plt.cm.gist_ncar)
    coll.set_array(huev) #np.random.random(xy.shape[0]))
    ax.add_collection(coll)
    return ax


#%%
def plot_grouped_boxplots(mean_, palette='PRGn', 
                            between_group_spacing=1.5, 
                            within_group_spacing=0.5, box_width=0.3,
                            grouper='species', lw=0.5,
                            x='strain_name', y='vel', ax=None,
                            edgecolor='black'):
    '''
    Seaborn's box plot doesn't allow custom spacing (no gap functionality). 
    Custom function to plot boxplots with custom spacing.
    
    Args:
        mean_ (pd.DataFrame): Dataframe with mean values for each group
        palette (str): Seaborn color palette
        group_spacing (float): Spacing between groups
        x_spacing (float): Spacing between boxes within a group
        grouper (str): Column name to group by
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        ax (matplotlib.axes.Axes): Axes to plot on, if None create new figure and axes 
    '''
    if ax is None:
        fig, ax = plt.subplots()
        
    species_order = mean_[grouper].unique()
    strain_order = mean_[x].unique()
    strain_palette = sns.color_palette(palette, n_colors=len(strain_order))
    strain_colors = dict(zip(strain_order, strain_palette))
    # Set spacing
    #group_spacing = 1.5
    #x_spacing = 0.2
    # Set positions
    positions = {}
    x_ticks = []
    x_tick_labels = []
    x_pos = 0
    for species in species_order:
        strains = mean_[mean_[grouper] == species][x].unique()
        n = len(strains)
        # Center strains around the group midpoint
        offsets = [(i - (n - 1) / 2) * within_group_spacing for i in range(n)] 
        for i, strain in enumerate(strains):
            positions[(species, strain)] = x_pos + offsets[i] #i * within_group_spacing
        x_ticks.append(x_pos) #x_pos + (len(strains) - 1) * within_group_spacing / 2)
        x_tick_labels.append(species)
        x_pos += between_group_spacing  # move to next species group
    # Plot
    for (species, strain), pos in positions.items():
        data = mean_[(mean_[grouper] == species) & (mean_[x] == strain)][y]
        ax.boxplot(data, positions=[pos], widths=box_width, patch_artist=True,
                boxprops=dict(facecolor=strain_colors[strain], 
                              edgecolor=edgecolor,
                              linewidth=lw),
                whiskerprops=dict(color=edgecolor, linewidth=lw),
                capprops=dict(color=edgecolor, linewidth=lw),
                medianprops=dict(color=edgecolor), 
                flierprops=dict(marker='o', markersize=0, color=edgecolor))
    # Axis labels and legend
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels) 
    # Legend
#     if ai==2:
#         for strain in strain_order:
#             ax.plot([], [], color=strain_colors[strain], label=strain, 
#                     linewidth=5)
#         ax.legend(title='Strain', bbox_to_anchor=(1.05, 1), loc='upper left')
    return ax


# ----------------------------------------------------------------------
# Video / trajectory overlay utilities
# ----------------------------------------------------------------------
def load_video_frames(video_path, frame_indices):
    """Load specific frames from a video file.

    Returns an (N, H, W, 3) float32 array of RGB frames.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    for fr in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames, dtype=np.float32)


def composite_silhouettes(frame_stack, bg_threshold=15):
    """Create a time-weighted silhouette composite from stacked video frames.

    Pixels darker than the mean background (by bg_threshold) are stamped
    with increasing opacity so early positions are faint and later ones
    are bold.

    Args:
        frame_stack: (N, H, W, 3) float32 array from load_video_frames.
        bg_threshold: intensity difference to detect the fly vs background.

    Returns:
        (H, W, 3) uint8 composite image.
    """
    bg = frame_stack.mean(axis=0)
    composite = bg.copy()
    n = len(frame_stack)
    for i in range(n):
        alpha = 0.15 + 0.85 * (i / max(n - 1, 1))
        fly_mask = frame_stack[i] < (bg - bg_threshold)
        composite[fly_mask] = ((1 - alpha) * composite[fly_mask]
                               + alpha * frame_stack[i][fly_mask])
    return np.clip(composite, 0, 255).astype(np.uint8)


def plot_orientation_arrows(ax, x, y, ori, t_norm, arrow_len=20,
                            cmap='cool', width=0.005, **kwargs):
    """Plot quiver arrows colored by a time-normalized variable.

    Args:
        ax: matplotlib Axes.
        x, y: 1-D arrays of positions.
        ori: 1-D array of orientations (radians).
        t_norm: 1-D array in [0, 1] for colormap mapping.
        arrow_len: length of each arrow in data units.
        cmap: colormap name.
        width: arrow shaft width (fraction of axes width).
        **kwargs: forwarded to ax.quiver.
    """
    u = arrow_len * np.cos(ori)
    v = arrow_len * np.sin(ori)
    kw = dict(scale=1, scale_units='xy', angles='xy',
              headwidth=3, headlength=4, zorder=3)
    kw.update(kwargs)
    return ax.quiver(x, y, u, v, t_norm, cmap=cmap, width=width, **kw)


def plot_video_overlay(ax, video_path, frame_indices, flydf=None, targdf=None,
                       f_start=None, f_end=None, arrow_len=20,
                       fly_cmap='cool', targ_cmap='autumn',
                       bg_threshold=15, targ_size=6):
    """Plot a silhouette composite with orientation arrows and target dots.

    Args:
        ax: matplotlib Axes.
        video_path: path to the video file.
        frame_indices: list of frame numbers to load.
        flydf: DataFrame with columns 'frame', 'pos_x', 'pos_y', 'ori'
               (only rows matching frame_indices are plotted).
        targdf: DataFrame with 'frame', 'pos_x', 'pos_y' for the target.
        f_start, f_end: frame range for time-normalization.
        arrow_len: length of orientation arrows.
        fly_cmap, targ_cmap: colormaps for fly and target.
        bg_threshold: passed to composite_silhouettes.
        targ_size: marker size for target dots.

    Returns:
        composite image (H, W, 3) uint8.
    """
    frame_stack = load_video_frames(video_path, frame_indices)
    composite = composite_silhouettes(frame_stack, bg_threshold=bg_threshold)
    h, w = composite.shape[:2]
    ax.imshow(composite, extent=[0, w, h, 0])

    if f_start is None:
        f_start = frame_indices[0]
    if f_end is None:
        f_end = frame_indices[-1]
    f_range = max(f_end - f_start, 1)

    if flydf is not None:
        fdf = flydf[flydf['frame'].isin(frame_indices)]
        t_norm = (fdf['frame'].values - f_start) / f_range
        plot_orientation_arrows(ax, fdf['pos_x'].values, fdf['pos_y'].values,
                                fdf['ori'].values, t_norm,
                                arrow_len=arrow_len, cmap=fly_cmap)

    if targdf is not None:
        tdf = targdf[targdf['frame'].isin(frame_indices)]
        t_norm_t = (tdf['frame'].values - f_start) / f_range
        ax.scatter(tdf['pos_x'], tdf['pos_y'],
                   c=t_norm_t, cmap=targ_cmap, s=targ_size, zorder=3)

    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    return composite


def colored_line(ax, x, y, c, cmap, lw=1.2):
    """Draw a line colored by array *c* using LineCollection."""
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mpl.collections.LineCollection(segments, cmap=cmap,
                                        norm=plt.Normalize(0, 1))
    lc.set_array(c[:-1])
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


# ----------------------------------------------------------------------
# Diagnostic plotting functions
# ----------------------------------------------------------------------
def diagnostics_plot_timecourses(stretch, f_start, f_end, example_lag=6,
                                 fps=60, bg_color='gray', title=None):
    """Plot time-course diagnostic: target pos, theta error, ori, ang_vel comparison.

    Args:
        stretch: DataFrame snippet with required columns.
        f_start, f_end: frame range.
        example_lag: lag in frames for shifted column name.
        fps: frames per second.
        bg_color: color for reference lines.
        title: optional plot title override.

    Returns:
        fig, axes
    """
    sec_vals = stretch['sec'].values
    t_norm = (stretch['frame'].values - f_start) / max(f_end - f_start, 1)

    stretch = stretch.copy()
    if 'theta_error_deg' not in stretch.columns:
        stretch['theta_error_deg'] = np.rad2deg(stretch['theta_error'])
    if 'targ_pos_theta_deg' not in stretch.columns:
        stretch['targ_pos_theta_deg'] = np.rad2deg(stretch['targ_pos_theta'])
    if 'ang_vel_fly_deg' not in stretch.columns:
        stretch['ang_vel_fly_deg'] = np.rad2deg(stretch['ang_vel_fly'])
    if 'ang_vel_deg' not in stretch.columns:
        stretch['ang_vel_deg'] = -1 * np.rad2deg(stretch['ang_vel'])
    if 'ori_deg' not in stretch.columns:
        stretch['ori_deg'] = np.rad2deg(stretch['ori'])

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    colored_line(axes[0], sec_vals, stretch['targ_pos_theta_deg'].values,
                 t_norm, 'autumn')
    axes[0].set_ylabel('Target position (deg)')
    if title:
        axes[0].set_title(title)

    colored_line(axes[1], sec_vals, stretch['theta_error_deg'].values,
                 t_norm, 'cool')
    axes[1].set_ylabel('Theta error (deg)')
    axes[1].axhline(0, color=bg_color, ls='--', lw=0.5)

    colored_line(axes[2], sec_vals, stretch['ori_deg'].values,
                 t_norm, 'cool', lw=1)
    axes[2].set_ylabel('Orientation (deg)')

    axes[3].plot(stretch['sec'], stretch['ang_vel_deg'],
                 color='lime', lw=1, alpha=0.8, label='ang_vel (FlyTracker)')
    axes[3].plot(stretch['sec'], stretch['ang_vel_fly_deg'],
                 color='deepskyblue', lw=1, alpha=0.8, label='ang_vel_fly')
    axes[3].set_ylabel('Ang vel (deg/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].axhline(0, color=bg_color, ls='--', lw=0.5)
    axes[3].legend(frameon=False)
    plt.tight_layout()
    return fig, axes


def diagnostics_plot_timecourses_zoom(stretch, zoom_offset=1.0, zoom_dur=1.0,
                                      bg_color='gray'):
    """Plot a zoomed time-course showing individual frames.

    Args:
        stretch: DataFrame snippet with 'sec', 'ori_deg', 'ang_vel_fly_deg',
                 'ang_vel_deg' columns.
        zoom_offset: seconds from snippet start to begin zoom window.
        zoom_dur: duration of zoom window in seconds.
        bg_color: color for reference lines.

    Returns:
        fig, (ax_ori, ax_vel)
    """
    zoom_start = stretch['sec'].iloc[0] + zoom_offset
    zoom_end = zoom_start + zoom_dur
    zoom = stretch[(stretch['sec'] >= zoom_start) & (stretch['sec'] <= zoom_end)]
    zoom_sec = zoom['sec'].values

    fig, (ax_ori, ax_vel) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax_ori.plot(zoom_sec, zoom['ori_deg'], '-o', color='cyan', lw=1, ms=3)
    ax_ori.set_ylabel('Orientation (deg)')
    ax_ori.set_title(f'Zoomed: {zoom_start:.1f}\u2013{zoom_end:.1f}s  (dots = individual frames)')

    ax_vel.plot(zoom_sec, zoom['ang_vel_fly_deg'], '-o', color='deepskyblue',
                lw=1, ms=3, label='ang_vel_fly')
    ax_vel.plot(zoom_sec, zoom['ang_vel_deg'], '-o', color='lime',
                lw=1, ms=3, label='ang_vel (FT, sign-flipped)')
    ax_vel.axhline(0, color=bg_color, ls='--', lw=0.5)
    ax_vel.set_ylabel('Ang vel (deg/s)')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.legend(frameon=False)
    plt.tight_layout()
    return fig, (ax_ori, ax_vel)


def diagnostics_plot_2d_traj_and_rel(stretch, stretch_targ, f_start, f_end,
                                     arrow_len=20, arrow_every=None,
                                     fly_cmap='cool', targ_cmap='autumn',
                                     title=None):
    """Plot 2D trajectory with orientation arrows and relative target position.

    Args:
        stretch: fly DataFrame with 'frame', 'pos_x', 'pos_y', 'ori',
                 'targ_rel_pos_x', 'targ_rel_pos_y'.
        stretch_targ: target DataFrame with 'frame', 'pos_x', 'pos_y'.
        f_start, f_end: frame range for time normalization.
        arrow_len: quiver arrow length.
        arrow_every: subsample rate (None = auto).
        fly_cmap, targ_cmap: colormaps.
        title: optional suptitle.

    Returns:
        fig, (ax_traj, ax_rel)
    """
    f_range = max(f_end - f_start, 1)
    t_norm = (stretch['frame'].values - f_start) / f_range
    t_norm_targ = (stretch_targ['frame'].values - f_start) / f_range

    if arrow_every is None:
        arrow_every = max(1, len(stretch) // 40)
    s_sub = stretch.iloc[::arrow_every]
    t_sub = t_norm[::arrow_every]

    fig, (ax_traj, ax_rel) = plt.subplots(1, 2, figsize=(12, 5))

    plot_orientation_arrows(ax_traj, s_sub['pos_x'].values,
                            s_sub['pos_y'].values, s_sub['ori'].values,
                            t_sub, arrow_len=arrow_len, cmap=fly_cmap,
                            width=0.006, zorder=2)
    ax_traj.scatter(stretch_targ['pos_x'], stretch_targ['pos_y'],
                    c=t_norm_targ, cmap=targ_cmap, s=4, zorder=2, label='target')
    ax_traj.plot(stretch['pos_x'].iloc[0], stretch['pos_y'].iloc[0],
                 'o', color='lime', ms=8, zorder=3, label='start')
    ax_traj.set_xlabel('x (px)')
    ax_traj.set_ylabel('y (px)')
    ax_traj.invert_yaxis()
    ax_traj.set_aspect('equal')
    ax_traj.legend(frameon=False, markerscale=2)
    ax_traj.set_title('2D trajectories')

    # Rotate so heading (forward) points north: data x -> plot y, data y -> plot x
    ax_rel.scatter(stretch['targ_rel_pos_y'], stretch['targ_rel_pos_x'],
                   c=t_norm, cmap=targ_cmap, s=6, zorder=2)
    _arrow_len = stretch['targ_rel_pos_x'].std() * 0.5
    ax_rel.plot([0, 0], [0, _arrow_len], '-', color='cyan', lw=1.5, zorder=3)
    ax_rel.plot(0, _arrow_len, 'o', color='cyan', ms=8, zorder=4, label='fly head (0\u00b0)')
    ax_rel.set_xlabel('Left / Right')
    ax_rel.set_ylabel('Forward (heading direction)')
    ax_rel.set_aspect('equal')
    ax_rel.legend(frameon=False)
    ax_rel.set_title('Target relative to fly')

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig, (ax_traj, ax_rel)


def annotate_axis(ax, annot_str, fontsize=6, color='k'):
    """Place a centered annotation near the top of an axes."""
    ax.annotate(annot_str, xy=(0.5, 0.95), xycoords='axes fraction',
                fontsize=fontsize, ha='center', va='center', color=color)


def plot_ang_v_fwd_vel_by_theta_error_size(
    chase_, var1='vel_shifted', var2='ang_vel_shifted',
    lw=1, err_palette=None,
    figsize=(10, 4), fly_marker='o', fly_marker_size=5, fly_color='gray',
    median_marker_size=3, scatter_size=3, scatter_alpha=0.5,
    axis_off=True, plot_dir='E', use_mm=True,
    plot_scatter_axes=True, x_scale=5,
    scatter_xlim=None, scatter_ylim=None, scatter_int=1,
):
    """3-panel figure: spatial error distribution, forward-vel histogram, angular-vel histogram.

    Requires an 'error_size' column (see ``split_theta_error``).
    """
    if err_palette is None:
        err_palette = {'small': 'r', 'large': 'b'}

    fig = pl.figure(figsize=figsize, dpi=300)

    ax = fig.add_subplot(1, 3, 1)
    if plot_dir == 'E':
        xvar = 'targ_rel_pos_x_mm' if use_mm else 'targ_rel_pos_x'
        yvar = 'targ_rel_pos_y_mm' if use_mm else 'targ_rel_pos_y'
    elif plot_dir == 'N':
        xvar = 'targ_rel_pos_y_mm' if use_mm else 'targ_rel_pos_y'
        yvar = 'targ_rel_pos_x_mm' if use_mm else 'targ_rel_pos_x'
    sns.scatterplot(data=chase_.iloc[0::scatter_int], x=xvar, y=yvar, ax=ax,
                    hue='error_size', palette=err_palette, s=scatter_size,
                    alpha=scatter_alpha)
    if plot_scatter_axes:
        ax.set_xlabel('Relative x (mm)' if use_mm else 'Relative x (px)',
                       labelpad=2)
        ax.set_ylabel('Relative y (mm)' if use_mm else 'Relative y (px)',
                       labelpad=2)
        pl.tick_params(axis='both', which='both', pad=0)
    else:
        ax.set_xticks([0, x_scale])
        ax.set_xticklabels([
            '', '{} mm'.format(x_scale) if use_mm else '{} px'.format(x_scale)
        ])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks([])
        sns.despine(ax=ax, bottom=True, left=True, trim=True)
        ax.spines['left'].set_visible(False)
    if scatter_xlim is not None:
        if plot_dir == 'E':
            ax.set_xlim([0, scatter_xlim])
            ax.set_ylim([-scatter_ylim, scatter_ylim])
        else:
            ax.set_xlim([-scatter_xlim, scatter_xlim])
            ax.set_ylim([0, scatter_ylim])
    ax.set_aspect(1)
    ax.plot(0, 0, marker=fly_marker, color=fly_color,
            markersize=fly_marker_size)
    if axis_off:
        ax.axis('off')
    ax.legend_.remove()

    ax1 = fig.add_subplot(1, 3, 2)
    sns.histplot(data=chase_, x=var1, ax=ax1, bins=50, linewidth=lw,
                 stat='probability', cumulative=False, element='step',
                 fill=False, hue='error_size', palette=err_palette,
                 common_norm=False, legend=0)
    ax1.set_xlabel('Forward vel', labelpad=2)
    pl.tick_params(axis='both', which='both', pad=0)

    ax2 = fig.add_subplot(1, 3, 3, sharey=ax1)
    sns.histplot(data=chase_, x=var2, ax=ax2, color='r', bins=50,
                 stat='probability', cumulative=False, element='step',
                 fill=False, hue='error_size', palette=err_palette,
                 common_norm=False)
    ax2.set_xlabel('Angular vel', labelpad=2)
    pl.tick_params(axis='both', which='both', pad=0)
    sns.move_legend(ax2, bbox_to_anchor=(1, 1), loc='upper left',
                    frameon=False)

    curr_ylim = np.round(ax1.get_ylim()[-1], 2) * 1.15
    for v, ax in zip([var1, var2], [ax1, ax2]):
        med_ = chase_.groupby('error_size')[v].median()
        for mval, cval in err_palette.items():
            ax.plot(med_[mval], curr_ylim, color=cval, marker='v',
                    markersize=median_marker_size)
        ax.set_box_aspect(1)

    return fig
