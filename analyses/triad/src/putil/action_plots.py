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
import libs.plotting as putil

from ._helpers import _filter_by_focal_fly


def plot_action_fraction_across_assays(assay_dfs, action_col,
                                        assay_colors=None,
                                        save_dir=None, figsize=(6, 5)):
    '''
    Plot fraction of frames in which action_col occurs, compared across assay types.
    Fraction is computed per acquisition as unique action frames / total frames.
    Each point is one acquisition; bar shows the mean +/- SE.

    Arguments:
        assay_dfs  -- dict mapping assay type to its dataframe
        action_col -- name of action column

    Keyword Arguments:
        assay_colors -- dict mapping assay type to color (default: None)
        save_dir     -- directory to save figure (default: None)
        figsize      -- figure size (default: (6, 5))

    Returns:
        fig, ax
    '''
    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in sorted(assay_dfs.keys())}

    records = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns:
            print(f"Action '{action_col}' not found in assay '{assay_type}', skipping.")
            continue
        for acq_name, acq_df in assay_df.groupby('acquisition'):
            total_frames = acq_df['frame'].nunique()
            if total_frames == 0:
                continue
            action_frames = acq_df[acq_df[action_col] != -1]['frame'].nunique()
            records.append({
                'assay_type': assay_type,
                'acquisition': acq_name,
                'fraction': action_frames / total_frames,
            })

    if len(records) == 0:
        print("No data to plot.")
        return None, None

    plot_data = pd.DataFrame(records)
    assay_order = sorted(assay_dfs.keys())
    palette = [assay_colors.get(a, putil.courtship_color(a)) for a in assay_order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_data, x='assay_type', y='fraction',
                order=assay_order, palette=palette,
                errorbar='se', ax=ax, alpha=0.7)
    sns.stripplot(data=plot_data, x='assay_type', y='fraction',
                  order=assay_order, palette=palette,
                  size=7, jitter=True, ax=ax, alpha=0.9,
                  linewidth=0.5, edgecolor='white')

    n_per_assay = plot_data.groupby('assay_type')['acquisition'].nunique()
    ax.set_xticklabels([f'{a}\n(n={n_per_assay.get(a, 0)})' for a in assay_order])
    ax.set_xlabel('assay type')
    ax.set_ylabel('fraction of frames')
    ax.set_title(f'{action_col}: fraction of frames')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'{action_col}_fraction_across_assays.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, ax


def plot_action_rate_across_assays(assay_dfs, action_col, fps=None,
                                    norm_minutes=30,
                                    assay_colors=None,
                                    save_dir=None, figsize=(6, 5)):
    '''
    Plot number of action bouts compared across assay types.
    If norm_minutes is set, normalizes by recording duration (bouts per norm_minutes).
    If norm_minutes is None, plots raw bout count — useful for one-off events like copulation.
    Each point is one acquisition; bar shows the mean +/- SE.

    Arguments:
        assay_dfs  -- dict mapping assay type to its dataframe
        action_col -- name of action column

    Keyword Arguments:
        fps          -- frames per second; required when norm_minutes is set (default: None)
        norm_minutes -- normalize to this many minutes; if None, plot raw count (default: 30)
        assay_colors -- dict mapping assay type to color (default: None)
        save_dir     -- directory to save figure (default: None)
        figsize      -- figure size (default: (6, 5))

    Returns:
        fig, ax
    '''
    if norm_minutes is not None and fps is None:
        raise ValueError("fps is required when norm_minutes is set.")

    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in sorted(assay_dfs.keys())}

    boutnum_col = f'{action_col}_boutnum'
    records = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns:
            print(f"Action '{action_col}' not found in assay '{assay_type}', skipping.")
            continue
        if boutnum_col not in assay_df.columns:
            print(f"Boutnum column '{boutnum_col}' not found in assay '{assay_type}', skipping.")
            continue
        for acq_name, acq_df in assay_df.groupby('acquisition'):
            total_frames = acq_df['frame'].nunique()
            if total_frames == 0:
                continue
            n_bouts = acq_df[acq_df[action_col] != -1][boutnum_col].nunique()
            if norm_minutes is not None:
                duration_min = (total_frames / fps) / 60
                value = n_bouts / duration_min * norm_minutes
            else:
                value = n_bouts
            records.append({
                'assay_type': assay_type,
                'acquisition': acq_name,
                'value': value,
                'n_bouts': n_bouts,
            })

    if len(records) == 0:
        print("No data to plot.")
        return None, None

    plot_data = pd.DataFrame(records)
    assay_order = sorted(assay_dfs.keys())
    palette = [assay_colors.get(a, putil.courtship_color(a)) for a in assay_order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_data, x='assay_type', y='value',
                order=assay_order, palette=palette,
                errorbar='se', ax=ax, alpha=0.7)
    sns.stripplot(data=plot_data, x='assay_type', y='value',
                  order=assay_order, palette=palette,
                  size=7, jitter=True, ax=ax, alpha=0.9,
                  linewidth=0.5, edgecolor='white')

    n_per_assay = plot_data.groupby('assay_type')['acquisition'].nunique()
    ax.set_xticklabels([f'{a}\n(n={n_per_assay.get(a, 0)})' for a in assay_order])
    ax.set_xlabel('assay type')
    ylabel = f'bouts per {norm_minutes} min' if norm_minutes is not None else 'bout count'
    ax.set_ylabel(ylabel)
    ax.set_title(f'{action_col}: {ylabel}')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'{action_col}_rate_across_assays.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, ax


def plot_target_sex_fraction_across_assays(assay_dfs, action_col, sex_map,
                                            focal_flies_map=None,
                                            assay_colors=None,
                                            save_dir=None, figsize=(8, 5)):
    '''
    Plot fraction of action frames in which the focal fly targeted a male vs female,
    compared across assay types.

    Fraction is computed per acquisition as:
        frames targeting sex X / total frames with an assigned target

    Each point is one acquisition; bar shows mean +/- SE.
    Male and female fractions are shown side by side as a grouped bar.

    Arguments:
        assay_dfs  -- dict mapping assay type to its dataframe
        action_col -- name of action column (e.g. 'courtship')
        sex_map    -- dict mapping triad_type to {fly_id: 'M' or 'F'},
                      e.g. {'MMF': {0: 'M', 1: 'M', 2: 'F'},
                             'MFF': {0: 'M', 1: 'F', 2: 'F'}}

    Keyword Arguments:
        focal_flies_map -- dict mapping triad_type to list of focal fly ids (default: None)
        assay_colors    -- dict mapping assay type to color (default: None)
        save_dir        -- directory to save figure (default: None)
        figsize         -- figure size (default: (8, 5))

    Returns:
        fig, ax
    '''
    target_col = f'{action_col}_target'

    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in sorted(assay_dfs.keys())}

    records = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns or target_col not in assay_df.columns:
            print(f"Missing '{action_col}' or '{target_col}' in assay '{assay_type}', skipping.")
            continue

        triad_type = assay_df['triad_type'].iloc[0]
        fly_sex = sex_map.get(triad_type, {})
        if not fly_sex:
            print(f"No sex_map entry for triad_type '{triad_type}', skipping assay '{assay_type}'.")
            continue

        focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
        plot_df = _filter_by_focal_fly(assay_df, focal_flies)

        # rows where focal fly is acting and a target is assigned
        acting_df = plot_df[
            (plot_df[action_col] == plot_df['id']) &
            (pd.to_numeric(plot_df[target_col], errors='coerce').fillna(-1).astype(int) != -1)
        ].drop_duplicates(['acquisition', 'frame', 'id'])

        for acq_name, acq_df in acting_df.groupby('acquisition'):
            total = len(acq_df)
            if total == 0:
                continue

            target_sex = acq_df[target_col].astype(int).map(fly_sex)
            for sex in ['M', 'F']:
                count = (target_sex == sex).sum()
                records.append({
                    'assay_type': assay_type,
                    'acquisition': acq_name,
                    'sex': sex,
                    'fraction': count / total,
                })

    if len(records) == 0:
        print("No data to plot.")
        return None, None

    plot_data = pd.DataFrame(records)
    assay_order = sorted(assay_dfs.keys())

    sex_palette = {'M': '#5599ff', 'F': '#ff6666'}

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_data, x='assay_type', y='fraction', hue='sex',
                order=assay_order, hue_order=['M', 'F'],
                palette=sex_palette, errorbar='se', ax=ax, alpha=0.7)
    sns.stripplot(data=plot_data, x='assay_type', y='fraction', hue='sex',
                  order=assay_order, hue_order=['M', 'F'],
                  palette=sex_palette, size=6, jitter=True, ax=ax, alpha=0.85,
                  linewidth=0.5, edgecolor='white', dodge=True,
                  legend=False)

    n_per_assay = plot_data.groupby('assay_type')['acquisition'].nunique()
    ax.set_xticklabels([f'{a}\n(n={n_per_assay.get(a, 0)})' for a in assay_order])
    ax.set_xlabel('assay type')
    ax.set_ylabel('fraction of action frames')
    ax.set_title(f'{action_col}: fraction of frames targeting male vs female')
    ax.legend(title='target sex')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'{action_col}_target_sex_fraction_across_assays.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, ax
