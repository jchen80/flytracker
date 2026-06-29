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


def plot_action_relative_timing_across_assays(timing_dfs, assay_colors=None,
                                               jitter_seed=42, figsize=None):
    '''
    Scatter plot showing when a target behavior occurs relative to cumulative
    action pseudotime, one row per assay type.

    X-axis: relative timing (0–100% through action pseudotime)
    Y-axis: assay types (one row each, points jittered vertically)
    Each point: one target bout onset, colored by assay type
    Vertical tick: mean relative timing per assay

    Arguments:
        timing_dfs -- dict {assay_type: DataFrame from tutil.get_action_relative_timing}

    Keyword Arguments:
        assay_colors -- dict {assay_type: color} (default: auto)
        jitter_seed  -- random seed for vertical jitter (default: 42)
        figsize      -- figure size tuple (default: auto)

    Returns:
        fig, ax
    '''
    assay_order = sorted(timing_dfs.keys())
    n_assays = len(assay_order)
    if n_assays == 0:
        return None, None

    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in assay_order}

    if figsize is None:
        figsize = (8, max(3, 2 * n_assays))

    rng = np.random.default_rng(jitter_seed)
    jitter_amp = 0.3

    fig, ax = plt.subplots(figsize=figsize)

    yticks, yticklabels = [], []
    for y_idx, assay_type in enumerate(assay_order):
        df = timing_dfs.get(assay_type, pd.DataFrame())
        color = assay_colors.get(assay_type, putil.courtship_color(assay_type))

        n_acqs = df['acquisition'].nunique() if len(df) > 0 else 0
        n_events = len(df)
        yticks.append(y_idx)
        yticklabels.append(f'{assay_type}\n({n_events} events, {n_acqs} acq)')

        if len(df) == 0:
            continue

        x = df['relative_timing'].values * 100
        jitter = rng.uniform(-jitter_amp, jitter_amp, size=len(x))
        ax.scatter(x, y_idx + jitter, color=color, s=35, alpha=0.75,
                   linewidths=0, zorder=3)

        # Mean marker as a vertical tick spanning the jitter band
        mean_x = x.mean()
        ax.vlines(mean_x, y_idx - jitter_amp - 0.05, y_idx + jitter_amp + 0.05,
                  color=color, linewidth=2.5, zorder=5)

    ax.set_xlabel('courtship pseudotime (%)')
    ax.set_xlim(0, 100)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-0.7, n_assays - 0.3)
    ax.axvline(50, color='gray', lw=0.8, ls='--', alpha=0.4)

    plt.tight_layout()
    return fig, ax
