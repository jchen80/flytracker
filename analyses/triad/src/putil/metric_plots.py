import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Patch
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

from ._helpers import _build_condition_panels, _compute_metric_histogram, _exclude_action_frames, _filter_by_focal_fly, _filter_to_target_pairs, _select_action_frames


# Units for metric axis labels. dist_to_other / dist_to_other_body_adj are in mm
# (compute_dist_to_other and compute_dist_body_adj both divide by pix_per_mm).
_METRIC_UNITS = {
    'dist_to_other': 'mm',
    'dist_to_other_body_adj': 'mm',
    'abs_theta_error_deg': 'deg',
    'theta_error_deg': 'deg',
}


def _metric_label(metric):
    """Axis label for a metric, with units appended when known."""
    unit = _METRIC_UNITS.get(metric)
    return f'{metric} ({unit})' if unit else metric


def plot_metric_distribution(df, metric='dist_to_other',
                              action_cols=None, focal_fly_ids=None,
                              save_dir=None, acq=None,
                              figsize=(10, 5), bins=50,
                              xlim_percentile=99):
    plot_df = _filter_by_focal_fly(df, focal_fly_ids)
    if len(plot_df) == 0:
        print("No rows found after focal fly filtering.")
        return None, None
    if metric not in plot_df.columns:
        print(f"Metric '{metric}' not found.")
        return None, None

    panels = _build_condition_panels(plot_df, action_cols)
    palette = sns.color_palette('husl', len(panels))

    fig, ax = plt.subplots(figsize=figsize)
    for (label, panel_df), color in zip(panels, palette):
        # per-condition bin edges, clipped at xlim_percentile to suppress long tails
        panel_vals = panel_df.drop_duplicates(['frame', 'pair'])[metric].dropna()
        x_max = np.percentile(panel_vals, xlim_percentile) if xlim_percentile is not None else panel_vals.max()
        bin_edges = np.linspace(panel_vals.min(), x_max, bins + 1)
        counts, n = _compute_metric_histogram(panel_df, metric, bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        ax.bar(bin_centers, counts, width=bin_width, color=color, alpha=0.5,
               edgecolor=color, linewidth=0.5, label=f'{label} (n={n})')

    ax.set_xlabel(_metric_label(metric))
    ax.set_ylabel('density')
    ax.legend(fontsize=9)
    title = f'{acq} — ' if acq else ''
    focal_str = f'focal fly: {focal_fly_ids}' if focal_fly_ids is not None else 'all pairs'
    ax.set_title(f'{title}{metric} distribution\n{focal_str}')
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        savepath = os.path.join(save_dir, f'{metric}_distribution_{action_str}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, ax


def plot_metric_distribution_across_assays(assay_dfs, metric='dist_to_other',
                                            action_cols=None, focal_flies_map=None,
                                            assay_colors=None, bins=50,
                                            include_non_action=False,
                                            target_action_col=None,
                                            xlim_percentile=99, kde=False,
                                            save_dir=None, figsize=None, ax=None):
    """
    Plot distribution of a metric across multiple assay types, optionally split by action conditions.
    Arguments:
    - assay_dfs: dict mapping assay type to its dataframe which is processed to contain per-frame, per-pair of fly metrics such as dist_to_other, theta_error, etc.
    - metric: name of the metric column to plot (e.g. 'dist_to_other)
    - action_cols: list of action column names to split by (e.g. ['courtship', 'aggression']). If None, only 'all frames' panel is plotted.
    - focal_flies_map: dict mapping triad_type to list of focal fly ids to include in the plot when considering pairs. If None, all flies are included.
    - assay_colors: dict mapping assay type to color for plotting. If None, a default color palette is used.
    - bins: number of bins for the histogram
    - include_non_action: if True, add a non-action panel for each action, labeled
                          'non-<action> (all pairs)' to indicate it covers all fly pairs (default: False)
    - target_action_col: if set, the regular action panel is labeled '<action> (all pairs)' and
                         an additional '<action> (target only)' panel is added showing only the
                         focal fly → assigned target pair. When not set, the action panel keeps
                         its plain label. (e.g. 'courtship'). (default: None)
    - xlim_percentile: clip bin range to this percentile of the data to suppress long tails;
                       set to None to use the full data range (default: 99)
    - save_dir: directory to save the figure. If None, figure is not saved.
    - figsize: tuple specifying figure size. If None, size is determined by number of panels.
    - ax: optional existing Axes to draw a single panel onto (requires exactly one
          condition, e.g. action_cols=None). When given, no new figure is created and
          the suptitle/tight_layout/save steps are skipped so the caller can compose
          several calls into one combined figure. (default: None)
    """
    # Build condition labels
    condition_labels = ['all frames']
    if action_cols is not None:
        for a in action_cols:
            if any(a in df.columns for df in assay_dfs.values()):
                condition_labels.append(f'{a} (all pairs)' if target_action_col == a else a)
                if target_action_col == a:
                    condition_labels.append(f'{a} (target only)')
                if include_non_action:
                    condition_labels.append(f'non-{a} (all pairs)')

    n_panels = len(condition_labels)
    if figsize is None:
        figsize = (6 * n_panels, 5)

    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in sorted(assay_dfs.keys())}

    def _apply_condition_filter(plot_df, cond_label):
        '''Filter plot_df to the rows for a given condition label. Returns None to skip.'''
        if cond_label == 'all frames':
            return plot_df
        elif cond_label.startswith('non-'):
            action = cond_label[4:].replace(' (all pairs)', '')
            if action not in plot_df.columns:
                return None
            return _exclude_action_frames(plot_df, action)
        elif cond_label.endswith(' (target only)'):
            action = cond_label[:-len(' (target only)')]
            if action not in plot_df.columns:
                return None
            return _filter_to_target_pairs(_select_action_frames(plot_df, action), action)
        else:
            action = cond_label.replace(' (all pairs)', '')
            if action not in plot_df.columns:
                return None
            return _select_action_frames(plot_df, action)

    # Pass 1: filter data per (cond_label, assay_type)
    panel_dfs = {}
    for cond_label in condition_labels:
        for assay_type in sorted(assay_dfs.keys()):
            assay_df = assay_dfs[assay_type]
            triad_type = assay_df['triad_type'].iloc[0]
            focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
            filtered = _apply_condition_filter(
                _filter_by_focal_fly(assay_df, focal_flies), cond_label)
            panel_dfs[(cond_label, assay_type)] = filtered

    # Pass 2: compute per-condition bin edges from data pooled across assays
    condition_bin_edges = {}
    for cond_label in condition_labels:
        cond_vals = [
            panel_dfs[(cond_label, at)].drop_duplicates(['frame', 'pair', 'acquisition'])[metric].dropna()
            for at in sorted(assay_dfs.keys())
            if panel_dfs[(cond_label, at)] is not None and len(panel_dfs[(cond_label, at)]) > 0
        ]
        if not cond_vals:
            continue
        combined = pd.concat(cond_vals)
        x_max = np.percentile(combined, xlim_percentile) if xlim_percentile is not None else combined.max()
        condition_bin_edges[cond_label] = np.linspace(combined.min(), x_max, bins + 1)

    # Pass 3: plot
    external_ax = ax is not None
    if external_ax:
        if n_panels != 1:
            raise ValueError("ax= requires a single condition (e.g. action_cols=None)")
        fig, axes = ax.figure, [ax]
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=False)
        if n_panels == 1:
            axes = [axes]

    for panel_ax, cond_label in zip(axes, condition_labels):
        bin_edges = condition_bin_edges.get(cond_label)
        if bin_edges is None:
            continue
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        panel_handles = []
        for assay_type in sorted(assay_dfs.keys()):
            plot_df = panel_dfs.get((cond_label, assay_type))
            if plot_df is None or len(plot_df) == 0:
                continue
            counts, n = _compute_metric_histogram(
                plot_df, metric, bin_edges,
                dedup_cols=['frame', 'pair', 'acquisition'])
            num_acq = plot_df['acquisition'].nunique()
            bin_width = bin_edges[1] - bin_edges[0]
            color = assay_colors.get(assay_type, putil.courtship_color(assay_type))
            if kde:
                # smooth KDE curve over the same x-range (filled + outlined)
                vals = plot_df.drop_duplicates(['frame', 'pair', 'acquisition'])[metric].dropna().to_numpy()
                if len(vals) > 50000:             # subsample so gaussian_kde stays fast
                    vals = np.random.default_rng(0).choice(vals, 50000, replace=False)
                if len(vals) > 1 and np.ptp(vals) > 0:
                    grid = np.linspace(bin_edges[0], bin_edges[-1], 200)
                    try:
                        dens = gaussian_kde(vals)(grid)
                        panel_ax.fill_between(grid, dens, color=color, alpha=0.15, lw=0)
                        panel_ax.plot(grid, dens, color=color, linewidth=1.8, alpha=0.9)
                    except np.linalg.LinAlgError:
                        pass
            else:
                # low-opacity fill with a step outline across the top
                panel_ax.bar(bin_centers, counts, width=bin_width, color=color, alpha=0.2,
                             edgecolor='none', linewidth=0)
                panel_ax.step(np.append(bin_edges[:-1], bin_edges[-1]),
                              np.append(counts, counts[-1]),
                              where='post', color=color, linewidth=1.5, alpha=0.9)
            panel_handles.append(Patch(facecolor=color, edgecolor=color,
                                       label=f'{assay_type} (n={num_acq})'))

        panel_ax.set_xlabel(_metric_label(metric))
        panel_ax.set_title(cond_label)
        panel_ax.legend(handles=panel_handles, fontsize=8)

    axes[0].set_ylabel('density')
    if not external_ax:
        mtag = ' (KDE)' if kde else ''
        fig.suptitle(f'{metric} distribution across assay types{mtag}', fontsize=12)
        plt.tight_layout()

        if save_dir is not None:
            action_str = '_'.join(action_cols) if action_cols else 'all'
            msuf = '_kde' if kde else ''
            savepath = os.path.join(save_dir,
                                    f'{metric}_distribution_across_assays_{action_str}{msuf}.png')
            fig.savefig(savepath, dpi=150, bbox_inches='tight')
            print(f"Saved to {savepath}")
    return fig, axes


def _condition_labels(assay_dfs, action_cols, target_action_col, include_non_action):
    """Condition-panel labels shared by the across-assay metric plots."""
    labels = ['all frames']
    if action_cols is not None:
        for a in action_cols:
            if any(a in df.columns for df in assay_dfs.values()):
                labels.append(f'{a} (all pairs)' if target_action_col == a else a)
                if target_action_col == a:
                    labels.append(f'{a} (target only)')
                if include_non_action:
                    labels.append(f'non-{a} (all pairs)')
    return labels


def _apply_condition_filter(plot_df, cond_label):
    """Filter a (focal-filtered) df to the rows for one condition label, or None."""
    if cond_label == 'all frames':
        return plot_df
    if cond_label.startswith('non-'):
        action = cond_label[4:].replace(' (all pairs)', '')
        return _exclude_action_frames(plot_df, action) if action in plot_df.columns else None
    if cond_label.endswith(' (target only)'):
        action = cond_label[:-len(' (target only)')]
        return (_filter_to_target_pairs(_select_action_frames(plot_df, action), action)
                if action in plot_df.columns else None)
    action = cond_label.replace(' (all pairs)', '')
    return _select_action_frames(plot_df, action) if action in plot_df.columns else None


def plot_metric_per_acquisition_across_assays(assay_dfs, metric='dist_to_other',
                                              action_cols=None, focal_flies_map=None,
                                              assay_colors=None, agg='mean',
                                              include_non_action=False,
                                              target_action_col=None,
                                              save_dir=None, figsize=None):
    """Per-acquisition summary of a metric across assay types: one point per
    acquisition (its `agg` over the per-frame, per-pair values), grouped by assay
    type, with the across-acquisition mean ± SEM overlaid. One panel per action
    condition (same conditions as plot_metric_distribution_across_assays).

    Complements the per-frame distribution/violin views by making the acquisition
    the unit of replication, so assays can be compared as points. `agg` is 'mean'
    (default) or 'median'.
    """
    condition_labels = _condition_labels(assay_dfs, action_cols, target_action_col,
                                          include_non_action)
    assay_types = sorted(assay_dfs.keys())
    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in assay_types}
    _agg = np.nanmedian if agg == 'median' else np.nanmean

    # per (condition, assay): Series of one aggregated value per acquisition
    per_acq = {}
    for cond_label in condition_labels:
        for assay_type in assay_types:
            adf = assay_dfs[assay_type]
            focal = focal_flies_map.get(adf['triad_type'].iloc[0]) if focal_flies_map else None
            filt = _apply_condition_filter(_filter_by_focal_fly(adf, focal), cond_label)
            if filt is None or len(filt) == 0 or metric not in filt.columns:
                per_acq[(cond_label, assay_type)] = pd.Series(dtype=float)
                continue
            dedup = filt.drop_duplicates(['frame', 'pair', 'acquisition'])
            per_acq[(cond_label, assay_type)] = (
                dedup.groupby('acquisition')[metric].agg(_agg).dropna())

    n_panels = len(condition_labels)
    if figsize is None:
        figsize = (max(3.0, 1.0 * len(assay_types)) * n_panels, 5)
    # independent y per panel: each condition (all / all-pairs / target-only) auto-
    # scales to its own data range rather than sharing one axis
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False, sharey=False)
    axes = axes[0]
    rng = np.random.default_rng(0)

    for panel_ax, cond_label in zip(axes, condition_labels):
        for i, assay_type in enumerate(assay_types):
            vals = per_acq[(cond_label, assay_type)].to_numpy()
            color = assay_colors.get(assay_type, 'gray')
            if not len(vals):
                continue
            jit = (rng.random(len(vals)) - 0.5) * 0.25
            panel_ax.scatter(i + jit, vals, s=42, color=color, alpha=0.85,
                             edgecolor='black', linewidths=0.4, zorder=3)
            m = float(np.nanmean(vals))
            se = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            panel_ax.plot([i - 0.28, i + 0.28], [m, m], color=color, lw=3, zorder=4)
            panel_ax.errorbar(i, m, yerr=se, color=color, capsize=4, lw=1.5, zorder=4)
        panel_ax.set_xticks(range(len(assay_types)))
        panel_ax.set_xticklabels(
            [f'{at}\n(n={len(per_acq[(cond_label, at)])})' for at in assay_types],
            rotation=30, ha='right', fontsize=8)
        panel_ax.set_title(cond_label)
        panel_ax.margins(x=0.15)
    axes[0].set_ylabel(f'{agg} {_metric_label(metric)} / acquisition')
    fig.suptitle(f'Per-acquisition {agg} {metric} across assay types', fontsize=13)
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        savepath = os.path.join(
            save_dir, f'{metric}_per_acquisition_across_assays_{action_str}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, axes


def plot_metric_violin_across_assays(assay_dfs, metric='dist_to_other',
                                      action_cols=None, focal_flies_map=None,
                                      assay_colors=None,
                                      include_non_action=False,
                                      target_action_col=None,
                                      ylim_percentile=99,
                                      save_dir=None, figsize=None):
    """
    Violin plot of a metric across assay types, one panel per condition.
    Assay types on the x-axis, metric value on the y-axis.

    Arguments mirror plot_metric_distribution_across_assays; histgram-specific
    params (bins, xlim_percentile) are replaced by ylim_percentile.

    Arguments:
        assay_dfs      -- dict mapping assay_type to processed DataFrame
        metric         -- column name to plot (e.g. 'dist_to_other')

    Keyword Arguments:
        action_cols       -- list of action column names to split by (default: None)
        focal_flies_map   -- dict mapping triad_type to focal fly id list (default: None)
        assay_colors      -- dict mapping assay_type to color (default: None)
        include_non_action-- add non-action panels (default: False)
        target_action_col -- show target-only panel for this action (default: None)
        ylim_percentile   -- clip y-axis to this percentile to suppress tails (default: 99)
        save_dir          -- directory to save figure (default: None)
        figsize           -- figure size tuple (default: auto)

    Returns:
        fig, axes
    """
    condition_labels = ['all frames']
    if action_cols is not None:
        for a in action_cols:
            if any(a in df.columns for df in assay_dfs.values()):
                condition_labels.append(f'{a} (all pairs)' if target_action_col == a else a)
                if target_action_col == a:
                    condition_labels.append(f'{a} (target only)')
                if include_non_action:
                    condition_labels.append(f'non-{a} (all pairs)')

    n_panels = len(condition_labels)
    if figsize is None:
        figsize = (max(4, 2 * len(assay_dfs)) * n_panels, 5)

    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in sorted(assay_dfs.keys())}

    def _apply_condition_filter(plot_df, cond_label):
        if cond_label == 'all frames':
            return plot_df
        elif cond_label.startswith('non-'):
            action = cond_label[4:].replace(' (all pairs)', '')
            if action not in plot_df.columns:
                return None
            return _exclude_action_frames(plot_df, action)
        elif cond_label.endswith(' (target only)'):
            action = cond_label[:-len(' (target only)')]
            if action not in plot_df.columns:
                return None
            return _filter_to_target_pairs(_select_action_frames(plot_df, action), action)
        else:
            action = cond_label.replace(' (all pairs)', '')
            if action not in plot_df.columns:
                return None
            return _select_action_frames(plot_df, action)

    assay_order = sorted(assay_dfs.keys())

    # Collect filtered + deduped values per (condition, assay_type)
    panel_data = {}
    for cond_label in condition_labels:
        rows = []
        for assay_type in assay_order:
            assay_df = assay_dfs[assay_type]
            triad_type = assay_df['triad_type'].iloc[0]
            focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
            filtered = _apply_condition_filter(
                _filter_by_focal_fly(assay_df, focal_flies), cond_label)
            if filtered is None or len(filtered) == 0:
                continue
            vals = (filtered
                    .drop_duplicates(['frame', 'pair', 'acquisition'])[metric]
                    .dropna())
            tmp = pd.DataFrame({'value': vals, 'assay_type': assay_type})
            rows.append(tmp)
        panel_data[cond_label] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Global y-limit from pooled data
    all_vals = pd.concat([d['value'] for d in panel_data.values() if len(d) > 0],
                         ignore_index=True)
    y_max = np.percentile(all_vals, ylim_percentile) if ylim_percentile is not None and len(all_vals) > 0 else None

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, cond_label in zip(axes, condition_labels):
        data = panel_data.get(cond_label, pd.DataFrame())
        if data.empty:
            ax.set_visible(False)
            continue

        palette = [assay_colors.get(at, putil.courtship_color(at)) for at in assay_order
                   if at in data['assay_type'].values]
        present_order = [at for at in assay_order if at in data['assay_type'].values]

        sns.violinplot(data=data, x='assay_type', y='value', order=present_order,
                       palette=palette, ax=ax, inner='box', cut=0, linewidth=0.8)

        ax.set_xlabel('')
        ax.set_ylabel(_metric_label(metric) if cond_label == condition_labels[0] else '')
        ax.set_title(cond_label)
        counts = data.groupby('assay_type').size()
        tick_labels = [f'{at}\n(n={counts.get(at, 0)})' for at in present_order]
        ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
        if y_max is not None:
            ax.set_ylim(bottom=0, top=y_max)

    fig.suptitle(f'{metric} across assay types', fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        savepath = os.path.join(save_dir,
                                f'{metric}_violin_across_assays_{action_str}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, axes


def plot_metric_by_velocity_bin_across_assays(
        binned_assay_dfs, metric,
        focal_flies_map=None,
        assay_colors=None,
        ylim_percentile=95,
        figsize=None):
    """
    Grouped violin plot: velocity bins on x-axis, metric on y-axis,
    colored/grouped by assay type.

    Arguments:
        binned_assay_dfs -- ordered list of (bin_label, {assay_type: df})
                            where each df is already filtered to that velocity
                            bin and to action/target frames
        metric           -- column name to plot on y-axis

    Keyword Arguments:
        focal_flies_map  -- dict mapping triad_type to focal fly id list
        assay_colors     -- dict mapping assay_type to color
        ylim_percentile  -- clip y-axis to this percentile (default: 95)
        figsize          -- (w, h) override (default: auto)

    Returns:
        fig, ax
    """
    # Build tidy DataFrame
    rows = []
    bin_order = []
    for bin_label, assay_dfs in binned_assay_dfs:
        if bin_label not in bin_order:
            bin_order.append(bin_label)
        for assay_type, df in assay_dfs.items():
            if metric not in df.columns:
                continue
            triad_type = df['triad_type'].iloc[0] if 'triad_type' in df.columns else None
            focal_flies = (focal_flies_map.get(triad_type)
                           if focal_flies_map and triad_type else None)
            plot_df = _filter_by_focal_fly(df, focal_flies)
            vals = (plot_df
                    .drop_duplicates(['frame', 'pair', 'acquisition'])[metric]
                    .dropna())
            if len(vals) == 0:
                continue
            rows.append(pd.DataFrame({
                'bin':        bin_label,
                'assay_type': assay_type,
                'value':      vals.values,
            }))

    if not rows:
        fig, ax = plt.subplots()
        ax.set_visible(False)
        return fig, ax

    data = pd.concat(rows, ignore_index=True)

    assay_order = sorted(data['assay_type'].unique())
    if assay_colors is None:
        assay_colors = {at: putil.courtship_color(at) for at in assay_order}

    y_max = (np.percentile(data['value'].dropna(), ylim_percentile)
             if ylim_percentile is not None else None)

    n_bins   = len(bin_order)
    n_assays = len(assay_order)
    if figsize is None:
        figsize = (max(6, n_bins * n_assays * 0.9), 5)

    fig, ax = plt.subplots(figsize=figsize)

    palette = [assay_colors.get(at, putil.courtship_color(at)) for at in assay_order]
    sns.violinplot(
        data=data, x='bin', y='value',
        hue='assay_type',
        order=bin_order, hue_order=assay_order,
        palette=palette,
        ax=ax, inner='box', cut=0, linewidth=0.8,
        density_norm='width',
    )

    ax.set_xlabel('velocity bin (mm/s)')
    ax.set_ylabel(_metric_label(metric))
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max)

    # Annotate n per (bin, assay)
    counts = data.groupby(['bin', 'assay_type']).size()
    annot_lines = [
        f"{at}: " + ", ".join(
            str(counts.get((bl, at), 0)) for bl in bin_order)
        for at in assay_order
    ]
    ax.annotate(
        "n per bin — " + " | ".join(annot_lines),
        xy=(0.01, 0.99), xycoords='axes fraction',
        va='top', ha='left', fontsize=7,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=8, framealpha=0.4)

    plt.tight_layout()
    return fig, ax


def plot_metric_by_assay_then_velocity_bin(
        binned_assay_dfs, metric,
        focal_flies_map=None,
        bin_colors=None,
        ylim_percentile=95,
        figsize=None):
    """
    Grouped violin plot: assay type on x-axis, metric on y-axis,
    colored/grouped by velocity bin within each assay.

    Same inputs as plot_metric_by_velocity_bin_across_assays but with
    x and hue roles swapped.

    Arguments:
        binned_assay_dfs -- ordered list of (bin_label, {assay_type: df})
        metric           -- column name to plot on y-axis

    Keyword Arguments:
        focal_flies_map  -- dict mapping triad_type to focal fly id list
        bin_colors       -- dict mapping bin_label to color
        ylim_percentile  -- clip y-axis to this percentile (default: 95)
        figsize          -- (w, h) override (default: auto)

    Returns:
        fig, ax
    """
    rows = []
    bin_order = []
    for bin_label, assay_dfs in binned_assay_dfs:
        if bin_label not in bin_order:
            bin_order.append(bin_label)
        for assay_type, df in assay_dfs.items():
            if metric not in df.columns:
                continue
            triad_type = df['triad_type'].iloc[0] if 'triad_type' in df.columns else None
            focal_flies = (focal_flies_map.get(triad_type)
                           if focal_flies_map and triad_type else None)
            plot_df = _filter_by_focal_fly(df, focal_flies)
            vals = (plot_df
                    .drop_duplicates(['frame', 'pair', 'acquisition'])[metric]
                    .dropna())
            if len(vals) == 0:
                continue
            rows.append(pd.DataFrame({
                'bin':        bin_label,
                'assay_type': assay_type,
                'value':      vals.values,
            }))

    if not rows:
        fig, ax = plt.subplots()
        ax.set_visible(False)
        return fig, ax

    data = pd.concat(rows, ignore_index=True)

    assay_order = sorted(data['assay_type'].unique())
    if bin_colors is None:
        palette = sns.color_palette('husl', len(bin_order))
        bin_colors = dict(zip(bin_order, palette))

    y_max = (np.percentile(data['value'].dropna(), ylim_percentile)
             if ylim_percentile is not None else None)

    n_assays = len(assay_order)
    n_bins   = len(bin_order)
    if figsize is None:
        figsize = (max(6, n_assays * n_bins * 0.9), 5)

    fig, ax = plt.subplots(figsize=figsize)

    palette = [bin_colors.get(bl, 'white') for bl in bin_order]
    sns.violinplot(
        data=data, x='assay_type', y='value',
        hue='bin',
        order=assay_order, hue_order=bin_order,
        palette=palette,
        ax=ax, inner='box', cut=0, linewidth=0.8,
        density_norm='width',
    )

    ax.set_xlabel('assay type')
    ax.set_ylabel(_metric_label(metric))
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max)

    counts = data.groupby(['assay_type', 'bin']).size()
    annot_lines = [
        f"{at}: " + ", ".join(
            str(counts.get((at, bl), 0)) for bl in bin_order)
        for at in assay_order
    ]
    ax.annotate(
        "n per bin — " + " | ".join(annot_lines),
        xy=(0.01, 0.99), xycoords='axes fraction',
        va='top', ha='left', fontsize=7,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='velocity bin (mm/s)',
              loc='upper right', fontsize=8, framealpha=0.4)

    plt.tight_layout()
    return fig, ax


def plot_metric_per_acquisition_by_velocity_bin_across_assays(
        binned_assay_dfs, metric,
        focal_flies_map=None,
        assay_colors=None,
        agg='mean',
        ylim_percentile=None,
        figsize=None):
    """Per-acquisition summary of a metric, one panel per velocity bin.

    The per-acquisition analog of plot_metric_by_velocity_bin_across_assays:
    within each velocity-bin panel, plot one point per acquisition (its `agg`
    over the per-frame, per-pair values) grouped by assay type, with the
    across-acquisition mean ± SEM overlaid. Makes the acquisition the unit of
    replication so assays can be compared as points within each speed bin.

    Arguments:
        binned_assay_dfs -- ordered list of (bin_label, {assay_type: df}) where
                            each df is already filtered to that velocity bin and
                            to action/target frames
        metric           -- column name to aggregate per acquisition

    Keyword Arguments:
        focal_flies_map -- dict mapping triad_type to focal fly id list
        assay_colors    -- dict mapping assay_type to color
        agg             -- per-acquisition aggregator: 'mean' or 'median'
        ylim_percentile -- clip y-axis to this percentile across all points;
                           None uses the full range (default: None)
        figsize         -- (w, h) override (default: auto)

    Returns:
        fig, axes
    """
    bin_order = [bl for bl, _ in binned_assay_dfs]
    assay_types = sorted({at for _, ad in binned_assay_dfs for at in ad})
    if assay_colors is None:
        assay_colors = {at: putil.courtship_color(at) for at in assay_types}
    _agg = np.nanmedian if agg == 'median' else np.nanmean

    # per (bin_label, assay_type): Series of one aggregated value per acquisition
    per_acq = {}
    for bin_label, assay_dfs in binned_assay_dfs:
        for assay_type in assay_types:
            df = assay_dfs.get(assay_type)
            if df is None or len(df) == 0 or metric not in df.columns:
                per_acq[(bin_label, assay_type)] = pd.Series(dtype=float)
                continue
            triad_type = df['triad_type'].iloc[0] if 'triad_type' in df.columns else None
            focal = (focal_flies_map.get(triad_type)
                     if focal_flies_map and triad_type else None)
            dedup = _filter_by_focal_fly(df, focal).drop_duplicates(
                ['frame', 'pair', 'acquisition'])
            per_acq[(bin_label, assay_type)] = (
                dedup.groupby('acquisition')[metric].agg(_agg).dropna())

    n_panels = len(bin_order)
    if figsize is None:
        figsize = (max(3.0, 1.0 * len(assay_types)) * n_panels, 5)
    # independent y per panel: each velocity bin auto-scales to its own data range
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False, sharey=False)
    axes = axes[0]
    rng = np.random.default_rng(0)

    all_vals = []
    for panel_ax, bin_label in zip(axes, bin_order):
        for i, assay_type in enumerate(assay_types):
            vals = per_acq[(bin_label, assay_type)].to_numpy()
            color = assay_colors.get(assay_type, 'gray')
            if not len(vals):
                continue
            all_vals.append(vals)
            jit = (rng.random(len(vals)) - 0.5) * 0.25
            panel_ax.scatter(i + jit, vals, s=42, color=color, alpha=0.85,
                             edgecolor='black', linewidths=0.4, zorder=3)
            m = float(np.nanmean(vals))
            se = (float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
                  if len(vals) > 1 else 0.0)
            panel_ax.plot([i - 0.28, i + 0.28], [m, m], color=color, lw=3, zorder=4)
            panel_ax.errorbar(i, m, yerr=se, color=color, capsize=4, lw=1.5, zorder=4)
        panel_ax.set_xticks(range(len(assay_types)))
        panel_ax.set_xticklabels(
            [f'{at}\n(n={len(per_acq[(bin_label, at)])})' for at in assay_types],
            rotation=30, ha='right', fontsize=8)
        panel_ax.set_title(bin_label)
        panel_ax.margins(x=0.15)

    if ylim_percentile is not None and all_vals:
        axes[0].set_ylim(top=np.percentile(np.concatenate(all_vals), ylim_percentile))
    axes[0].set_ylabel(f'{agg} {_metric_label(metric)} / acquisition')
    plt.tight_layout()
    return fig, axes


def plot_metric_condition_distribution_across_assays(
        condition_assay_dfs, metric,
        focal_flies_map=None,
        assay_colors=None,
        bins=50, xlim_percentile=95,
        figsize=None):
    '''
    Per-assay overlay histograms comparing conditions.
    One subplot per assay type; conditions overlaid within each subplot.

    Arguments:
        condition_assay_dfs -- dict {condition_label: {assay_type: df}}
        metric              -- column name to plot

    Keyword Arguments:
        focal_flies_map  -- dict mapping triad_type to focal fly id list (default: None)
        assay_colors     -- dict mapping assay_type to color (default: auto)
        bins             -- histogram bin count (default: 50)
        xlim_percentile  -- clip x-axis to this percentile (default: 95)
        figsize          -- figure size (default: auto)

    Returns:
        fig, axes
    '''
    condition_labels = list(condition_assay_dfs.keys())
    all_assay_types = sorted({at for cdf in condition_assay_dfs.values() for at in cdf.keys()})
    n_assays = len(all_assay_types)
    if n_assays == 0:
        return None, None

    if assay_colors is None:
        assay_colors = {at: putil.courtship_color(at) for at in all_assay_types}

    if figsize is None:
        figsize = (5 * n_assays, 4)

    # Shared bin edges from all data pooled
    all_vals = []
    for cond_label, cdf in condition_assay_dfs.items():
        for assay_type, df in cdf.items():
            if metric not in df.columns or len(df) == 0:
                continue
            triad_type = df['triad_type'].iloc[0] if 'triad_type' in df.columns else None
            focal_flies = focal_flies_map.get(triad_type) if focal_flies_map and triad_type else None
            vals = _filter_by_focal_fly(df, focal_flies).drop_duplicates(
                ['frame', 'pair', 'acquisition'])[metric].dropna()
            all_vals.append(vals)
    if not all_vals:
        return None, None
    pooled = pd.concat(all_vals)
    x_max = np.percentile(pooled, xlim_percentile) if xlim_percentile is not None else pooled.max()
    bin_edges = np.linspace(pooled.min(), x_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, axes = plt.subplots(1, n_assays, figsize=figsize, sharey=False)
    if n_assays == 1:
        axes = [axes]

    for ax, assay_type in zip(axes, all_assay_types):
        color = assay_colors.get(assay_type, putil.courtship_color(assay_type))
        for i, cond_label in enumerate(condition_labels):
            df = condition_assay_dfs[cond_label].get(assay_type)
            if df is None or len(df) == 0:
                continue
            triad_type = df['triad_type'].iloc[0] if 'triad_type' in df.columns else None
            focal_flies = focal_flies_map.get(triad_type) if focal_flies_map and triad_type else None
            vals = _filter_by_focal_fly(df, focal_flies).drop_duplicates(
                ['frame', 'pair', 'acquisition'])[metric].dropna()
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals, bins=bin_edges, density=True)
            if i == 0:
                ax.step(np.append(bin_edges[:-1], bin_edges[-1]),
                        np.append(counts, counts[-1]),
                        where='post', color=color, lw=1.8, alpha=0.9,
                        label=f'{cond_label} (n={len(vals)})')
            else:
                ax.bar(bin_centers, counts, width=bin_width,
                       color=color, alpha=0.45, edgecolor='none',
                       label=f'{cond_label} (n={len(vals)})')
        ax.set_xlabel(_metric_label(metric))
        ax.set_title(assay_type)
        ax.legend(fontsize=7)

    axes[0].set_ylabel('density')
    plt.tight_layout()
    return fig, axes


def plot_metric_condition_violin_across_assays(
        condition_assay_dfs, metric,
        focal_flies_map=None,
        assay_colors=None,
        ylim_percentile=95,
        figsize=None):
    '''
    Grouped violin: x=assay_type, y=metric, hue=condition.

    Arguments:
        condition_assay_dfs -- dict {condition_label: {assay_type: df}}
        metric              -- column name to plot on y-axis

    Keyword Arguments:
        focal_flies_map  -- dict mapping triad_type to focal fly id list (default: None)
        assay_colors     -- not used for violin colors (conditions get their own palette);
                            kept for API consistency (default: None)
        ylim_percentile  -- clip y-axis to this percentile (default: 95)
        figsize          -- figure size (default: auto)

    Returns:
        fig, ax
    '''
    condition_labels = list(condition_assay_dfs.keys())
    all_assay_types = sorted({at for cdf in condition_assay_dfs.values() for at in cdf.keys()})

    rows = []
    for cond_label, cdf in condition_assay_dfs.items():
        for assay_type, df in cdf.items():
            if metric not in df.columns or len(df) == 0:
                continue
            triad_type = df['triad_type'].iloc[0] if 'triad_type' in df.columns else None
            focal_flies = focal_flies_map.get(triad_type) if focal_flies_map and triad_type else None
            vals = _filter_by_focal_fly(df, focal_flies).drop_duplicates(
                ['frame', 'pair', 'acquisition'])[metric].dropna()
            if len(vals) == 0:
                continue
            rows.append(pd.DataFrame({
                'value': vals.values,
                'assay_type': assay_type,
                'condition': cond_label,
            }))

    if not rows:
        return None, None

    data = pd.concat(rows, ignore_index=True)
    y_max = (np.percentile(data['value'].dropna(), ylim_percentile)
             if ylim_percentile is not None else None)

    n_assays = len(all_assay_types)
    n_conds = len(condition_labels)
    if figsize is None:
        figsize = (max(6, n_assays * n_conds * 1.5), 5)

    fig, ax = plt.subplots(figsize=figsize)

    cond_palette = sns.color_palette('Set2', n_conds)
    palette = [cond_palette[i] for i in range(n_conds)]

    sns.violinplot(
        data=data, x='assay_type', y='value',
        hue='condition',
        order=all_assay_types, hue_order=condition_labels,
        palette=palette,
        ax=ax, inner='box', cut=0, linewidth=0.8,
        density_norm='width',
    )

    ax.set_xlabel('assay type')
    ax.set_ylabel(_metric_label(metric))
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig, ax


def plot_courtship_multiplicity_fraction(assay_dfs, focal_flies_map, triad='MMF',
                                         action_col='courtship', assay_colors=None,
                                         save_dir=None, save_name=None, figsize=None):
    """Fraction of courtship frames that are 2-male vs 1-male, per `triad` assay.

    For each assay of triad_type==`triad` (e.g. the MMF assays), computes the
    per-acquisition fraction of courtship frames (>=1 focal male courting) with both
    focal males courting (2M) vs exactly one (1M). Draws a stacked mean bar per assay
    (2M in the assay color, 1M a lighter shade; the two sum to 1) with per-acquisition
    2M-fraction points overlaid so the spread across acquisitions is visible.
    """
    keys = sorted(k for k, df in assay_dfs.items()
                  if df['triad_type'].iloc[0] == triad)
    if not keys:
        print(f"[warn] no {triad} assays for courtship-multiplicity plot")
        return None, None
    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in keys}

    per = {k: tutil.courtship_multiplicity_per_acquisition(
                 assay_dfs[k], focal_flies_map.get(triad) if focal_flies_map else None,
                 action_col)
           for k in keys}

    if figsize is None:
        figsize = (1.7 * len(keys) + 1.5, 4.5)
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        p = per[k]
        if p.empty:
            continue
        base = assay_colors.get(k, putil.courtship_color(k))
        light = putil.lighten(base, 0.45)
        m2 = float(p['frac_2M'].mean())
        ax.bar(i, m2, width=0.7, color=base, edgecolor='white', lw=0.5, zorder=1,
               label='2M (both males)' if i == 0 else None)
        ax.bar(i, 1 - m2, bottom=m2, width=0.7, color=light, edgecolor='white', lw=0.5,
               zorder=1, label='1M (one male)' if i == 0 else None)
        jit = (rng.random(len(p)) - 0.5) * 0.3
        ax.scatter(i + jit, p['frac_2M'], s=30, color='black', alpha=0.75, zorder=3,
                   edgecolor='white', linewidths=0.3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction of courtship frames')
    ax.set_title(f'{triad}: 2-male vs 1-male courtship\n(points = per-acquisition 2M fraction)')
    ax.legend(title='# focal males courting', fontsize=8, loc='upper right')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, save_name or 'courtship_multiplicity_fraction.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, ax


# ── metric-vs-metric relationship (e.g. distance vs |θ error|) ────────────────

def _courtship_xy_table(adf, x_metric, y_metric, action_col, focal_flies):
    """Courtship target-pair frames for the focal flies → DataFrame[acquisition, x, y].
    Uses the same focal-fly + action + target-pair filtering as the 'target-only' metric
    condition, so x/y are read on the focal↔pursued-target pair row each courtship frame."""
    filt = _filter_by_focal_fly(adf, focal_flies)
    tp = _filter_to_target_pairs(_select_action_frames(filt, action_col), action_col)
    if tp is None or len(tp) == 0:
        return None
    cols = ['acquisition', x_metric, y_metric]
    if any(c not in tp.columns for c in cols):
        return None
    return tp[cols].dropna()


def _assay_species_scenario_grid(keys):
    """(species_rows, scenario_cols) ordering for split/4-way assay keys
    ('Dmel_MFF', 'Dmel_MMF_1M', ...)."""
    species = sorted({k.split('_', 1)[0] for k in keys})
    order = ['MFF', 'MMF', 'MMF_1M', 'MMF_2M', 'MMF_2M_F']
    scens = [s for s in order if any(f'{sp}_{s}' in keys for sp in species)]
    scens += sorted({k.split('_', 1)[1] for k in keys if '_' in k} - set(scens))
    return species, scens


def _xy_assay_data(assay_dfs, x_metric, y_metric, action_col, focal_flies_map):
    data = {}
    for key in sorted(assay_dfs):
        adf = assay_dfs[key]
        triad = adf['triad_type'].iloc[0] if 'triad_type' in adf.columns else None
        focal = focal_flies_map.get(triad) if focal_flies_map else None
        t = _courtship_xy_table(adf, x_metric, y_metric, action_col, focal)
        if t is not None and len(t) > 2:
            data[key] = t
    return data


def _bin_edges(xmin, xmax, n_bins, bin_width):
    """Bin edges/centers: fixed-width (round) edges from 0 when bin_width is given
    (e.g. 1 mm → 0, 1, 2, …), else n_bins equal-width bins over [xmin, xmax]."""
    if bin_width:
        hi = np.ceil(xmax / bin_width) * bin_width
        edges = np.arange(0.0, hi + bin_width * 0.5, bin_width)
        if len(edges) < 2:
            edges = np.array([0.0, float(bin_width)])
    else:
        edges = np.linspace(xmin, xmax, n_bins + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def plot_metric_xy_scatter_across_assays(assay_dfs, x_metric='dist_to_other_body_adj',
                                         y_metric='abs_theta_error_deg', action_col='courtship',
                                         focal_flies_map=None, assay_colors=None,
                                         x_lim=None, y_lim=None, n_bins=15, bin_width=None,
                                         max_points=8000,
                                         save_dir=None, save_name=None, figsize=None):
    """Per-assay scatter of y_metric vs x_metric over courtship target-pair frames (focal
    flies), with a binned-mean trend line overlaid. One panel per assay (species rows ×
    scenario cols). Points light grey (subsampled); trend = mean y per x-bin in the assay
    color. Saves '<y>_vs_<x>_scatter.png' (or save_name)."""
    data = _xy_assay_data(assay_dfs, x_metric, y_metric, action_col, focal_flies_map)
    if not data:
        print(f"[warn] no courtship data for {y_metric} vs {x_metric} scatter")
        return None, None
    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in data}
    allx = np.concatenate([d[x_metric].to_numpy() for d in data.values()])
    ally = np.concatenate([d[y_metric].to_numpy() for d in data.values()])
    xmin, xmax = (x_lim if x_lim else (min(0.0, float(np.nanpercentile(allx, 1))),
                                       float(np.nanpercentile(allx, 99))))
    ymin, ymax = (y_lim if y_lim else (min(0.0, float(np.nanpercentile(ally, 1))),
                                       float(np.nanpercentile(ally, 99))))
    edges, centers = _bin_edges(xmin, xmax, n_bins, bin_width)
    n_bins = len(edges) - 1
    species, scens = _assay_species_scenario_grid(list(data))
    nrows, ncols = len(species), len(scens)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (3.5 * ncols, 3.2 * nrows),
                             squeeze=False, sharex=True, sharey=True, constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, sp in enumerate(species):
        for j, sc in enumerate(scens):
            ax = axes[i][j]
            d = data.get(f'{sp}_{sc}')
            if d is None:
                ax.axis('off')
                continue
            x = d[x_metric].to_numpy(); y = d[y_metric].to_numpy()
            color = assay_colors.get(f'{sp}_{sc}', putil.courtship_color(f'{sp}_{sc}'))
            xs, ys = x, y
            if len(x) > max_points:
                idx = rng.choice(len(x), max_points, replace=False)
                xs, ys = x[idx], y[idx]
            ax.scatter(xs, ys, s=4, alpha=0.12, color='0.7', edgecolors='none',
                       rasterized=True, zorder=1)
            bi = np.digitize(x, edges) - 1
            mean = np.array([y[bi == b].mean() if (bi == b).any() else np.nan
                             for b in range(n_bins)])
            ax.plot(centers, mean, color=color, lw=2.4, zorder=3)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_title(f'{sp}_{sc}\n(n={len(x)})', fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel(_metric_label(x_metric))
            if j == 0:
                ax.set_ylabel(_metric_label(y_metric))
    fig.suptitle(f'{_metric_label(y_metric)} vs {_metric_label(x_metric)} during courtship '
                 f'(grey = frames; line = binned mean)', fontsize=13)
    if save_dir is not None:
        name = save_name or f'{y_metric}_vs_{x_metric}_scatter.png'
        path = os.path.join(save_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def plot_metric_xy_binned_across_assays(assay_dfs, x_metric='dist_to_other_body_adj',
                                        y_metric='abs_theta_error_deg', action_col='courtship',
                                        focal_flies_map=None, assay_colors=None,
                                        x_lim=None, n_bins=15, bin_width=None, min_acq=2,
                                        style='band', save_dir=None, save_name=None, figsize=None):
    """Binned x_metric → mean y_metric ± SEM ACROSS ACQUISITIONS (acquisition = unit of
    replication). One panel per species (shared x/y axes), with that species' scenarios
    overlaid. Courtship target-pair frames, focal flies. Per bin: each acquisition's mean y,
    then mean ± SEM across acquisitions (bins with < min_acq acquisitions dropped).

    style -- 'band' (mean line + shaded ±SEM, the default) or 'points' (individual
    per-acquisition points jittered around each bin + a mean ±SEM error bar, scenarios
    dodged sideways). Saves '<y>_vs_<x>_binned.png' for 'band' / '..._binned_points.png'
    for 'points' (or save_name)."""
    data = _xy_assay_data(assay_dfs, x_metric, y_metric, action_col, focal_flies_map)
    if not data:
        print(f"[warn] no courtship data for {y_metric} vs {x_metric} binned plot")
        return None, None
    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in data}
    allx = np.concatenate([d[x_metric].to_numpy() for d in data.values()])
    xmin, xmax = (x_lim if x_lim else (min(0.0, float(np.nanpercentile(allx, 1))),
                                       float(np.nanpercentile(allx, 99))))
    edges, centers = _bin_edges(xmin, xmax, n_bins, bin_width)
    n_bins = len(edges) - 1
    species, scens = _assay_species_scenario_grid(list(data))

    ncols = len(species)
    fig, axes = plt.subplots(1, ncols, figsize=figsize or (5.0 * ncols, 5.0),
                             squeeze=False, sharex=True, sharey=True, constrained_layout=True)
    axes = axes[0]
    rng = np.random.default_rng(0)
    binwidth = edges[1] - edges[0]
    # for 'points': dodge each scenario sideways within a bin so they don't overlap
    offsets = ((np.arange(len(scens)) - (len(scens) - 1) / 2)
               * (0.55 * binwidth / max(len(scens), 1)))
    for ax, sp in zip(axes, species):
        for k, sc in enumerate(scens):
            key = f'{sp}_{sc}'
            d = data.get(key)
            if d is None:
                continue
            d = d.copy()
            d['_bin'] = np.digitize(d[x_metric].to_numpy(), edges) - 1
            d = d[(d['_bin'] >= 0) & (d['_bin'] < n_bins)]
            if d.empty:
                continue
            per_acq = d.groupby(['acquisition', '_bin'])[y_metric].mean().reset_index()
            stat = per_acq.groupby('_bin')[y_metric].agg(['mean', 'sem', 'count'])
            stat = stat[stat['count'] >= min_acq]
            if stat.empty:
                continue
            m = stat['mean'].to_numpy()
            se = np.nan_to_num(stat['sem'].to_numpy())
            color = assay_colors.get(key, putil.courtship_color(key))
            label = f'{sc} (n_acq={per_acq["acquisition"].nunique()})'
            if style == 'points':
                xoff = offsets[k]
                pa = per_acq[per_acq['_bin'].isin(stat.index)]
                jit = (rng.random(len(pa)) - 0.5) * 0.25 * binwidth
                ax.scatter(centers[pa['_bin'].to_numpy()] + xoff + jit, pa[y_metric].to_numpy(),
                           s=10, color=color, alpha=0.4, edgecolors='none', zorder=2)
                ax.errorbar(centers[stat.index.to_numpy()] + xoff, m, yerr=se, color=color,
                            fmt='o-', ms=4, lw=1.5, capsize=3, zorder=3, label=label)
            else:                                            # 'band'
                c = centers[stat.index.to_numpy()]
                ax.fill_between(c, m - se, m + se, color=color, alpha=0.2, lw=0, zorder=2)
                ax.plot(c, m, color=color, lw=2, marker='o', ms=3, zorder=3, label=label)
        ax.set_xlim(xmin, xmax)
        ax.set_title(sp, fontsize=11)
        ax.set_xlabel(_metric_label(x_metric))
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f'mean {_metric_label(y_metric)}')
    fig.suptitle(f'{_metric_label(y_metric)} vs {_metric_label(x_metric)} during courtship '
                 f'(mean ± SEM across acquisitions)', fontsize=13)
    if save_dir is not None:
        suf = '_points' if style == 'points' else ''
        name = save_name or f'{y_metric}_vs_{x_metric}_binned{suf}.png'
        path = os.path.join(save_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes
