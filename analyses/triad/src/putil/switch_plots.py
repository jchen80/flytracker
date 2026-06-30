import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Patch, Rectangle
from scipy.stats import linregress, gaussian_kde
import cv2
from adjustText import adjust_text
import glob
import numpy as np
import os
import re
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns


from analyses.triad.src import util as tutil
import libs.plotting as putil   # shared Ruta-lab courtship color scheme


def _nice_ring_step(max_r):
    """Return a round ring interval that gives 3–5 rings up to max_r."""
    for step in [0.5, 1, 2, 5, 10, 20, 50]:
        if max_r / step <= 5:
            return step
    return round(max_r / 4)


def _count_label(n_events, n_acq):
    """Consistent sample-size label: switch events and contributing acquisitions."""
    return f'n={n_events}, n_assays={n_acq}'


def _n_acq(df):
    """Number of distinct acquisitions in df (0 if no 'acquisition' column)."""
    return df['acquisition'].nunique() if 'acquisition' in df.columns else 0


def _metric_label(col):
    """Friendly axis/colorbar label (with units) for a metric column name."""
    name = col
    for pre in ('new_', 'old_', 'delta_'):
        if name.startswith(pre):
            name = name[len(pre):]
            break
    if 'target_ang_vel_fov' in name:
        base = 'target ang. vel. in FOV'
        if 'signed' in name:
            base += ' (signed, +prog/−regr)'
        unit = '°/s' if name.endswith('_deg') else 'rad/s'
        return f'{base} ({unit})'
    return col


def _scale_radius(r_vals, scale_percentile):
    """Radius used to size the egocentric view / ring extent.

    scale_percentile=None -> max radius (no clipping, every point in view);
    otherwise that percentile of the radial distances (a few outliers fall
    outside the axes so the bulk fills the view).
    """
    r_vals = np.asarray(r_vals)
    if r_vals.size == 0:
        return None
    return float(np.max(r_vals) if scale_percentile is None
                 else np.percentile(r_vals, scale_percentile))


SWITCH_CASE_COLORS = {'new_lower': 'mediumseagreen',
                      'similar': 'darkgray',
                      'new_higher': 'orchid'}


def _switch_case_labels(threshold_deg):
    """Human-readable case labels showing the actual threshold and θ notation."""
    thr = f'{threshold_deg:g}°'
    return {'new_higher': f'θ_new − θ_old > {thr}',
            'similar':    f'|θ_new − θ_old| < {thr}',
            'new_lower':  f'θ_old − θ_new > {thr}'}


def plot_switch_case_counts_across_assays(classified_df, threshold_deg=15.0,
                                          case_order=('new_lower', 'similar', 'new_higher'),
                                          case_colors=None, case_labels=None,
                                          normalize=False, figsize=(7, 5),
                                          save_dir=None):
    '''
    Grouped-bar count of switch events per theta-error case, compared across assays.

    classified_df must have columns 'assay_type' and 'switch_case' (from
    tutil.classify_switches_by_theta_error). Bars are grouped by assay on the x
    axis with one colored bar per switch_case; raw counts are annotated above each
    bar. If normalize=True the bar height is the fraction of that assay's switches
    in each case (annotations stay the raw counts).

    Arguments:
        classified_df -- DataFrame with 'assay_type' and 'switch_case' columns

    Keyword Arguments:
        threshold_deg -- threshold used for the labels (default: 15.0); pass the
                         same value used in classify_switches_by_theta_error
        case_order    -- order of cases left→right within each assay group
        case_colors   -- dict {case: color} (default: module SWITCH_CASE_COLORS)
        case_labels   -- dict {case: legend label} (default: from threshold_deg)
        normalize     -- plot per-assay fraction instead of raw count (default: False)
        figsize       -- figure size (default: (7, 5))
        save_dir      -- directory to save figure (default: None)

    Returns:
        fig, ax  (or None, None if empty)
    '''
    if classified_df is None or len(classified_df) == 0:
        print("  plot_switch_case_counts_across_assays: empty input.")
        return None, None
    if case_colors is None:
        case_colors = SWITCH_CASE_COLORS
    if case_labels is None:
        case_labels = _switch_case_labels(threshold_deg)

    present = set(classified_df['switch_case'])
    case_order = [c for c in case_order if c in present]
    assay_order = sorted(classified_df['assay_type'].unique())

    counts = (classified_df.groupby(['assay_type', 'switch_case']).size()
              .reset_index(name='count'))
    totals = counts.groupby('assay_type')['count'].sum()

    fig, ax = plt.subplots(figsize=figsize)
    n_cases = max(1, len(case_order))
    width = 0.8 / n_cases
    x = np.arange(len(assay_order))
    for i, case in enumerate(case_order):
        raw = [int(counts[(counts['assay_type'] == a)
                          & (counts['switch_case'] == case)]['count'].sum())
               for a in assay_order]
        if normalize:
            heights = [r / totals[a] if totals.get(a, 0) else 0
                       for r, a in zip(raw, assay_order)]
        else:
            heights = raw
        bars = ax.bar(x + (i - (n_cases - 1) / 2) * width, heights, width,
                      color=case_colors.get(case, 'gray'),
                      label=case_labels.get(case, case), edgecolor='white')
        for b, r in zip(bars, raw):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(r),
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}\n(n={int(totals.get(a, 0))})' for a in assay_order])
    ax.set_xlabel('assay type')
    ax.set_ylabel('fraction of switches' if normalize else 'number of switches')
    ax.set_title('Switch counts by Δ|theta error| case')
    ax.legend(fontsize=8, title='case')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, 'switch_case_counts_across_assays.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, ax


def plot_switch_vectors_across_assays(vector_dfs, ppm_dict,
                                       focal_flies_map=None,
                                       old_color='tomato', new_color='dodgerblue',
                                       assay_colors=None,
                                       event_alpha=0.45, event_lw=0.8,
                                       scale_percentile=None, vectors_only=False,
                                       save_dir=None, figsize=None):
    '''
    Show old and new target positions at courtship switch time.
    Three rows × one column per assay type, all Cartesian (egocentric, focal fly at origin):
      Row 1 — old target positions only (open circles)
      Row 2 — new target positions only (filled circles)
      Row 3 — both connected by thin gray lines

    When vectors_only=True, instead draws just row 3 as a standalone single row:
    old (grey open circles) + new (assay-coloured) positions connected by thin
    grey lines, one panel per assay.

    Arguments:
        vector_dfs -- dict mapping assay_type to DataFrame from
                      tutil.get_switch_frame_vectors
        ppm_dict   -- dict mapping assay_type to pixels-per-mm

    Keyword Arguments:
        focal_flies_map -- dict mapping triad_type to focal fly id list (default: None)
        old_color       -- color for old-target markers (default: 'tomato')
        new_color       -- color for new-target markers (default: 'dodgerblue')
        assay_colors    -- dict mapping assay_type to color overriding new_color (default: None)
        event_alpha     -- alpha for individual markers/lines (default: 0.45)
        event_lw        -- line width for connecting lines in row 3 (default: 0.8)
        save_dir        -- directory to save figure (default: None)
        figsize         -- figure size tuple (default: auto)

    Returns:
        fig, axes  (shape (3, n_assays))
    '''
    assay_types = sorted(vector_dfs.keys())
    n_cols = len(assay_types)
    if n_cols == 0:
        return None, np.empty((3, 0))
    if figsize is None:
        figsize = (5 * n_cols, 5.5) if vectors_only else (5 * n_cols, 14)

    # ── Pre-pass: filter and compute global scale ─────────────────────────────
    assay_data = {}
    global_r_vals = []
    for assay_type in assay_types:
        vdf = vector_dfs[assay_type].copy()
        ppm = ppm_dict.get(assay_type, 1)
        if focal_flies_map is not None and 'triad_type' in vdf.columns:
            focal_flies = focal_flies_map.get(vdf['triad_type'].iloc[0])
            if focal_flies is not None:
                vdf = vdf[vdf['id'].isin(focal_flies)]
        if len(vdf) == 0:
            assay_data[assay_type] = None
            continue
        old_x = vdf['old_x'].values / ppm
        old_y = vdf['old_y'].values / ppm
        new_x = vdf['new_x'].values / ppm
        new_y = vdf['new_y'].values / ppm
        assay_data[assay_type] = (old_x, old_y, new_x, new_y, len(vdf), _n_acq(vdf))
        global_r_vals.extend([np.hypot(old_x, old_y), np.hypot(new_x, new_y)])

    all_r = np.concatenate(global_r_vals) if global_r_vals else np.array([1.0])
    global_max_r     = _scale_radius(all_r, scale_percentile)
    global_ring_step = _nice_ring_step(global_max_r)
    global_plot_r    = np.max(all_r) * 1.05
    theta_ring       = np.linspace(0, 2 * np.pi, 300)

    def _style_cart(ax, col_idx, n, n_acq, row_label):
        for r_ring in np.arange(global_ring_step,
                                global_max_r + global_ring_step, global_ring_step):
            ax.plot(r_ring * np.cos(theta_ring), r_ring * np.sin(theta_ring),
                    color='gray', lw=0.5, alpha=0.3, ls='--', zorder=0)
            ax.text(0, r_ring, f'{r_ring:.0f}', fontsize=7, color='gray',
                    ha='center', va='bottom', alpha=0.6)
        ax.scatter(0, 0, s=60, color='white', zorder=7, marker='o',
                   linewidths=0.8, edgecolors='gray')
        ax.set_xlim(-global_plot_r, global_plot_r)
        ax.set_ylim(-global_plot_r, global_plot_r)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.25)
        ax.axvline(0, color='gray', lw=0.4, alpha=0.25)
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)')
        if col_idx == 0:
            ax.set_ylabel(f'{row_label}\ny (mm)')
        ax.set_title(f'{assay_types[col_idx]}  ({_count_label(n, n_acq)})')

    # ── vectors-only mode: standalone row-3 (old + new + connecting line) ─────
    if vectors_only:
        fig = plt.figure(figsize=figsize)
        axes_row = []
        for col_idx, assay_type in enumerate(assay_types):
            ax = fig.add_subplot(1, n_cols, col_idx + 1)
            axes_row.append(ax)
            if assay_data[assay_type] is None:
                ax.set_visible(False)
                continue
            old_x, old_y, new_x, new_y, n, n_acq = assay_data[assay_type]
            this_new_color = (assay_colors.get(assay_type, new_color)
                              if assay_colors is not None else new_color)
            segs = [[(ox, oy), (nx, ny)]
                    for ox, oy, nx, ny in zip(old_x, old_y, new_x, new_y)]
            ax.add_collection(LineCollection(segs, colors='gray',
                              linewidths=event_lw, alpha=event_alpha, zorder=1))
            ax.scatter(old_x, old_y, s=16, facecolors='none', edgecolors=old_color,
                       alpha=min(event_alpha * 2, 1.0), linewidths=1.0, zorder=2,
                       label='old target')
            ax.scatter(new_x, new_y, s=16, color=this_new_color,
                       alpha=min(event_alpha * 2, 1.0), linewidths=0, zorder=2,
                       label='new target')
            _style_cart(ax, col_idx, n, n_acq, 'old → new')
            if col_idx == 0:
                ax.legend(fontsize=7, loc='upper right')
        fig.suptitle('Target positions at switch: old → new (egocentric)', fontsize=12)
        plt.tight_layout()
        if save_dir is not None:
            savepath = os.path.join(save_dir, 'switch_target_oldnew_lines.png')
            fig.savefig(savepath, dpi=150, bbox_inches='tight')
            print(f"Saved to {savepath}")
        return fig, np.array(axes_row).reshape(1, n_cols)

    # ── Subplots ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    row1_axes, row2_axes, row3_axes = [], [], []
    for col_idx in range(n_cols):
        row1_axes.append(fig.add_subplot(3, n_cols, col_idx + 1))
        row2_axes.append(fig.add_subplot(3, n_cols, n_cols + col_idx + 1))
        row3_axes.append(fig.add_subplot(3, n_cols, 2 * n_cols + col_idx + 1))

    for col_idx, assay_type in enumerate(assay_types):
        ax1, ax2, ax3 = row1_axes[col_idx], row2_axes[col_idx], row3_axes[col_idx]
        if assay_data[assay_type] is None:
            for ax in [ax1, ax2, ax3]:
                ax.set_visible(False)
            continue

        old_x, old_y, new_x, new_y, n, n_acq = assay_data[assay_type]
        this_new_color = (assay_colors.get(assay_type, new_color)
                          if assay_colors is not None else new_color)

        # Row 1: old only
        ax1.scatter(old_x, old_y, s=16, facecolors='none', edgecolors=old_color,
                    alpha=min(event_alpha * 2, 1.0), linewidths=1.0, zorder=2)
        _style_cart(ax1, col_idx, n, n_acq, 'old target')

        # Row 2: new only
        ax2.scatter(new_x, new_y, s=16, color=this_new_color,
                    alpha=min(event_alpha * 2, 1.0), linewidths=0, zorder=2)
        _style_cart(ax2, col_idx, n, n_acq, 'new target')

        # Row 3: old + new + connecting lines
        segs = [[(ox, oy), (nx, ny)]
                for ox, oy, nx, ny in zip(old_x, old_y, new_x, new_y)]
        lc = LineCollection(segs, colors='gray', linewidths=event_lw,
                            alpha=event_alpha, zorder=1)
        ax3.add_collection(lc)
        ax3.scatter(old_x, old_y, s=16, facecolors='none', edgecolors=old_color,
                    alpha=min(event_alpha * 2, 1.0), linewidths=1.0, zorder=2,
                    label='old target')
        ax3.scatter(new_x, new_y, s=16, color=this_new_color,
                    alpha=min(event_alpha * 2, 1.0), linewidths=0, zorder=2,
                    label='new target')
        _style_cart(ax3, col_idx, n, n_acq, 'old → new')
        if col_idx == 0:
            ax3.legend(fontsize=7, loc='upper right')

    fig.suptitle('Target positions at switch (egocentric)', fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, 'switch_target_vectors.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    axes = np.array(row1_axes + row2_axes + row3_axes).reshape(3, n_cols)
    return fig, axes


def plot_switch_rate_across_assays(assay_dfs, action_col='courtship', fps=60,
                                   focal_flies_map=None, norm_minutes=30,
                                   switch_source='prefer_manual',
                                   assay_colors=None, save_dir=None, figsize=(6, 5)):
    '''
    Plot target switch count or rate compared across assay types.

    When norm_minutes is set: switch rate = switches / courtship duration,
    expressed as switches per norm_minutes of courtship.
    When norm_minutes is None: raw switch count per acquisition.

    Switch events are defined exactly as in tutil.get_switch_frame_vectors (using
    switch_source — manual 'switching' annotations preferred when present — and
    requiring a valid previous target and non-NaN positions), counting only focal
    flies. This keeps the counts here consistent with the position/trajectory plots.

    Arguments:
        assay_dfs  -- dict mapping assay type to its dataframe
        action_col -- base action column name (e.g. 'courtship')

    Keyword Arguments:
        fps             -- frames per second; required when norm_minutes is set (default: 60)
        focal_flies_map -- dict mapping triad_type to list of focal fly ids (default: None)
        norm_minutes    -- normalize to this many minutes of courtship; if None plots
                          raw switch count per acquisition (default: 30)
        switch_source   -- 'auto', 'manual', or 'prefer_manual' (default: 'prefer_manual')
        assay_colors    -- dict mapping assay type to color (default: None)
        save_dir        -- directory to save figure (default: None)
        figsize         -- figure size (default: (6, 5))

    Returns:
        fig, ax
    '''
    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in sorted(assay_dfs.keys())}

    records = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns:
            print(f"  '{action_col}' not in assay '{assay_type}', skipping.")
            continue

        triad_type = assay_df['triad_type'].iloc[0]
        focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None

        # Switch events use the same definition (and switch_source) as the
        # position/trajectory plots, so the counts match.
        events = tutil.get_switch_frame_vectors(
            assay_df, action_col=action_col, switch_source=switch_source)
        if focal_flies is not None and len(events) > 0:
            events = events[events['id'].isin(focal_flies)]
        ev_per_acq = (events.groupby('acquisition').size()
                      if len(events) > 0 else pd.Series(dtype=int))

        for acq_name, acq_df in assay_df.groupby('acquisition'):
            if focal_flies is not None:
                acq_df = acq_df[acq_df['id'].isin(focal_flies)]

            n_switches = int(ev_per_acq.get(acq_name, 0))
            dedup = acq_df.drop_duplicates(['frame', 'id'])

            if norm_minutes is not None:
                courtship_frames = (dedup[action_col] == dedup['id']).sum()
                if courtship_frames == 0:
                    continue
                value = n_switches / (courtship_frames / fps / 60) * norm_minutes
            else:
                value = n_switches

            records.append({
                'assay_type': assay_type,
                'acquisition': acq_name,
                'value': value,
                'n_switches': n_switches,
            })

    if not records:
        print("No switch data to plot.")
        return None, None

    plot_data = pd.DataFrame(records)
    assay_order = sorted(assay_dfs.keys())
    palette = [assay_colors.get(a, putil.courtship_color(a)) for a in assay_order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_data, x='assay_type', y='value',
                order=assay_order, palette=palette, errorbar='se', ax=ax, alpha=0.7)
    sns.stripplot(data=plot_data, x='assay_type', y='value',
                  order=assay_order, palette=palette, size=7, jitter=True,
                  ax=ax, alpha=0.9, linewidth=0.5, edgecolor='white')

    n_per = plot_data.groupby('assay_type')['acquisition'].nunique()
    sw_per = plot_data.groupby('assay_type')['n_switches'].sum()
    ax.set_xticklabels([f'{a}\n({_count_label(int(sw_per.get(a, 0)), int(n_per.get(a, 0)))})'
                        for a in assay_order])
    ax.set_xlabel('assay type')

    if norm_minutes is not None:
        ylabel = f'switches per {norm_minutes} min courtship'
    else:
        ylabel = 'total switches'
    ax.set_ylabel(ylabel)
    ax.set_title(f'{action_col} {ylabel}')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'{action_col}_switch_rate_across_assays.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, ax


def plot_switch_trajectory_across_assays(traj_dfs, ppm_dict,
                                          fps=60,
                                          focal_flies_map=None,
                                          assay_colors=None,
                                          pre_color='dimgray',
                                          event_alpha=0.25,
                                          event_lw=0.7,
                                          show_speed=True,
                                          scale_percentile=None,
                                          figsize=None):
    '''
    Plot the ego-centric trajectory of the new target around each switch event.

    Mirrors the layout of plot_switch_vectors_across_assays: one column per assay,
    rows are Cartesian (top), polar (middle), and optionally relative-speed
    timecourse (bottom).

    Pre-switch portion of each trajectory is drawn in pre_color; post-switch in
    the assay colour.  The speed panel shows mean ± SE of the ego-centric
    relative speed (mm/s) computed from consecutive frame displacements.

    Arguments:
        traj_dfs -- dict {assay_type: df from tutil.get_switch_new_target_trajectories}
        ppm_dict -- dict {assay_type: pixels-per-mm}

    Keyword Arguments:
        fps             -- frames per second for speed calculation (default: 60)
        focal_flies_map -- dict {triad_type: [focal_fly_ids]} (default: None)
        assay_colors    -- dict {assay_type: color} for post-switch segments (default: auto)
        pre_color       -- color for pre-switch trajectory portion (default: 'dimgray')
        event_alpha     -- alpha for individual event lines (default: 0.25)
        event_lw        -- line width for individual event lines (default: 0.7)
        show_speed      -- add a third row with speed timecourse (default: True)
        figsize         -- figure size tuple (default: auto)

    Returns:
        fig, (cart_axes, polar_axes, speed_axes, theta_dt_axes, vel_axes,
              ang_dt_axes, ang_dt_signed_axes)
        — signal axes are [] if show_speed=False
    '''
    assay_types = sorted(traj_dfs.keys())
    n_cols = len(assay_types)
    if n_cols == 0:
        return None, ([], [], [], [], [], [], [])

    n_rows = 7 if show_speed else 2
    if figsize is None:
        figsize = (12 * n_cols, 6 * n_rows)

    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in assay_types}

    # ── Pre-pass: filter and collect global scale ─────────────────────────────
    assay_data = {}
    global_r_vals = []
    for assay_type in assay_types:
        tdf = traj_dfs[assay_type].copy()
        ppm = ppm_dict.get(assay_type, 1)
        if focal_flies_map is not None and 'triad_type' in tdf.columns:
            triad_type = tdf['triad_type'].iloc[0]
            focal_flies = focal_flies_map.get(triad_type)
            if focal_flies is not None:
                tdf = tdf[tdf['id'].isin(focal_flies)]
        if len(tdf) == 0:
            assay_data[assay_type] = None
            continue
        tdf = tdf.copy()
        tdf['x_mm'] = tdf['x_ego'] / ppm
        tdf['y_mm'] = tdf['y_ego'] / ppm
        n_events = tdf.drop_duplicates(['acquisition', 'id', 'switch_frame']).shape[0]
        assay_data[assay_type] = (tdf, ppm, n_events, _n_acq(tdf))
        traj_mask = tdf['t_rel'].between(-1.0, 0)
        r_vals = np.hypot(tdf.loc[traj_mask, 'x_mm'].values,
                          tdf.loc[traj_mask, 'y_mm'].values)
        global_r_vals.append(r_vals)

    all_r = np.concatenate(global_r_vals) if global_r_vals else np.array([1.0])
    global_max_r     = _scale_radius(all_r, scale_percentile)
    global_ring_step = _nice_ring_step(global_max_r)
    global_plot_r    = np.max(all_r) * 1.05
    theta_ring       = np.linspace(0, 2 * np.pi, 300)

    # ── Create subplots ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    cart_axes, polar_axes, speed_axes, theta_dt_axes, vel_axes = [], [], [], [], []
    ang_dt_axes, ang_dt_signed_axes = [], []
    for col_idx in range(n_cols):
        cart_axes.append(fig.add_subplot(n_rows, n_cols, col_idx + 1))
        polar_axes.append(fig.add_subplot(n_rows, n_cols, n_cols + col_idx + 1,
                                          projection='polar'))
        if show_speed:
            speed_axes.append(fig.add_subplot(n_rows, n_cols,
                                              2 * n_cols + col_idx + 1))
            theta_dt_axes.append(fig.add_subplot(n_rows, n_cols,
                                                 3 * n_cols + col_idx + 1))
            vel_axes.append(fig.add_subplot(n_rows, n_cols,
                                            4 * n_cols + col_idx + 1))
            ang_dt_axes.append(fig.add_subplot(n_rows, n_cols,
                                               5 * n_cols + col_idx + 1))
            ang_dt_signed_axes.append(fig.add_subplot(n_rows, n_cols,
                                                      6 * n_cols + col_idx + 1))

    # ── Draw per assay ────────────────────────────────────────────────────────
    for col_idx, assay_type in enumerate(assay_types):
        ax_cart   = cart_axes[col_idx]
        ax_pol    = polar_axes[col_idx]
        ax_spd    = speed_axes[col_idx]          if show_speed else None
        ax_tdt    = theta_dt_axes[col_idx]       if show_speed else None
        ax_vel    = vel_axes[col_idx]            if show_speed else None
        ax_adt    = ang_dt_axes[col_idx]         if show_speed else None
        ax_adts   = ang_dt_signed_axes[col_idx]  if show_speed else None

        if assay_data[assay_type] is None:
            for ax in ([ax_cart, ax_pol]
                       + ([ax_spd, ax_tdt, ax_vel, ax_adt, ax_adts] if show_speed else [])):
                ax.set_visible(False)
            continue

        tdf, ppm, n_events, n_acq = assay_data[assay_type]
        color = assay_colors.get(assay_type, putil.courtship_color(assay_type))

        event_groups = [
            (key, grp.sort_values('frame'))
            for key, grp in tdf.groupby(['acquisition', 'id', 'switch_frame'])
        ]

        # Individual event trajectories
        for _, ev_df in event_groups:
            x = ev_df['x_mm'].values
            y = ev_df['y_mm'].values
            t = ev_df['t_rel'].values
            r = np.hypot(x, y)
            theta = np.arctan2(y, x)

            pre  = (t >= -1.0) & (t <= 0)

            for ax, xd, yd in [(ax_cart, x, y), (ax_pol, theta, r)]:
                if pre.sum() > 1:
                    ax.plot(xd[pre],  yd[pre],  color=pre_color, alpha=event_alpha,
                            lw=event_lw)
                sw = t == 0
                if sw.any():
                    ax.scatter(xd[sw], yd[sw], s=10, color=color,
                               alpha=min(event_alpha * 3, 1.0), zorder=3)

        def _plot_signal(ax, tdf, new_col, old_col, ylabel, col_idx, n_events,
                         assay_type, new_color, old_color, fps, scale=1.0):
            """Plot mean±SE timecourse for new and old target in distinct colors."""
            trel = (tdf['t_rel'] * fps).round() / fps
            plotted = False
            for col, c, lbl in [(new_col, new_color, 'new target'),
                                 (old_col, old_color, 'old target')]:
                if col not in tdf.columns:
                    continue
                vals = tdf[col] * scale
                grp  = vals.groupby(trel)
                mean, se = grp.mean(), grp.sem().fillna(0)
                ax.plot(mean.index, mean.values, color=c, lw=1.5, label=lbl)
                ax.fill_between(mean.index,
                                mean.values - se.values,
                                mean.values + se.values,
                                color=c, alpha=0.2)
                plotted = True
            if plotted:
                ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.6)
                ax.axvline(0, color='white', ls='--', lw=0.8, alpha=0.5)
            ax.set_xlabel('time from switch (s)')
            ax.set_ylabel(ylabel if col_idx == 0 else '')
            ax.set_title(f'{assay_type}  ({_count_label(n_events, n_acq)})')
            if plotted and col_idx == 0:
                ax.legend(fontsize=7)

        # Row 3: rel_vel — new target in assay color, old target in pre_color
        if ax_spd is not None:
            if 'new_rel_vel' in tdf.columns or 'old_rel_vel' in tdf.columns:
                _plot_signal(ax_spd, tdf, 'new_rel_vel', 'old_rel_vel',
                             'rel. vel. (mm/s)', col_idx, n_events,
                             assay_type, color, pre_color, fps, scale=1.0 / ppm)
            else:
                ax_spd.set_visible(False)

        # Row 4: abs_theta_error_dt — new target in assay color, old target in pre_color
        if ax_tdt is not None:
            if 'new_abs_theta_error_dt' in tdf.columns or 'old_abs_theta_error_dt' in tdf.columns:
                _plot_signal(ax_tdt, tdf, 'new_abs_theta_error_dt', 'old_abs_theta_error_dt',
                             '|dθ|/dt (deg/s)', col_idx, n_events,
                             assay_type, color, pre_color, fps, scale=1.0)
            else:
                ax_tdt.set_visible(False)

        # Row 5: absolute fly velocity — new target in assay color, old target in pre_color
        if ax_vel is not None:
            if 'new_vel' in tdf.columns or 'old_vel' in tdf.columns:
                _plot_signal(ax_vel, tdf, 'new_vel', 'old_vel',
                             'vel (mm/s)', col_idx, n_events,
                             assay_type, color, pre_color, fps, scale=1.0)
            else:
                ax_vel.set_visible(False)

        # Row 6: target's angular velocity in focal FOV (target motion only)
        if ax_adt is not None:
            if ('new_target_ang_vel_fov_deg' in tdf.columns
                    or 'old_target_ang_vel_fov_deg' in tdf.columns):
                _plot_signal(ax_adt, tdf,
                             'new_target_ang_vel_fov_deg', 'old_target_ang_vel_fov_deg',
                             'target ang. vel.\nin FOV (°/s)', col_idx, n_events,
                             assay_type, color, pre_color, fps, scale=1.0)
            else:
                ax_adt.set_visible(False)

        # Row 7: signed by focal heading (positive = progressive, negative = regressive)
        if ax_adts is not None:
            if ('new_target_ang_vel_fov_signed_deg' in tdf.columns
                    or 'old_target_ang_vel_fov_signed_deg' in tdf.columns):
                _plot_signal(ax_adts, tdf,
                             'new_target_ang_vel_fov_signed_deg',
                             'old_target_ang_vel_fov_signed_deg',
                             'target ang. vel. signed\n(°/s, +prog/−regr)', col_idx, n_events,
                             assay_type, color, pre_color, fps, scale=1.0)
            else:
                ax_adts.set_visible(False)

        # Cartesian styling (mirrors plot_switch_vectors_across_assays)
        for r_ring in np.arange(global_ring_step,
                                global_max_r + global_ring_step,
                                global_ring_step):
            ax_cart.plot(r_ring * np.cos(theta_ring), r_ring * np.sin(theta_ring),
                         color='gray', lw=0.5, alpha=0.3, ls='--', zorder=0)
            ax_cart.text(0, r_ring, f'{r_ring:.0f}', fontsize=7, color='gray',
                         ha='center', va='bottom', alpha=0.6)
        ax_cart.scatter(0, 0, s=60, color='white', zorder=7, marker='o',
                        linewidths=0.8, edgecolors='gray')
        ax_cart.set_xlim(-global_plot_r, global_plot_r)
        ax_cart.set_ylim(-global_plot_r, global_plot_r)
        ax_cart.axhline(0, color='gray', lw=0.4, alpha=0.25)
        ax_cart.axvline(0, color='gray', lw=0.4, alpha=0.25)
        ax_cart.set_aspect('equal')
        ax_cart.set_xlabel('x (mm)')
        if col_idx == 0:
            ax_cart.set_ylabel('y (mm)')
        ax_cart.set_title(f'{assay_type}  ({_count_label(n_events, n_acq)})')

        ax_pol.set_theta_zero_location('E')
        ax_pol.set_theta_direction(1)
        ax_pol.set_ylim(0, global_plot_r)
        ax_pol.tick_params(labelsize=7)

    if cart_axes:
        from matplotlib.lines import Line2D
        cart_axes[0].legend(handles=[
            Line2D([0], [0], color=pre_color, lw=1.5, label='pre-switch'),
        ], fontsize=7, loc='upper right')

    fig.suptitle('New target trajectory around switch (ego-centric, ±2 s window)',
                 fontsize=12)
    plt.tight_layout()
    return fig, (cart_axes, polar_axes, speed_axes, theta_dt_axes, vel_axes,
                 ang_dt_axes, ang_dt_signed_axes)


def plot_switch_theta_error_delta_across_assays(switch_comparison_dfs, assay_colors=None,
                                                 xlim_percentile=None, figsize=None):
    '''
    Visualise the change in orientation error at each switch event.

    Left panel: histogram of (new − old) abs_theta_error_deg, one bar series per assay
        type, with a dashed reference line at x = 0.
        Negative values = switched toward better-oriented (lower-error) target.
    Right panels (one per assay): paired slope plot — each switch event as a thin line
        from old-target theta_error to new-target theta_error, with mean ± SE overlay.

    Arguments:
        switch_comparison_dfs -- dict {assay_type: DataFrame} with columns
            old_abs_theta_error_deg, new_abs_theta_error_deg, delta_abs_theta_error_deg
            (output of tutil.get_switch_theta_error_comparison, split by assay_type)

    Keyword Arguments:
        assay_colors    -- dict mapping assay_type to color (default: auto)
        xlim_percentile -- percentile used to clip x-axis on the density panel and
                           y-axis on the slope panels; None = no clipping (default: None)
        figsize         -- figure size tuple (default: auto)

    Returns:
        fig, axes
    '''
    assay_order = sorted(switch_comparison_dfs.keys())
    n_assays = len(assay_order)
    if n_assays == 0:
        return None, None

    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in assay_order}

    n_cols = 1 + n_assays
    if figsize is None:
        figsize = (4 * n_cols, 5)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    # ── Left panel: delta histogram ──────────────────────────────────────────
    ax = axes[0]
    all_delta_vals = [
        switch_comparison_dfs[at]['delta_abs_theta_error_deg'].dropna()
        for at in assay_order
    ]
    pooled = pd.concat(all_delta_vals).dropna() if all_delta_vals else pd.Series(dtype=float)
    if xlim_percentile is not None and len(pooled) > 0:
        lim = np.percentile(pooled.abs(), xlim_percentile)
    else:
        lim = pooled.abs().max() if len(pooled) > 0 else 180.0
    bin_edges = np.linspace(-lim, lim, 41)

    legend_handles = []
    for assay_type, vals in zip(assay_order, all_delta_vals):
        if len(vals) > 0:
            color = assay_colors.get(assay_type, putil.courtship_color(assay_type))
            n_acq = _n_acq(switch_comparison_dfs[assay_type].dropna(
                subset=['delta_abs_theta_error_deg']))
            label = f'{assay_type} ({_count_label(len(vals), n_acq)})'
            ax.hist(vals, bins=bin_edges, density=True, histtype='bar',
                    color=color, edgecolor='none', alpha=0.2)
            # bold outline of the raw histogram (no KDE smoothing)
            ax.hist(vals, bins=bin_edges, density=True, histtype='step',
                    color=color, linewidth=2.0)
            legend_handles.append(Patch(facecolor=color, edgecolor=color,
                                        label=label))

    ax.axvline(0, color='white', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('new − old  |θ error|  (deg)')
    ax.set_ylabel('density')
    ax.set_title('Δ |θ error| at switch')
    ax.legend(handles=legend_handles, fontsize=8)
    ax.set_xlim(-lim, lim)

    # ── Right panels: paired slope per assay ─────────────────────────────────
    slope_ylim = None
    if xlim_percentile is not None:
        slope_vals = pd.concat([
            pd.concat([df['old_abs_theta_error_deg'], df['new_abs_theta_error_deg']])
            for df in switch_comparison_dfs.values()
        ]).dropna()
        if len(slope_vals) > 0:
            slope_ylim = (0, np.percentile(slope_vals, xlim_percentile))

    for ax_i, assay_type in enumerate(assay_order):
        ax = axes[1 + ax_i]
        df = switch_comparison_dfs[assay_type].dropna(
            subset=['old_abs_theta_error_deg', 'new_abs_theta_error_deg'])
        color = assay_colors.get(assay_type, putil.courtship_color(assay_type))

        for _, row in df.iterrows():
            ax.plot([0, 1],
                    [row['old_abs_theta_error_deg'], row['new_abs_theta_error_deg']],
                    color=color, alpha=0.3, linewidth=0.8)

        if len(df) > 0:
            old_mean = df['old_abs_theta_error_deg'].mean()
            new_mean = df['new_abs_theta_error_deg'].mean()
            ax.plot([0, 1], [old_mean, new_mean], color=color, linewidth=2.5, zorder=5)
            ax.scatter([0, 1], [old_mean, new_mean], color=color, s=60, zorder=6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['old target', 'new target'])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f'{assay_type}\n({_count_label(len(df), _n_acq(df))})')
        ax.set_ylabel('|θ error|  (deg)' if ax_i == 0 else '')
        if slope_ylim is not None:
            ax.set_ylim(*slope_ylim)

    fig.suptitle('Orientation error: old vs new target at switch', fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_switch_target_ang_vel_fov_delta_across_assays(
        switch_comparison_dfs, assay_colors=None,
        xlim_percentile=None, figsize=None):
    '''
    Visualise the change in target_ang_vel_fov at each switch event.

    Left panel: histogram of (new − old) target_ang_vel_fov, one series per assay.
    Right panels (one per assay): paired slope plot — each switch event as a thin
    line from old-target to new-target value, with mean overlay.

    Arguments:
        switch_comparison_dfs -- dict {assay_type: DataFrame} output of
            tutil.get_switch_target_ang_vel_fov_comparison

    Keyword Arguments:
        assay_colors    -- dict mapping assay_type to color (default: auto)
        xlim_percentile -- clip x/y axes to this percentile; None = no clipping (default: None)
        figsize         -- figure size (default: auto)

    Returns:
        fig, axes
    '''
    assay_order = sorted(switch_comparison_dfs.keys())
    n_assays = len(assay_order)
    if n_assays == 0:
        return None, None

    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in assay_order}

    n_cols = 1 + n_assays
    if figsize is None:
        figsize = (4 * n_cols, 5)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    # ── Left panel: delta histogram ──────────────────────────────────────────
    ax = axes[0]
    all_delta = [switch_comparison_dfs[at]['delta_target_ang_vel_fov_signed_deg'].dropna()
                 for at in assay_order]
    pooled = pd.concat(all_delta).dropna() if all_delta else pd.Series(dtype=float)
    if xlim_percentile is not None and len(pooled) > 0:
        lim = np.percentile(pooled.abs(), xlim_percentile)
    else:
        lim = pooled.abs().max() if len(pooled) > 0 else 10.0
    bin_edges = np.linspace(-lim, lim, 41)

    legend_handles = []
    for assay_type, vals in zip(assay_order, all_delta):
        if len(vals) > 0:
            color = assay_colors.get(assay_type, putil.courtship_color(assay_type))
            n_acq = _n_acq(switch_comparison_dfs[assay_type].dropna(
                subset=['delta_target_ang_vel_fov_signed_deg']))
            label = f'{assay_type} ({_count_label(len(vals), n_acq)})'
            ax.hist(vals, bins=bin_edges, density=True, histtype='bar',
                    color=color, edgecolor='none', alpha=0.2)
            # bold outline of the raw histogram (no KDE smoothing)
            ax.hist(vals, bins=bin_edges, density=True, histtype='step',
                    color=color, linewidth=2.0)
            legend_handles.append(Patch(facecolor=color, edgecolor=color,
                                        label=label))

    ax.axvline(0, color='white', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('new − old  signed target ang. vel. in FOV  (°/s; + prog / − reg)')
    ax.set_ylabel('density')
    ax.set_title('Δ signed target ang. vel. in FOV at switch')
    ax.legend(handles=legend_handles, fontsize=8)
    ax.set_xlim(-lim, lim)

    # ── Right panels: paired slope per assay ─────────────────────────────────
    slope_ylim = None
    if xlim_percentile is not None:
        slope_vals = pd.concat([
            pd.concat([df['old_target_ang_vel_fov_signed_deg'],
                       df['new_target_ang_vel_fov_signed_deg']])
            for df in switch_comparison_dfs.values()
        ]).dropna()
        if len(slope_vals) > 0:
            lim_y = np.percentile(slope_vals.abs(), xlim_percentile)
            slope_ylim = (-lim_y, lim_y)

    for ax_i, assay_type in enumerate(assay_order):
        ax = axes[1 + ax_i]
        df = switch_comparison_dfs[assay_type].dropna(
            subset=['old_target_ang_vel_fov_signed_deg', 'new_target_ang_vel_fov_signed_deg'])
        color = assay_colors.get(assay_type, putil.courtship_color(assay_type))

        for _, row in df.iterrows():
            ax.plot([0, 1],
                    [row['old_target_ang_vel_fov_signed_deg'],
                     row['new_target_ang_vel_fov_signed_deg']],
                    color=color, alpha=0.3, linewidth=0.8)

        if len(df) > 0:
            old_mean = df['old_target_ang_vel_fov_signed_deg'].mean()
            new_mean = df['new_target_ang_vel_fov_signed_deg'].mean()
            ax.plot([0, 1], [old_mean, new_mean], color=color, linewidth=2.5, zorder=5)
            ax.scatter([0, 1], [old_mean, new_mean], color=color, s=60, zorder=6)

        ax.axhline(0, color='white', ls='--', lw=0.6, alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['old target', 'new target'])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f'{assay_type}\n({_count_label(len(df), _n_acq(df))})')
        ax.set_ylabel('signed target ang. vel. in FOV  (°/s; + prog / − reg)'
                      if ax_i == 0 else '')
        if slope_ylim is not None:
            ax.set_ylim(*slope_ylim)

    fig.suptitle('Signed target angular velocity in focal FOV: old vs new target at switch',
                 fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_switch_target_ang_vel_fov_vs_theta_error_across_assays(
        traj_assay_dfs,
        t_rel_points=((-0.2, '−0.2 s'), (-0.1, '−0.1 s'), (0.0, 'at switch')),
        x_col='new_theta_error_deg', y_col='new_target_ang_vel_fov_deg',
        xlabel='new target θ error (°)', ylabel='target ang. vel. in FOV (°/s)',
        focal_flies_map=None, assay_colors=None,
        xlim_percentile=98, ylim_percentile=98,
        xlim=None, ylim=None,
        shade_quadrants=True, prog_color='#00BB44', regr_color='#CC00CC',
        shade_alpha=0.22, figsize=None):
    '''
    Scatter of the new target's FOV angular velocity vs its orientation error, one
    column per time point relative to the switch.

    traj_assay_dfs must come from tutil.get_switch_new_target_trajectories (grouped
    by assay_type); it carries, per event, a window of frames with t_rel plus the
    new_/old_ metric columns. For each event the frame nearest each requested t_rel
    is used. Panels form a grid: one row per assay type, one column per time point
    (assays are not overlaid). A least-squares linear fit is drawn in each panel, and
    panels use a white background. Reference lines (x=0, y=0) and axis limits are
    shared across panels.

    When shade_quadrants is on, the background is shaded by progressive/regressive
    motion: same-sign quadrants (bottom-left and top-right, where x·y > 0, i.e.
    sign(theta_error)·target_ang_vel_fov > 0) are progressive (green); the
    opposite-sign quadrants (top-left, bottom-right) are regressive (magenta).

    Arguments:
        traj_assay_dfs -- dict {assay_type: df from tutil.get_switch_new_target_trajectories}

    Keyword Arguments:
        t_rel_points    -- iterable of (t_rel_seconds, label); one column per entry.
                           Each t_rel must lie within ±window_sec/2 of the
                           get_switch_new_target_trajectories call
                           (default: [(-0.2, '−0.2 s'), (-0.1, '−0.1 s'), (0, 'at switch')])
        x_col           -- x metric (default: 'new_theta_error_deg')
        y_col           -- y metric (default: 'new_target_ang_vel_fov')
        focal_flies_map -- dict {triad_type: [focal_fly_ids]} (default: None)
        assay_colors    -- dict {assay_type: color} (default: auto)
        xlim_percentile -- clip x-axis to this percentile of |x| (default: 98)
        ylim_percentile -- clip y-axis to this percentile of |y| (default: 98)
        xlim/ylim       -- explicit symmetric axis half-limits (override the
                           percentile auto-scaling; useful to keep per-assay figures
                           on the same scale) (default: None)
        shade_quadrants -- shade progressive (green) / regressive (magenta) quadrants
                           (default: True)
        prog_color/regr_color -- quadrant shading colors (default green/magenta)
        shade_alpha     -- quadrant shading opacity (default: 0.12)
        figsize         -- figure size (default: auto)

    Returns:
        fig, axes
    '''
    assay_types = sorted(traj_assay_dfs.keys())
    t_rel_points = list(t_rel_points)
    n_cols = len(t_rel_points)
    n_rows = len(assay_types)
    if n_cols == 0 or n_rows == 0:
        return None, None
    if assay_colors is None:
        assay_colors = {k: putil.courtship_color(k) for k in assay_types}
    if figsize is None:
        figsize = (4.6 * n_cols, 4.4 * n_rows)

    def _rows_at_t(tdf, t_target):
        '''One row per event nearest the requested t_rel (exact for t_rel==0).'''
        g = tdf.assign(_td=(tdf['t_rel'] - t_target).abs())
        idx = g.groupby(['acquisition', 'id', 'switch_frame'])['_td'].idxmin()
        return g.loc[idx]

    # Collect per (assay, t_idx) the x/y values; track ranges for shared limits
    panel_xy = {}
    all_x, all_y = [], []
    for assay_type in assay_types:
        tdf = traj_assay_dfs[assay_type].copy()
        if focal_flies_map is not None and 'triad_type' in tdf.columns:
            focal_flies = focal_flies_map.get(tdf['triad_type'].iloc[0])
            if focal_flies is not None:
                tdf = tdf[tdf['id'].isin(focal_flies)]
        if len(tdf) == 0 or x_col not in tdf.columns or y_col not in tdf.columns:
            for t_idx in range(n_cols):
                panel_xy[(assay_type, t_idx)] = None
            continue
        t_lo, t_hi = tdf['t_rel'].min(), tdf['t_rel'].max()
        for t_idx, (t_target, _label) in enumerate(t_rel_points):
            if t_target < t_lo or t_target > t_hi:
                print(f"  plot_switch_target_ang_vel_fov_vs_theta_error_across_assays: "
                      f"requested t_rel={t_target}s is outside the window "
                      f"[{t_lo:.2f}, {t_hi:.2f}]s for {assay_type} — using nearest frame.")
            sf = _rows_at_t(tdf, t_target).dropna(subset=[x_col, y_col])
            if len(sf) == 0:
                panel_xy[(assay_type, t_idx)] = None
                continue
            x, y = sf[x_col].values, sf[y_col].values
            panel_xy[(assay_type, t_idx)] = (x, y)
            all_x.append(x)
            all_y.append(y)

    if not all_x:
        return None, None
    px = np.concatenate(all_x)
    py = np.concatenate(all_y)
    if xlim is None:
        xlim = (np.percentile(np.abs(px), xlim_percentile)
                if xlim_percentile is not None else np.abs(px).max())
    if ylim is None:
        ylim = (np.percentile(np.abs(py), ylim_percentile)
                if ylim_percentile is not None else np.abs(py).max())

    def _shade(ax):
        # progressive (x·y > 0): bottom-left + top-right
        for x0, y0 in [(0, 0), (-xlim, -ylim)]:
            ax.add_patch(Rectangle((x0, y0), xlim, ylim, facecolor=prog_color,
                                   edgecolor='none', alpha=shade_alpha, zorder=0))
        # regressive (x·y < 0): top-left + bottom-right
        for x0, y0 in [(-xlim, 0), (0, -ylim)]:
            ax.add_patch(Rectangle((x0, y0), xlim, ylim, facecolor=regr_color,
                                   edgecolor='none', alpha=shade_alpha, zorder=0))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True,
                             squeeze=False)

    for r, assay_type in enumerate(assay_types):
        color = assay_colors.get(assay_type, putil.courtship_color(assay_type))
        for c, (t_target, t_label) in enumerate(t_rel_points):
            ax = axes[r, c]
            if shade_quadrants:
                _shade(ax)
            ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.6)
            ax.axvline(0, color='gray', lw=0.6, ls='--', alpha=0.6)

            xy = panel_xy.get((assay_type, c))
            if xy is not None:
                x, y = xy
                ax.scatter(x, y, color=color, s=18, alpha=0.7, linewidths=0, zorder=3)
                # least-squares linear fit
                if len(x) >= 2 and np.ptp(x) > 0:
                    lr = linregress(x, y)
                    xs = np.array([-xlim, xlim])
                    ax.plot(xs, lr.slope * xs + lr.intercept,
                            color='white', lw=1.8, zorder=4)
                    ax.text(0.03, 0.97,
                            f'slope={lr.slope:.2f}\nr={lr.rvalue:.2f}\n'
                            f'p={lr.pvalue:.2g}\nn={len(x)}',
                            transform=ax.transAxes, va='top', ha='left', fontsize=7,
                            color='white',
                            bbox=dict(boxstyle='round,pad=0.2', fc='black',
                                      ec='gray', alpha=0.6))

            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            if r == 0:
                ax.set_title(t_label, fontsize=11)
            if r == n_rows - 1:
                ax.set_xlabel(xlabel if xlabel is not None else x_col)
            if c == 0:
                ax.set_ylabel(f'{assay_type}\n{ylabel if ylabel is not None else y_col}')

    # progressive/regressive legend on the first panel
    if shade_quadrants:
        axes[0, 0].legend(handles=[
            Patch(facecolor=prog_color, alpha=max(shade_alpha * 2, 0.3), label='progressive'),
            Patch(facecolor=regr_color, alpha=max(shade_alpha * 2, 0.3), label='regressive'),
        ], fontsize=7, loc='upper right')

    fig.suptitle('New-target FOV angular velocity vs orientation error around switch',
                 fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_switch_positions_colored_by_metric_across_assays(
        traj_assay_dfs, ppm_dict,
        new_color_col='new_target_ang_vel_fov_signed_deg',
        old_color_col='old_target_ang_vel_fov_signed_deg',
        focal_flies_map=None,
        t_rel_points=((-1.0, '1 s before'), (0.0, 'at switch')),
        cmap=None, vlim_percentile=100, scale_percentile=None,
        event_lw=0.5, event_alpha=0.4,
        single_row=False, figsize=None):
    '''
    Three-row Cartesian plot of old/new target positions, colored by a metric, at
    one or more time points relative to the switch.

    When single_row=True, draws only row 3 (old + new positions connected by lines,
    each coloured by its metric) as a standalone single row, one panel per
    (time-point × assay) — typically called with a single t_rel_point at the switch.

    traj_assay_dfs must come from tutil.get_switch_new_target_trajectories (grouped
    by assay_type): each df holds, per event, a window of frames around the switch
    with a t_rel column (seconds from switch; negative = before) plus ego positions
    (x_ego/y_ego = new target, old_x_ego/old_y_ego = old target) and the new_/old_
    metric columns.

    Rows (within each time-point/assay column):
      Row 1 — old target positions colored by old_color_col
      Row 2 — new target positions colored by new_color_col
      Row 3 — both connected by thin gray lines, each colored by their metric

    Columns are grouped by time point (left block first), then assay type within
    each block — e.g. with t_rel_points = [(-1, '1 s before'), (0, 'at switch')],
    the left columns show 1 s before the switch and the right columns at the switch.
    For each event, the frame nearest the requested t_rel is used.

    Valid t_rel values are bounded by the window_sec passed to
    get_switch_new_target_trajectories: that call expands each event to
    switch_frame ± round(window_sec*fps/2), so t_rel only spans ±window_sec/2
    (the generate_switch_plots driver uses window_sec=4.0 → ±2 s). A requested
    t_rel outside that window does not error — the nearest available frame (the
    window edge) is used instead — but a warning is printed.

    Default colormap: magenta (regressive/negative) → white → green (progressive/positive).
    Color and spatial scales are shared across all time points and assays so panels
    are directly comparable. A single shared vertical colorbar is placed to the right.

    Arguments:
        traj_assay_dfs -- dict {assay_type: df from tutil.get_switch_new_target_trajectories}
        ppm_dict       -- dict {assay_type: pixels-per-mm}

    Keyword Arguments:
        new_color_col   -- metric for new target (default: 'new_target_ang_vel_fov_signed')
        old_color_col   -- metric for old target (default: 'old_target_ang_vel_fov_signed')
        focal_flies_map -- dict {triad_type: [focal_fly_ids]} (default: None)
        t_rel_points    -- iterable of (t_rel_seconds, label); one column block per
                           entry. Each t_rel must lie within ±window_sec/2 of the
                           get_switch_new_target_trajectories call
                           (default: [(-1, '1 s before'), (0, 'at switch')])
        cmap            -- colormap; None uses magenta→white→green (default: None)
        vlim_percentile -- percentile of |values| the colormap spans;
                           100 = full range, no clipping (default: 100)
        event_lw        -- line width for connecting lines in row 3 (default: 0.5)
        event_alpha     -- alpha for lines (default: 0.4)
        figsize         -- figure size (default: auto)

    Returns:
        fig, axes  (shape (3, n_times * n_assays))
    '''
    from matplotlib.colors import LinearSegmentedColormap
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            'prog_regr', ['#CC00CC', 'white', '#00BB44']
        )
    assay_types = sorted(traj_assay_dfs.keys())
    n_assays = len(assay_types)
    t_rel_points = list(t_rel_points)
    n_times = len(t_rel_points)
    n_cols = n_assays * n_times
    if n_cols == 0:
        return None, np.empty((3, 0))
    if figsize is None:
        figsize = (5 * n_cols, 5.5) if single_row else (5 * n_cols, 14)

    pos_cols = ['x_ego', 'y_ego', 'old_x_ego', 'old_y_ego']

    def _rows_at_t(tdf, t_target):
        '''One row per event nearest the requested t_rel (exact for t_rel==0).'''
        g = tdf.assign(_td=(tdf['t_rel'] - t_target).abs())
        idx = g.groupby(['acquisition', 'id', 'switch_frame'])['_td'].idxmin()
        return g.loc[idx]

    # ── Pre-pass: per (time, assay) build mm positions; collect global scales ──
    panel_data = {}      # (t_idx, assay) -> sf or None
    global_r_vals = []
    all_color_vals = []
    for assay_type in assay_types:
        tdf = traj_assay_dfs[assay_type].copy()
        ppm = ppm_dict.get(assay_type, 1)
        if focal_flies_map is not None and 'triad_type' in tdf.columns:
            focal_flies = focal_flies_map.get(tdf['triad_type'].iloc[0])
            if focal_flies is not None:
                tdf = tdf[tdf['id'].isin(focal_flies)]
        if len(tdf) == 0 or not all(c in tdf.columns for c in pos_cols):
            for t_idx in range(n_times):
                panel_data[(t_idx, assay_type)] = None
            continue
        t_lo, t_hi = tdf['t_rel'].min(), tdf['t_rel'].max()
        for t_idx, (t_target, _label) in enumerate(t_rel_points):
            if t_target < t_lo or t_target > t_hi:
                print(f"  plot_switch_positions_colored_by_metric_across_assays: "
                      f"requested t_rel={t_target}s is outside the trajectory window "
                      f"[{t_lo:.2f}, {t_hi:.2f}]s for {assay_type} — using nearest "
                      f"frame (window edge).")
            sf = _rows_at_t(tdf, t_target).dropna(subset=pos_cols).copy()
            if len(sf) == 0:
                panel_data[(t_idx, assay_type)] = None
                continue
            sf['_nx'] = sf['x_ego']     / ppm
            sf['_ny'] = sf['y_ego']     / ppm
            sf['_ox'] = sf['old_x_ego'] / ppm
            sf['_oy'] = sf['old_y_ego'] / ppm
            panel_data[(t_idx, assay_type)] = sf
            global_r_vals.extend([
                np.hypot(sf['_nx'].values, sf['_ny'].values),
                np.hypot(sf['_ox'].values, sf['_oy'].values),
            ])
            for col in [new_color_col, old_color_col]:
                if col in sf.columns:
                    all_color_vals.append(sf[col].dropna())

    all_r = np.concatenate(global_r_vals) if global_r_vals else np.array([1.0])
    global_max_r     = _scale_radius(all_r, scale_percentile)
    global_ring_step = _nice_ring_step(global_max_r)
    global_plot_r    = np.max(all_r) * 1.05
    theta_ring       = np.linspace(0, 2 * np.pi, 300)

    vlim = 1.0
    if all_color_vals:
        pooled = pd.concat(all_color_vals).dropna()
        if len(pooled) > 0:
            vlim = np.percentile(pooled.abs(), vlim_percentile)

    def _style(ax, col_idx, row_label, title):
        for r_ring in np.arange(global_ring_step,
                                global_max_r + global_ring_step, global_ring_step):
            ax.plot(r_ring * np.cos(theta_ring), r_ring * np.sin(theta_ring),
                    color='gray', lw=0.5, alpha=0.3, ls='--', zorder=0)
            ax.text(0, r_ring, f'{r_ring:.0f}', fontsize=7, color='gray',
                    ha='center', va='bottom', alpha=0.6)
        ax.scatter(0, 0, s=60, color='white', zorder=7, marker='o',
                   linewidths=0.8, edgecolors='gray')
        ax.set_xlim(-global_plot_r, global_plot_r)
        ax.set_ylim(-global_plot_r, global_plot_r)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.25)
        ax.axvline(0, color='gray', lw=0.4, alpha=0.25)
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)')
        if col_idx == 0:
            ax.set_ylabel(f'{row_label}\ny (mm)')
        if title is not None:
            ax.set_title(title)

    # Column order: time-major, assay-minor (left block = first t_rel_point)
    col_specs = [(t_idx, t_label, assay_type)
                 for t_idx, (_t, t_label) in enumerate(t_rel_points)
                 for assay_type in assay_types]

    # ── single-row mode: standalone row-3 (old + new + line, colored) ─────────
    if single_row:
        fig = plt.figure(figsize=figsize)
        axes_row = []
        last_sc = None
        for col_idx, (t_idx, t_label, assay_type) in enumerate(col_specs):
            ax = fig.add_subplot(1, n_cols, col_idx + 1)
            axes_row.append(ax)
            sf = panel_data.get((t_idx, assay_type))
            if sf is None or len(sf) == 0:
                ax.set_visible(False)
                continue
            n, n_acq = len(sf), _n_acq(sf)
            c_old = sf[old_color_col].values if old_color_col in sf.columns else None
            c_new = sf[new_color_col].values if new_color_col in sf.columns else None
            segs = [[(ox, oy), (nx, ny)] for ox, oy, nx, ny in
                    zip(sf['_ox'], sf['_oy'], sf['_nx'], sf['_ny'])]
            ax.add_collection(LineCollection(segs, colors='gray',
                              linewidths=event_lw, alpha=event_alpha, zorder=1))
            if c_old is not None:
                ax.scatter(sf['_ox'], sf['_oy'], c=c_old, cmap=cmap, vmin=-vlim,
                           vmax=vlim, s=18, alpha=0.75, linewidths=0, marker='s', zorder=3)
            if c_new is not None:
                last_sc = ax.scatter(sf['_nx'], sf['_ny'], c=c_new, cmap=cmap, vmin=-vlim,
                                     vmax=vlim, s=18, alpha=0.75, linewidths=0, marker='o', zorder=4)
            title = f'{assay_type}  ({_count_label(n, n_acq)})'
            if n_times > 1:
                title = f'{t_label}\n{title}'
            _style(ax, col_idx, 'old → new', title)
        fig.suptitle(f'Target old → new at switch, colored by {new_color_col}', fontsize=12)
        plt.tight_layout()
        if last_sc is not None:
            fig.subplots_adjust(right=0.90)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
            cb = fig.colorbar(last_sc, cax=cbar_ax, label=_metric_label(new_color_col))
            cb.ax.text(0.5, 1.03, 'progressive', ha='center', va='bottom',
                       transform=cb.ax.transAxes, fontsize=7, color='#00BB44')
            cb.ax.text(0.5, -0.03, 'regressive', ha='center', va='top',
                       transform=cb.ax.transAxes, fontsize=7, color='#CC00CC')
        return fig, np.array(axes_row).reshape(1, n_cols)

    # ── Subplots ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    row1_axes, row2_axes, row3_axes = [], [], []
    for col_idx in range(n_cols):
        row1_axes.append(fig.add_subplot(3, n_cols, col_idx + 1))
        row2_axes.append(fig.add_subplot(3, n_cols, n_cols + col_idx + 1))
        row3_axes.append(fig.add_subplot(3, n_cols, 2 * n_cols + col_idx + 1))

    last_sc = None
    for col_idx, (t_idx, t_label, assay_type) in enumerate(col_specs):
        ax1, ax2, ax3 = row1_axes[col_idx], row2_axes[col_idx], row3_axes[col_idx]
        sf = panel_data.get((t_idx, assay_type))
        if sf is None or len(sf) == 0:
            for ax in [ax1, ax2, ax3]:
                ax.set_visible(False)
            continue
        n = len(sf)
        n_acq = _n_acq(sf)

        c_old = sf[old_color_col].values if old_color_col in sf.columns else None
        c_new = sf[new_color_col].values if new_color_col in sf.columns else None

        # Row 1: old positions colored by old_color_col
        ax1.scatter(sf['_ox'], sf['_oy'],
                    c=c_old if c_old is not None else 'white',
                    cmap=cmap if c_old is not None else None,
                    vmin=-vlim, vmax=vlim, s=18, alpha=0.75, linewidths=0,
                    marker='s', zorder=2)
        _style(ax1, col_idx, 'old target',
               f'{t_label}\n{assay_type}  ({_count_label(n, n_acq)})')

        # Row 2: new positions colored by new_color_col
        sc2 = ax2.scatter(sf['_nx'], sf['_ny'],
                          c=c_new if c_new is not None else 'white',
                          cmap=cmap if c_new is not None else None,
                          vmin=-vlim, vmax=vlim, s=18, alpha=0.75, linewidths=0,
                          marker='o', zorder=2)
        _style(ax2, col_idx, 'new target', None)
        if c_new is not None:
            last_sc = sc2

        # Row 3: old + new + connecting lines, colored by respective metric
        segs = [[(ox, oy), (nx, ny)]
                for ox, oy, nx, ny in zip(sf['_ox'], sf['_oy'], sf['_nx'], sf['_ny'])]
        lc = LineCollection(segs, colors='gray', linewidths=event_lw,
                            alpha=event_alpha, zorder=1)
        ax3.add_collection(lc)
        if c_old is not None:
            ax3.scatter(sf['_ox'], sf['_oy'], c=c_old, cmap=cmap,
                        vmin=-vlim, vmax=vlim, s=18, alpha=0.75, linewidths=0,
                        marker='s', zorder=3)
        if c_new is not None:
            sc3 = ax3.scatter(sf['_nx'], sf['_ny'], c=c_new, cmap=cmap,
                              vmin=-vlim, vmax=vlim, s=18, alpha=0.75, linewidths=0,
                              marker='o', zorder=4)
            last_sc = sc3
        _style(ax3, col_idx, 'old → new', None)

    fig.suptitle(f'Target positions around switch colored by {new_color_col}', fontsize=12)
    plt.tight_layout()
    if last_sc is not None:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
        cb = fig.colorbar(last_sc, cax=cbar_ax, label=_metric_label(new_color_col))
        cb.ax.text(0.5, 1.03, 'progressive', ha='center', va='bottom',
                   transform=cb.ax.transAxes, fontsize=7, color='#00BB44')
        cb.ax.text(0.5, -0.03, 'regressive', ha='center', va='top',
                   transform=cb.ax.transAxes, fontsize=7, color='#CC00CC')

    axes = np.array(row1_axes + row2_axes + row3_axes).reshape(3, n_cols)
    return fig, axes


def plot_switch_target_tail_colored_by_metric_across_assays(
        traj_assay_dfs, ppm_dict,
        new_color_col='new_target_ang_vel_fov_signed_deg',
        old_color_col='old_target_ang_vel_fov_signed_deg',
        focal_flies_map=None,
        tail_sec=0.25, cmap=None, vlim_percentile=100, scale_percentile=None,
        tail_lw=3.0, endpoint_size=16, tail_alpha=0.85,
        figsize=None):
    '''
    Old vs new target position at the switch frame, focal ego frame, each with a
    short lead-in tail colored by the target_ang_vel_fov metric.

    Two rows (row 0 = old target, row 1 = new target) × one column per assay,
    focal at origin facing +x. Every switch event contributes a track in each row:
    the endpoint is the target position at the switch frame (t_rel≈0) — circle for
    new, square for old — and the tail is a thick line over [-tail_sec, 0] s
    colored by the event's {new_,old_}target_ang_vel_fov value. Color and spatial
    scales are shared across all panels; a single diverging colorbar
    (magenta = regressive, green = progressive) is placed on the right.

    Arguments:
        traj_assay_dfs -- dict {assay_type: df from
                          tutil.get_switch_new_target_trajectories}
        ppm_dict       -- dict {assay_type: pixels-per-mm}

    Keyword Arguments:
        new_color_col   -- metric for the new target (default: 'new_target_ang_vel_fov_signed')
        old_color_col   -- metric for the old target (default: 'old_target_ang_vel_fov_signed')
        focal_flies_map -- dict {triad_type: [focal_fly_ids]} (default: None)
        tail_sec        -- tail length in seconds before the switch (default: 0.25)
        cmap            -- colormap; None → green→white→magenta (default: None)
        vlim_percentile -- percentile of |metric| the color scale spans;
                           100 = full range, no clipping (default: 100)
        scale_percentile-- percentile of radius for the view; None = max (default: None)
        tail_lw         -- tail line width (default: 3.0)
        endpoint_size   -- endpoint marker size (default: 16)
        tail_alpha      -- tail line alpha (default: 0.85)
        figsize         -- figure size tuple (default: auto)

    Returns:
        fig, axes  (shape (2, n_assays): row 0 = old target, row 1 = new target)
    '''
    from matplotlib.colors import Normalize
    if cmap is None:
        cmap = _prog_regr_cmap()

    assay_types = sorted(traj_assay_dfs.keys())
    n_cols = len(assay_types)
    if n_cols == 0:
        return None, np.empty((2, 0))
    if figsize is None:
        figsize = (7.0 * n_cols, 7.0 * 2)

    # ── Pre-pass: collect per-assay tracks per target, global scale + vlim ──────
    assay_data = {}
    global_r, metric_vals = [], []
    for assay_type in assay_types:
        tdf = traj_assay_dfs[assay_type].copy()
        ppm = ppm_dict.get(assay_type, 1)
        if focal_flies_map is not None and 'triad_type' in tdf.columns and len(tdf):
            ff = focal_flies_map.get(tdf['triad_type'].iloc[0])
            if ff is not None:
                tdf = tdf[tdf['id'].isin(ff)]
        if len(tdf) == 0:
            assay_data[assay_type] = None
            continue

        new_tracks, old_tracks = [], []
        for _, ev in tdf.groupby(['acquisition', 'id', 'switch_frame']):
            ev = ev.sort_values('frame')
            t = ev['t_rel'].values
            mask = (t >= -tail_sec) & (t <= 0)
            if mask.sum() < 1:
                continue
            for xc, yc, mcol, bucket in [
                    ('x_ego', 'y_ego', new_color_col, new_tracks),
                    ('old_x_ego', 'old_y_ego', old_color_col, old_tracks)]:
                if xc not in ev.columns:
                    continue
                x = ev[xc].values[mask] / ppm
                y = ev[yc].values[mask] / ppm
                fin = np.isfinite(x) & np.isfinite(y)
                if fin.sum() < 1:
                    continue
                x, y = x[fin], y[fin]
                c = (ev[mcol].values[mask][fin] if mcol in ev.columns
                     else np.full(x.shape, np.nan))
                bucket.append((x, y, c))
                global_r.append(np.hypot(x, y))
                if mcol in ev.columns:
                    metric_vals.append(c)
        if not new_tracks and not old_tracks:
            assay_data[assay_type] = None
            continue
        n_events = tdf.drop_duplicates(['acquisition', 'id', 'switch_frame']).shape[0]
        assay_data[assay_type] = (new_tracks, old_tracks, n_events, _n_acq(tdf))

    all_r = np.concatenate(global_r) if global_r else np.array([5.0])
    base_r = _scale_radius(all_r, scale_percentile) or 5.0
    global_plot_r = base_r * 1.05

    pooled = np.concatenate(metric_vals) if metric_vals else np.array([1.0])
    pooled = pooled[np.isfinite(pooled)]
    vlim = (np.percentile(np.abs(pooled), vlim_percentile)
            if pooled.size else 1.0)
    vlim = vlim if vlim > 0 else 1.0
    mnorm = Normalize(vmin=-vlim, vmax=vlim)

    # ── Draw: row 0 = old target, row 1 = new target ───────────────────────────
    fig, axes = plt.subplots(2, n_cols, figsize=figsize, squeeze=False)
    last_sc = None

    def _draw_tracks(ax, tracks, marker):
        nonlocal last_sc
        for x, y, c in tracks:
            if len(x) >= 2:
                pts = np.column_stack([x, y]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=mnorm,
                                    linewidths=tail_lw, alpha=tail_alpha, zorder=2)
                lc.set_array(c[:-1])
                ax.add_collection(lc)
            last_sc = ax.scatter(x[-1], y[-1], c=[c[-1]], cmap=cmap, norm=mnorm,
                                 s=endpoint_size, marker=marker, edgecolors='white',
                                 linewidths=0.5, zorder=4)
        ax.scatter(0, 0, s=60, color='white', edgecolors='gray', linewidths=0.8,
                   marker='o', zorder=6)
        _style_ego_ax(ax, global_plot_r)

    for col, assay_type in enumerate(assay_types):
        ax_old, ax_new = axes[0, col], axes[1, col]
        if assay_data[assay_type] is None:
            ax_old.set_visible(False)
            ax_new.set_visible(False)
            continue
        new_tracks, old_tracks, n_events, n_acq = assay_data[assay_type]
        _draw_tracks(ax_old, old_tracks, 's')
        _draw_tracks(ax_new, new_tracks, 'o')
        ax_old.set_title(f'{assay_type}  ({_count_label(n_events, n_acq)})')
        if col == 0:
            ax_old.set_ylabel('old target\ny (mm)')
            ax_new.set_ylabel('new target\ny (mm)')

    fig.suptitle(f'Target position at switch with {tail_sec:g} s tail, '
                 f'colored by {new_color_col}', fontsize=12)
    plt.tight_layout()
    if last_sc is not None:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
        cb = fig.colorbar(last_sc, cax=cbar_ax, label=_metric_label(new_color_col))
        cb.ax.text(0.5, 1.03, 'progressive', ha='center', va='bottom',
                   transform=cb.ax.transAxes, fontsize=7, color='#00BB44')
        cb.ax.text(0.5, -0.03, 'regressive', ha='center', va='top',
                   transform=cb.ax.transAxes, fontsize=7, color='#CC00CC')

    return fig, axes


def _time_lightness_cmap(base_color, light_blend=0.82, dark_blend=0.42):
    """Colormap that ramps a base color light → base → dark (for time encoding).

    light_blend pushes the light end toward white; dark_blend pushes the dark
    end toward black. With the default Normalize over t_rel this gives a
    light marker 2 s before the switch and a dark one 2 s after.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    base = np.array(to_rgb(base_color))
    light_end = base + (1.0 - base) * light_blend
    dark_end  = base * dark_blend
    return LinearSegmentedColormap.from_list('time_ramp',
                                             [light_end, base, dark_end])


def _marker_subsample(t, marker_interval_sec, fps):
    """Indices spaced ~marker_interval_sec apart, always including the switch
    frame (t_rel ≈ 0). Returns (sub, sw_idx)."""
    step = max(1, int(round(marker_interval_sec * fps)))
    sub = np.arange(0, len(t), step)
    sw_idx = int(np.argmin(np.abs(t)))
    if sw_idx not in sub:
        sub = np.sort(np.append(sub, sw_idx))
    return sub, sw_idx


def _draw_track(ax, x_mm, y_mm, ori, t, base_color, norm, sub, sw_idx,
                circle_size=45, arrow_mm=1.2, show_arrows=True, label=None):
    """Draw one fly's trajectory in mm: thick base-color path + time-gradient
    circles (light → dark) + optional orientation arrows + a ring on the switch
    frame. Returns (x_finite, y_finite) in mm for extent calc, or None."""
    finite = np.isfinite(x_mm) & np.isfinite(y_mm)
    if not finite.any():
        return None
    cmap = _time_lightness_cmap(base_color)
    colors = cmap(norm(t[sub]))

    # full-resolution path in the base color (every valid frame)
    ax.plot(x_mm[finite], y_mm[finite], color=base_color, alpha=0.55, lw=2.2, zorder=1)

    # orientation arrows at the subsampled frames (drawn under the circles)
    if show_arrows and ori is not None and np.isfinite(ori[sub]).any():
        u = arrow_mm * np.cos(ori[sub])
        v = arrow_mm * np.sin(ori[sub])
        ax.quiver(x_mm[sub], y_mm[sub], u, v, color=colors,
                  angles='xy', scale_units='xy', scale=1, width=0.012,
                  headwidth=3, headlength=4, zorder=3, alpha=0.95)

    # time-gradient circles
    ax.scatter(x_mm[sub], y_mm[sub], c=colors, s=circle_size, marker='o',
               edgecolors='white', linewidths=0.4, zorder=4, label=label)

    # ring the switch-frame marker
    ax.scatter(x_mm[sw_idx], y_mm[sw_idx], s=circle_size * 3.4, facecolors='none',
               edgecolors=cmap(norm(t[sw_idx])), linewidths=3.0, zorder=5)
    return x_mm[finite], y_mm[finite]


def plot_single_switch_target_trajectory(ax, event_df, ppm, frame='ego',
                                          new_color='tomato', old_color=putil.OLD_TARGET_COLOR,
                                          focal_color=putil.FOCAL_COLOR,
                                          fps=60, marker_interval_sec=0.1,
                                          arrow_mm=1.2, circle_size=45,
                                          show_old=True, plot_r=None, span=None,
                                          scale_percentile=None,
                                          title=None, legend=False):
    '''
    Draw one switch event as time-gradient tracks, in the focal fly's egocentric
    frame (frame='ego') or in lab/allocentric coordinates (frame='allo').

    Targets are circles sampled every marker_interval_sec across the ±window,
    colored light → dark with time, each with a body-orientation arrow; a thick
    same-color path connects them and the switch frame (t_rel≈0) is ringed.

    frame='ego'  -- new target (new_color) and old target (old_color) drawn in the
        focal frame; the focal fly sits at the origin with a +x heading arrow. The
        view is a radial ±plot_r mm with range rings.
    frame='allo' -- the focal fly (focal_color), new target and old target are each
        drawn as full tracks in world mm, so target motion in the focal's POV can
        be attributed to focal turning vs target movement. The view is centered on
        the drawn flies with half-width span mm.

    Arguments:
        ax       -- matplotlib Axes to draw on
        event_df -- rows for a single event (one acquisition, id, switch_frame)
                    from tutil.get_switch_new_target_trajectories (allo needs the
                    *_allo columns)
        ppm      -- pixels-per-mm for this acquisition

    Keyword Arguments:
        frame               -- 'ego' or 'allo' (default: 'ego')
        new_color           -- base color for the new target ramp (default: 'tomato')
        old_color           -- base color for the old target ramp (default: 'gray')
        focal_color         -- base color for the focal fly ramp, allo only
                               (default: focal dark purple)
        fps                 -- frames per second (default: 60)
        marker_interval_sec -- spacing between drawn circles/arrows (default: 0.1)
        arrow_mm            -- orientation/heading arrow length in mm (default: 1.2)
        circle_size         -- scatter size for the circles (default: 45)
        show_old            -- also draw the old target if present (default: True)
        plot_r              -- ego: fixed radius (mm) (default: None → auto)
        span                -- allo: fixed view half-width (mm) (default: None → auto)
        scale_percentile    -- ego: percentile of radius for auto plot_r;
                               None = max / no clipping (default: None)
        title               -- axes title (default: None)
        legend              -- draw a small legend (default: False)

    Returns:
        scale_used -- plot_r (ego) or span (allo) actually used (for sharing)
    '''
    from matplotlib.colors import Normalize
    if frame not in ('ego', 'allo'):
        raise ValueError("frame must be 'ego' or 'allo'")

    ev = event_df.sort_values('frame')
    t = ev['t_rel'].values
    if len(t) == 0:
        return plot_r if frame == 'ego' else span
    norm = Normalize(vmin=t.min(), vmax=t.max())
    sub, sw_idx = _marker_subsample(t, marker_interval_sec, fps)

    if frame == 'ego':
        tracks = [(new_color, 'x_ego', 'y_ego', 'new_targ_rel_ori', 'new target')]
        if show_old and 'old_x_ego' in ev.columns and ev['old_x_ego'].notna().any():
            tracks.append((old_color, 'old_x_ego', 'old_y_ego',
                           'old_targ_rel_ori', 'old target'))
    else:
        tracks = [(focal_color, 'focal_x_allo', 'focal_y_allo', 'focal_ori_allo', 'focal'),
                  (new_color, 'new_x_allo', 'new_y_allo', 'new_ori_allo', 'new target')]
        if show_old and 'old_x_allo' in ev.columns and ev['old_x_allo'].notna().any():
            tracks.append((old_color, 'old_x_allo', 'old_y_allo',
                           'old_ori_allo', 'old target'))

    xs, ys = [], []
    for base_color, xc, yc, oc, lbl in tracks:
        if xc not in ev.columns:
            continue
        x = ev[xc].values / ppm
        y = ev[yc].values / ppm
        ori = ev[oc].values if oc in ev.columns else None
        res = _draw_track(ax, x, y, ori, t, base_color, norm, sub, sw_idx,
                          circle_size=circle_size, arrow_mm=arrow_mm, label=lbl)
        if res is not None:
            xs.append(res[0])
            ys.append(res[1])

    if frame == 'ego':
        # focal fly at origin, heading +x
        ax.scatter(0, 0, s=70, color='white', edgecolors='gray', linewidths=0.8,
                   marker='o', zorder=6)
        ax.quiver(0, 0, arrow_mm, 0, color='white', angles='xy',
                  scale_units='xy', scale=1, width=0.012, headwidth=3,
                  headlength=4, zorder=6, alpha=0.8)
        if plot_r is None:
            r = np.hypot(np.concatenate(xs), np.concatenate(ys)) if xs else None
            base = _scale_radius(r, scale_percentile) if r is not None else None
            plot_r = base * 1.1 if base else 5.0
        _style_ego_ax(ax, plot_r)
        scale_used = plot_r
    else:
        scale_used = _style_allo_ax(ax, xs, ys, span)

    if title is not None:
        ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(fontsize=7, loc='upper right')
    return scale_used


def _prog_regr_cmap():
    """Diverging colormap regressive (magenta) → light → progressive (green),
    from the shared Ruta-lab courtship scheme."""
    return putil.prog_regr_cmap()


def _style_ego_ax(ax, plot_r):
    """Egocentric axis styling: range rings + origin crosshair, ±plot_r mm."""
    ax.set_xlim(-plot_r, plot_r)
    ax.set_ylim(-plot_r, plot_r)
    ring_step = _nice_ring_step(plot_r)
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    for r_ring in np.arange(ring_step, plot_r, ring_step):
        ax.plot(r_ring * np.cos(theta_ring), r_ring * np.sin(theta_ring),
                color='gray', lw=0.5, alpha=0.25, ls='--', zorder=0)
    ax.axhline(0, color='gray', lw=0.4, alpha=0.25)
    ax.axvline(0, color='gray', lw=0.4, alpha=0.25)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')


def _style_allo_ax(ax, xs, ys, span):
    """Allocentric axis styling: equal aspect, light grid, centered ±span mm."""
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.grid(True, alpha=0.15)
    if xs:
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        cx = 0.5 * (X.min() + X.max())
        cy = 0.5 * (Y.min() + Y.max())
        data_span = max(X.max() - X.min(), Y.max() - Y.min()) / 2
        if span is None:
            span = data_span * 1.1 + 1.0
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
    return span


def plot_single_switch_target_metric(ax, event_df, ppm, frame='ego',
                                     metric='target_ang_vel_fov_signed_deg',
                                     cmap=None, vlim=1.0,
                                     focal_color=putil.FOCAL_COLOR,
                                     new_color='tomato', old_color=putil.OLD_TARGET_COLOR,
                                     fps=60, marker_interval_sec=0.1,
                                     arrow_mm=1.2, circle_size=45,
                                     show_old=True, plot_r=None, span=None,
                                     scale_percentile=None,
                                     title=None, legend=False):
    '''
    Partner sanity panel: same geometry as the matching trajectory panel, but the
    focal fly is a single solid color (no time gradient) and the new/old targets
    are circles colored by their {new_,old_}{metric} value (with body-orientation
    arrows). Use alongside plot_single_switch_target_trajectory[_allo] to check
    visually that the metric tracks target motion in the focal FOV.

    Arguments:
        ax       -- matplotlib Axes to draw on
        event_df -- rows for a single event from
                    tutil.get_switch_new_target_trajectories
        ppm      -- pixels-per-mm for this acquisition

    Keyword Arguments:
        frame               -- 'ego' or 'allo' (default: 'ego')
        metric              -- base metric name; columns new_{metric}/old_{metric}
                               are read (default: 'target_ang_vel_fov_signed')
        cmap                -- colormap; None → green→white→magenta (default: None)
        vlim                -- symmetric color limit (±vlim) (default: 1.0)
        focal_color         -- solid color for the focal fly; also colors its
                               switch-frame ring (default: focal dark purple)
        new_color           -- color of the new target's switch-frame ring
                               (default: 'tomato')
        old_color           -- color of the old target's switch-frame ring
                               (default: 'gray')
        fps                 -- frames per second (default: 60)
        marker_interval_sec -- circle/arrow spacing in seconds (default: 0.1)
        arrow_mm            -- arrow length in mm (default: 1.2)
        circle_size         -- scatter size (default: 45)
        show_old            -- also draw the old target if present (default: True)
        plot_r              -- ego: fixed radius (mm) (default: None → auto)
        span                -- allo: fixed view half-width (mm) (default: None → auto)
        title               -- axes title (default: None)
        legend              -- draw a small legend (default: False)

    Returns:
        (scale_used, mappable) -- scale_used is plot_r (ego) or span (allo);
        mappable is the metric scatter for a shared colorbar (or None)
    '''
    from matplotlib.colors import Normalize

    if cmap is None:
        cmap = _prog_regr_cmap()
    mnorm = Normalize(vmin=-vlim, vmax=vlim)

    ev = event_df.sort_values('frame')
    t = ev['t_rel'].values
    if len(t) == 0:
        return (plot_r if frame == 'ego' else span), None
    sub, sw_idx = _marker_subsample(t, marker_interval_sec, fps)

    if frame == 'ego':
        focal_cols = None
        targs = [('new', 'x_ego', 'y_ego', 'new_targ_rel_ori', 'o', 'new target'),
                 ('old', 'old_x_ego', 'old_y_ego', 'old_targ_rel_ori', 'o', 'old target')]
    else:
        focal_cols = ('focal_x_allo', 'focal_y_allo', 'focal_ori_allo')
        targs = [('new', 'new_x_allo', 'new_y_allo', 'new_ori_allo', 'o', 'new target'),
                 ('old', 'old_x_allo', 'old_y_allo', 'old_ori_allo', 'o', 'old target')]

    xs, ys = [], []
    last_sc = None

    # focal fly — solid color, no gradient
    if frame == 'ego':
        ax.scatter(0, 0, s=70, color=focal_color, edgecolors='white',
                   linewidths=0.6, zorder=6, label='focal')
        ax.quiver(0, 0, arrow_mm, 0, color=focal_color, angles='xy',
                  scale_units='xy', scale=1, width=0.012, headwidth=3,
                  headlength=4, zorder=6, alpha=0.9)
    elif all(c in ev.columns for c in focal_cols):
        fx = ev[focal_cols[0]].values / ppm
        fy = ev[focal_cols[1]].values / ppm
        fo = ev[focal_cols[2]].values
        fin = np.isfinite(fx) & np.isfinite(fy)
        if fin.any():
            xs.append(fx[fin])
            ys.append(fy[fin])
            ax.plot(fx[fin], fy[fin], color=focal_color, alpha=0.7, lw=2.2, zorder=1)
            if np.isfinite(fo[sub]).any():
                ax.quiver(fx[sub], fy[sub], arrow_mm * np.cos(fo[sub]),
                          arrow_mm * np.sin(fo[sub]), color=focal_color,
                          angles='xy', scale_units='xy', scale=1, width=0.012,
                          headwidth=3, headlength=4, zorder=3, alpha=0.9)
            ax.scatter(fx[sub], fy[sub], color=focal_color, s=circle_size,
                       marker='o', edgecolors='white', linewidths=0.4, zorder=4,
                       label='focal')
            ax.scatter(fx[sw_idx], fy[sw_idx], s=circle_size * 3.4, facecolors='none',
                       edgecolors=focal_color, linewidths=3.0, zorder=5)

    # targets — circles colored by the metric
    for tag, xc, yc, oc, marker, lbl in targs:
        if tag == 'old' and not show_old:
            continue
        mcol = f'{tag}_{metric}'
        if xc not in ev.columns or mcol not in ev.columns:
            continue
        x = ev[xc].values / ppm
        y = ev[yc].values / ppm
        cv = ev[mcol].values
        ori = ev[oc].values if oc in ev.columns else None
        fin = np.isfinite(x) & np.isfinite(y)
        if not fin.any():
            continue
        xs.append(x[fin])
        ys.append(y[fin])
        ax.plot(x[fin], y[fin], color='gray', alpha=0.3, lw=1.5, zorder=1)
        cvals = cv[sub]
        if ori is not None and np.isfinite(ori[sub]).any():
            ax.quiver(x[sub], y[sub], arrow_mm * np.cos(ori[sub]),
                      arrow_mm * np.sin(ori[sub]), cvals, cmap=cmap, norm=mnorm,
                      angles='xy', scale_units='xy', scale=1, width=0.012,
                      headwidth=3, headlength=4, zorder=3, alpha=0.95)
        sc = ax.scatter(x[sub], y[sub], c=cvals, cmap=cmap, norm=mnorm,
                        s=circle_size, marker=marker, edgecolors='white',
                        linewidths=0.4, zorder=4, label=lbl)
        last_sc = sc
        ring_color = new_color if tag == 'new' else old_color
        ax.scatter(x[sw_idx], y[sw_idx], s=circle_size * 3.4, facecolors='none',
                   edgecolors=ring_color, linewidths=3.0, zorder=5)

    if frame == 'ego':
        if plot_r is None:
            r = np.hypot(np.concatenate(xs), np.concatenate(ys)) if xs else None
            base = _scale_radius(r, scale_percentile) if r is not None else None
            plot_r = base * 1.1 if base else 5.0
        _style_ego_ax(ax, plot_r)
        scale_used = plot_r
    else:
        scale_used = _style_allo_ax(ax, xs, ys, span)

    if title is not None:
        ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(fontsize=7, loc='upper right')
    return scale_used, last_sc


def _sample_switch_events(traj_df, n_samples, focal_flies=None, random_state=0):
    """Sample up to n_samples switch events, spread across acquisitions.

    One event per acquisition is taken first. If n_samples exceeds the number of
    acquisitions, additional (distinct) events are drawn by revisiting
    acquisitions that have more than one switch event, round-robin, until
    n_samples is reached or all events are exhausted.

    Returns a list of (acquisition, id, switch_frame) event keys.
    """
    df = traj_df
    if focal_flies is not None and 'id' in df.columns:
        df = df[df['id'].isin(focal_flies)]
    if len(df) == 0:
        return []
    rng = np.random.default_rng(random_state)

    events = (df[['acquisition', 'id', 'switch_frame']]
              .drop_duplicates()
              .sort_values(['acquisition', 'id', 'switch_frame']))

    # event keys grouped by acquisition, shuffled within and across acquisitions
    by_acq = {}
    for acq, grp in events.groupby('acquisition'):
        keys = [(r['acquisition'], int(r['id']), int(r['switch_frame']))
                for _, r in grp.iterrows()]
        rng.shuffle(keys)
        by_acq[acq] = keys
    acqs = list(by_acq.keys())
    rng.shuffle(acqs)

    # round-robin: one event per acquisition per pass until we have n_samples
    picks = []
    while len(picks) < n_samples:
        progressed = False
        for acq in acqs:
            if by_acq[acq]:
                picks.append(by_acq[acq].pop())
                progressed = True
                if len(picks) == n_samples:
                    break
        if not progressed:                 # all acquisitions exhausted
            break
    return picks


def plot_switch_target_trajectory_samples_for_assay(
        traj_df, ppm, assay_type,
        new_color='tomato', old_color=putil.OLD_TARGET_COLOR,
        focal_color=putil.FOCAL_COLOR, frame='ego',
        n_samples=5, fps=60, focal_flies=None,
        marker_interval_sec=0.1, random_state=0,
        share_scale=True, scale_percentile=None,
        panels_per_row=4, half_window_sec=None,
        figsize=None):
    '''
    Sample switch events for a single assay and plot each in its own panel.

    Samples up to n_samples switch events spread across acquisitions (one per
    acquisition first; if n_samples exceeds the number of acquisitions, some
    acquisitions are revisited for additional distinct events) and lays them out
    as a grid of panels (panels_per_row columns) via
    plot_single_switch_target_trajectory(frame=...): frame='ego' draws each panel
    in the focal fly's egocentric frame, frame='allo' in the lab frame with the
    focal fly drawn too. Axis scale is shared across the panels in this figure
    (within-assay only).

    Arguments:
        traj_df    -- trajectory df for ONE assay, from
                      tutil.get_switch_new_target_trajectories
        ppm        -- pixels-per-mm for this assay
        assay_type -- assay label (used in the figure title)

    Keyword Arguments:
        new_color           -- base color for new-target ramp (default: 'tomato')
        old_color           -- base color for old-target ramp (default: medium grey)
        focal_color         -- base color for the focal fly ramp, allo only
                               (default: focal dark purple)
        frame               -- 'ego' (egocentric) or 'allo' (lab frame) (default: 'ego')
        n_samples           -- switch events to sample (default: 5)
        fps                 -- frames per second (default: 60)
        focal_flies         -- list of focal fly ids to keep (default: None)
        marker_interval_sec -- circle/arrow spacing in seconds (default: 0.1)
        random_state        -- seed for reproducible sampling (default: 0)
        share_scale         -- share one scale across panels in this figure
                               (radius for ego, view half-width for allo) (default: True)
        panels_per_row      -- panels per grid row (default: 4)
        half_window_sec     -- window half-width (s) for the title; if None it is
                               read from the data's t_rel range (default: None)
        figsize             -- figure size tuple (default: auto)

    Returns:
        fig, axes  (or None, None if no events)
    '''
    if frame not in ('ego', 'allo'):
        raise ValueError("frame must be 'ego' or 'allo'")

    picks = _sample_switch_events(traj_df, n_samples, focal_flies=focal_flies,
                                  random_state=random_state)
    if not picks:
        print(f"  no switch events to sample for {assay_type}.")
        return None, None

    df = traj_df
    if focal_flies is not None and 'id' in df.columns:
        df = df[df['id'].isin(focal_flies)]
    event_dfs = []
    for acq, fid, sf in picks:
        ev = df[(df['acquisition'] == acq) & (df['id'] == fid)
                & (df['switch_frame'] == sf)]
        if len(ev) > 0:
            event_dfs.append(((acq, fid, sf), ev))

    if not event_dfs:
        return None, None

    # column sets used for the shared-scale pre-pass
    if frame == 'ego':
        scale_cols = [('x_ego', 'y_ego'), ('old_x_ego', 'old_y_ego')]
    else:
        scale_cols = [('focal_x_allo', 'focal_y_allo'),
                      ('new_x_allo', 'new_y_allo'),
                      ('old_x_allo', 'old_y_allo')]

    # shared spatial scale across panels in this figure for comparability:
    # ego -> radius about the origin; allo -> per-event view half-width (span)
    shared_scale = None
    if share_scale:
        ego_r, allo_spans = [], []
        for _, ev in event_dfs:
            xs, ys = [], []
            for xc, yc in scale_cols:
                if xc in ev.columns:
                    x = ev[xc].values / ppm
                    y = ev[yc].values / ppm
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.any():
                        xs.append(x[m])
                        ys.append(y[m])
            if not xs:
                continue
            X, Y = np.concatenate(xs), np.concatenate(ys)
            if frame == 'ego':
                ego_r.append(np.hypot(X, Y))
            else:
                allo_spans.append(max(X.max() - X.min(), Y.max() - Y.min()) / 2)
        if frame == 'ego' and ego_r:
            shared_scale = _scale_radius(np.concatenate(ego_r), scale_percentile) * 1.1
        elif frame == 'allo' and allo_spans:
            shared_scale = max(allo_spans) * 1.1 + 1.0

    n = len(event_dfs)
    ncols = min(panels_per_row, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.2 * ncols, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for i, ((acq, fid, sf), ev) in enumerate(event_dfs):
        title = f'{acq}\nfly {fid} · frame {sf}'
        plot_single_switch_target_trajectory(
            axes_flat[i], ev, ppm, frame=frame,
            new_color=new_color, old_color=old_color, focal_color=focal_color,
            fps=fps, marker_interval_sec=marker_interval_sec,
            plot_r=shared_scale if frame == 'ego' else None,
            span=shared_scale if frame == 'allo' else None,
            scale_percentile=scale_percentile, legend=(i == 0), title=title)
    for ax in axes_flat[n:]:                # hide unused grid cells
        ax.set_visible(False)

    if half_window_sec is None:
        half_window_sec = float(round(traj_df['t_rel'].abs().max(), 2))
    n_acq = len({acq for (acq, _, _), _ in event_dfs})
    frame_label = 'egocentric' if frame == 'ego' else 'allocentric / lab frame'
    fig.suptitle(f'{assay_type} — target trajectory ±{half_window_sec:g} s around '
                 f'switch, {frame_label} ({_count_label(n, n_acq)}; '
                 f'light → dark = time)', fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_switch_target_trajectory_pairs_for_assay(
        traj_df, ppm, assay_type, frame='ego',
        new_color='tomato', old_color=putil.OLD_TARGET_COLOR, focal_color=putil.FOCAL_COLOR,
        metric='target_ang_vel_fov_signed_deg', metric_focal_color=putil.FOCAL_COLOR,
        cmap=None, vlim_percentile=100, scale_percentile=None,
        n_samples=5, fps=60, focal_flies=None,
        marker_interval_sec=0.1, random_state=0,
        share_scale=True, pairs_per_row=2, half_window_sec=None, figsize=None):
    '''
    Sanity-check layout: for each sampled switch event, draw the time-gradient
    trajectory panel and, immediately to its right, a partner panel with the same
    geometry but the targets colored by {metric}. The focal fly is drawn in a
    single solid color in the partner panel (no time gradient). A shared colorbar
    (magenta = regressive/−, green = progressive/+) is placed on the right.

    Same sampling, scaling and frame handling as
    plot_switch_target_trajectory_samples_for_assay; pairs_per_row events sit per
    grid row (so 2*pairs_per_row subplot columns).

    Arguments:
        traj_df    -- trajectory df for ONE assay, from
                      tutil.get_switch_new_target_trajectories
        ppm        -- pixels-per-mm for this assay
        assay_type -- assay label (figure title)

    Keyword Arguments:
        frame              -- 'ego' or 'allo' (default: 'ego')
        new_color/old_color/focal_color -- trajectory-panel ramp colors
        metric             -- base metric name (default: 'target_ang_vel_fov_signed')
        metric_focal_color -- solid focal color in the partner panel; matches the
                              trajectory focal by default (default: focal dark purple)
        cmap               -- metric colormap; None → green→white→magenta (default: None)
        vlim_percentile    -- percentile (0–100) of |metric| the color scale spans;
                              100 = full range, no clipping (default: 100)
        n_samples          -- switch events to sample (default: 5)
        fps                -- frames per second (default: 60)
        focal_flies        -- focal fly ids to keep (default: None)
        marker_interval_sec-- circle/arrow spacing in seconds (default: 0.1)
        random_state       -- seed for reproducible sampling (default: 0)
        share_scale        -- share spatial scale across panels (default: True)
        pairs_per_row      -- event pairs per grid row (default: 2)
        half_window_sec    -- window half-width (s) for the title (default: None → from data)
        figsize            -- figure size (default: auto)

    Returns:
        fig, axes  (or None, None if no events)
    '''
    if frame not in ('ego', 'allo'):
        raise ValueError("frame must be 'ego' or 'allo'")
    if cmap is None:
        cmap = _prog_regr_cmap()

    picks = _sample_switch_events(traj_df, n_samples, focal_flies=focal_flies,
                                  random_state=random_state)
    if not picks:
        print(f"  no switch events to sample for {assay_type}.")
        return None, None

    df = traj_df
    if focal_flies is not None and 'id' in df.columns:
        df = df[df['id'].isin(focal_flies)]
    event_dfs = []
    for acq, fid, sf in picks:
        ev = df[(df['acquisition'] == acq) & (df['id'] == fid)
                & (df['switch_frame'] == sf)]
        if len(ev) > 0:
            event_dfs.append(((acq, fid, sf), ev))
    if not event_dfs:
        return None, None

    if frame == 'ego':
        scale_cols = [('x_ego', 'y_ego'), ('old_x_ego', 'old_y_ego')]
    else:
        scale_cols = [('focal_x_allo', 'focal_y_allo'),
                      ('new_x_allo', 'new_y_allo'),
                      ('old_x_allo', 'old_y_allo')]

    # shared spatial scale (ego radius / allo span) across all events
    shared_scale = None
    if share_scale:
        ego_r, allo_spans = [], []
        for _, ev in event_dfs:
            xs, ys = [], []
            for xc, yc in scale_cols:
                if xc in ev.columns:
                    x = ev[xc].values / ppm
                    y = ev[yc].values / ppm
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.any():
                        xs.append(x[m])
                        ys.append(y[m])
            if not xs:
                continue
            X, Y = np.concatenate(xs), np.concatenate(ys)
            if frame == 'ego':
                ego_r.append(np.hypot(X, Y))
            else:
                allo_spans.append(max(X.max() - X.min(), Y.max() - Y.min()) / 2)
        if frame == 'ego' and ego_r:
            shared_scale = _scale_radius(np.concatenate(ego_r), scale_percentile) * 1.1
        elif frame == 'allo' and allo_spans:
            shared_scale = max(allo_spans) * 1.1 + 1.0

    # shared symmetric color limit for the metric, pooled across events
    # (vlim_percentile is a 0–100 percentile; 100 → full range / no clipping)
    metric_vals = []
    for _, ev in event_dfs:
        for tag in ('new', 'old'):
            mcol = f'{tag}_{metric}'
            if mcol in ev.columns:
                metric_vals.append(ev[mcol].dropna().values)
    pooled = np.concatenate(metric_vals) if metric_vals else np.array([1.0])
    vlim = (np.percentile(np.abs(pooled), vlim_percentile)
            if pooled.size else 1.0)
    vlim = vlim if vlim > 0 else 1.0

    n = len(event_dfs)
    pairs_in_row = min(pairs_per_row, n)
    ncols = pairs_in_row * 2
    nrows = int(np.ceil(n / pairs_in_row))
    if figsize is None:
        figsize = (4.2 * ncols, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    last_sc = None
    for i, ((acq, fid, sf), ev) in enumerate(event_dfs):
        row = i // pairs_in_row
        col = (i % pairs_in_row) * 2
        ax_traj, ax_metric = axes[row, col], axes[row, col + 1]
        title = f'{acq}\nfly {fid} · frame {sf}'

        plot_single_switch_target_trajectory(
            ax_traj, ev, ppm, frame=frame,
            new_color=new_color, old_color=old_color, focal_color=focal_color,
            fps=fps, marker_interval_sec=marker_interval_sec,
            plot_r=shared_scale if frame == 'ego' else None,
            span=shared_scale if frame == 'allo' else None,
            scale_percentile=scale_percentile, legend=(i == 0), title=title)
        _, sc = plot_single_switch_target_metric(
            ax_metric, ev, ppm, frame=frame, metric=metric, cmap=cmap,
            vlim=vlim, focal_color=metric_focal_color,
            new_color=new_color, old_color=old_color, fps=fps,
            marker_interval_sec=marker_interval_sec,
            plot_r=shared_scale if frame == 'ego' else None,
            span=shared_scale if frame == 'allo' else None,
            scale_percentile=scale_percentile, legend=(i == 0),
            title=f'{metric}')
        if sc is not None:
            last_sc = sc

    # hide any unused cells in the final row
    for j in range(n, nrows * pairs_in_row):
        row = j // pairs_in_row
        col = (j % pairs_in_row) * 2
        axes[row, col].set_visible(False)
        axes[row, col + 1].set_visible(False)

    if half_window_sec is None:
        half_window_sec = float(round(traj_df['t_rel'].abs().max(), 2))
    n_acq = len({acq for (acq, _, _), _ in event_dfs})
    frame_label = 'egocentric' if frame == 'ego' else 'allocentric / lab frame'
    fig.suptitle(f'{assay_type} — trajectory (light→dark = time) vs {metric} '
                 f'±{half_window_sec:g} s around switch, {frame_label} '
                 f'({_count_label(n, n_acq)})', fontsize=12)
    plt.tight_layout()

    if last_sc is not None:
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.70])
        cb = fig.colorbar(last_sc, cax=cbar_ax, label=_metric_label(metric))
        cb.ax.text(0.5, 1.03, 'progressive', ha='center', va='bottom',
                   transform=cb.ax.transAxes, fontsize=7, color='#00BB44')
        cb.ax.text(0.5, -0.03, 'regressive', ha='center', va='top',
                   transform=cb.ax.transAxes, fontsize=7, color='#CC00CC')

    return fig, axes
