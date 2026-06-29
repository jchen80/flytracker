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

from ._helpers import _add_focal_fly_marker, _build_condition_panels, _compute_position_histogram, _compute_position_kde, _density_cmap, _exclude_action_frames, _filter_by_focal_fly, _filter_to_target_pairs, _focal_body_size_mm, _select_action_frames


def _density_fn(method):
    """Pick the per-panel density estimator: 'hist' (binned) or 'kde' (smoothed)."""
    if method not in ('hist', 'kde'):
        raise ValueError(f"method must be 'hist' or 'kde', got {method!r}")
    return _compute_position_kde if method == 'kde' else _compute_position_histogram


def plot_relative_position_density(df, action_cols=None, focal_fly_ids=None,
                                    save_dir=None, acq=None,
                                    figsize=None, n_grid=100, ppm=None,
                                    vmax_percentile=99, method='hist'):
    plot_df = _filter_by_focal_fly(df, focal_fly_ids)
    if len(plot_df) == 0:
        print("No rows found after focal fly filtering.")
        return None, None

    panels = _build_condition_panels(plot_df, action_cols)
    n_panels = len(panels)
    if figsize is None:
        figsize = (6 * n_panels, 6)

    all_x = plot_df['targ_rel_pos_x'].dropna()
    all_y = plot_df['targ_rel_pos_y'].dropna()
    if ppm is not None:
        x_lim = (all_x.min() / ppm, all_x.max() / ppm)
        y_lim = (all_y.min() / ppm, all_y.max() / ppm)
        xlabel, ylabel = 'x (egocentric, mm)', 'y (egocentric, mm)'
    else:
        x_lim = (all_x.min(), all_x.max())
        y_lim = (all_y.min(), all_y.max())
        xlabel, ylabel = 'x (egocentric, px)', 'y (egocentric, px)'

    # compute raw histograms
    histograms, edge_list, n_points = [], [], []
    for label, panel_df in panels:
        h, x_edges, y_edges, n = _density_fn(method)(
            panel_df, x_lim, y_lim, n_grid, ppm)
        histograms.append(h)
        edge_list.append((x_edges, y_edges))
        n_points.append(n)
        print(f"  '{label}': {n} points")

    # global color scale with percentile clip
    all_nonzero = np.concatenate([h[h > 0] for h in histograms])
    vmin = 0
    vmax = np.percentile(all_nonzero, vmax_percentile)
    unit_str = '% / mm²' if ppm is not None else 'prob. density'
    density_label = f'{unit_str} (clipped at {vmax_percentile}th pct, >{vmax:.3f})'

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, (label, _), h, (x_edges, y_edges), n in zip(
            axes, panels, histograms, edge_list, n_points):
        im = ax.imshow(np.ma.masked_where(h.T <= 0, h.T), origin='lower', aspect='equal', cmap=_density_cmap(),
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=density_label)
        _add_focal_fly_marker(ax, ppm)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f'{label}\n(n={n} frames)')

    title = f'{acq} — ' if acq else ''
    focal_str = f'focal fly: {focal_fly_ids}' if focal_fly_ids is not None else 'all pairs'
    mtag = ' (KDE)' if method == 'kde' else ''
    fig.suptitle(f'{title}relative position density{mtag}\n{focal_str}', fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        msuf = '_kde' if method == 'kde' else ''
        savepath = os.path.join(save_dir, f'relative_position_density_{action_str}{msuf}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, axes


def plot_relative_position_density_across_assays(assay_dfs, ppm_dict,
                                                  action_cols=None,
                                                  focal_flies_map=None,
                                                  n_grid=100,
                                                  vmax_percentile=99,
                                                  include_non_action=False,
                                                  target_action_col=None,
                                                  zoom_mm=None,
                                                  draw_focal_ellipse=False,
                                                  save_dir=None, figsize=None, method='hist'):
    '''
    Plot 2D relative position density (rows=conditions, cols=assay types) across multiple assays.

    Arguments:
        assay_dfs -- dict mapping assay type to its dataframe
        ppm_dict  -- dict mapping assay type to pixels-per-mm value

    Keyword Arguments:
        action_cols       -- list of action column names to split by (default: None)
        focal_flies_map   -- dict mapping triad_type to focal fly id list (default: None)
        n_grid            -- number of grid bins per axis (default: 100)
        vmax_percentile   -- percentile for clipping colormap across all panels (default: 99)
        include_non_action -- if True, add a 'non-<action> (all pairs)' panel for each action,
                              covering all fly pairs in non-action frames (default: False)
        target_action_col -- if set, the regular action panel is labeled '<action> (all pairs)'
                             and an additional '<action> (target only)' panel is added showing
                             only the focal fly → assigned target pair. When not set, the action
                             panel keeps its plain label. (default: None)
        zoom_mm           -- if set, adds a zoomed row for each condition showing only
                             ±zoom_mm around the focal fly. The zoom range is identical
                             across all assay types. Histograms are recomputed at full
                             n_grid resolution within the zoom window. (default: None)
        draw_focal_ellipse -- if True, draws the focal fly's body as an ellipse at the
                             origin (major axis along +x) using the median major_axis_len
                             and minor_axis_len from the data, converted to mm via PPM.
                             Falls back to a dot+arrow if those columns are absent.
                             (default: False)
        save_dir          -- directory to save figure (default: None)
        figsize           -- figure size tuple (default: auto)

    Returns:
        fig, axes
    '''
    assay_types = sorted(assay_dfs.keys())

    condition_labels = ['all frames']
    if action_cols is not None:
        for a in action_cols:
            if any(a in df.columns for df in assay_dfs.values()):
                condition_labels.append(f'{a} (all pairs)' if target_action_col == a else a)
                if target_action_col == a:
                    condition_labels.append(f'{a} (target only)')
                if include_non_action:
                    condition_labels.append(f'non-{a} (all pairs)')

    n_cond = len(condition_labels)
    n_rows = n_cond * 2 if zoom_mm is not None else n_cond
    n_cols = len(assay_types)
    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    # global axis limits (full range, symmetric square)
    all_x, all_y = [], []
    for assay_type, assay_df in assay_dfs.items():
        ppm = ppm_dict.get(assay_type, 1)
        triad_type = assay_df['triad_type'].iloc[0]
        focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
        plot_df = _filter_by_focal_fly(assay_df, focal_flies)
        dedup = plot_df.drop_duplicates(['frame', 'pair', 'acquisition'])
        all_x.append(dedup['targ_rel_pos_x'].dropna() / ppm)
        all_y.append(dedup['targ_rel_pos_y'].dropna() / ppm)
    xy_range = max(abs(pd.concat(all_x).min()), abs(pd.concat(all_x).max()),
                   abs(pd.concat(all_y).min()), abs(pd.concat(all_y).max()))
    x_lim = y_lim = (-xy_range, xy_range)
    zoom_lim = (-zoom_mm, zoom_mm) if zoom_mm is not None else None

    def _filtered_plot_df(assay_type, cond_label):
        assay_df = assay_dfs[assay_type]
        triad_type = assay_df['triad_type'].iloc[0]
        focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
        plot_df = _filter_by_focal_fly(assay_df, focal_flies)
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

    # pre-compute histograms for full range and (if requested) zoom range
    histograms = {}   # key: (cond_label, assay_type, 'full'|'zoom')
    n_points   = {}
    for cond_label in condition_labels:
        for assay_type in assay_types:
            ppm = ppm_dict.get(assay_type, 1)
            plot_df = _filtered_plot_df(assay_type, cond_label)
            if plot_df is None or len(plot_df) == 0:
                histograms[(cond_label, assay_type, 'full')] = None
                if zoom_lim:
                    histograms[(cond_label, assay_type, 'zoom')] = None
                continue

            h, x_edges, y_edges, n = _density_fn(method)(
                plot_df, x_lim, y_lim, n_grid, ppm,
                dedup_cols=['frame', 'pair', 'acquisition'])
            histograms[(cond_label, assay_type, 'full')] = (h, x_edges, y_edges)
            n_points[(cond_label, assay_type, 'full')] = n

            if zoom_lim is not None:
                hz, xze, yze, nz = _density_fn(method)(
                    plot_df, zoom_lim, zoom_lim, n_grid, ppm,
                    dedup_cols=['frame', 'pair', 'acquisition'])
                histograms[(cond_label, assay_type, 'zoom')] = (hz, xze, yze)
                n_points[(cond_label, assay_type, 'zoom')] = nz

    first_ppm = next(iter(ppm_dict.values()))
    unit_str = '% / mm²' if first_ppm is not None else 'prob. density'

    # Per-assay body size in mm (median over focal flies only, converted via PPM)
    body_size_mm = {}  # assay_type -> (length_mm, width_mm) or (None, None)
    for assay_type, assay_df in assay_dfs.items():
        focal_flies = (focal_flies_map.get(assay_df['triad_type'].iloc[0])
                       if focal_flies_map else None)
        body_size_mm[assay_type] = (
            _focal_body_size_mm(assay_df, ppm_dict.get(assay_type, 1), focal_flies)
            if draw_focal_ellipse else (None, None))

    def _vmax_for(keys):
        valid = [histograms[k] for k in keys if histograms.get(k) is not None]
        if not valid:
            return 1.0
        nonzero = np.concatenate([h[h > 0] for h, _, _ in valid])
        return float(np.percentile(nonzero, vmax_percentile)) if len(nonzero) else 1.0

    full_keys = [(cl, at, 'full') for cl in condition_labels for at in assay_types]
    vmax_full = _vmax_for(full_keys)
    print(f"Global vmax full ({vmax_percentile}th percentile): {vmax_full:.4f} {unit_str}")

    vmax_zoom = None
    if zoom_lim is not None:
        zoom_keys = [(cl, at, 'zoom') for cl in condition_labels for at in assay_types]
        vmax_zoom = _vmax_for(zoom_keys)
        print(f"Global vmax zoom ({vmax_percentile}th percentile): {vmax_zoom:.4f} {unit_str}")

    # build figure — full rows first, then zoom rows (one block per condition)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    def _draw_panel(ax, h, x_edges, y_edges, lim, n, ppm, vmax,
                    body_length_mm=None, body_width_mm=None,
                    title=None, ylabel=None, xlabel=False):
        density_label = f'{unit_str} (>{vmax:.3f} clipped at {vmax_percentile}th pct)'
        im = ax.imshow(np.ma.masked_where(h.T <= 0, h.T), origin='lower', aspect='equal', cmap=_density_cmap(),
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label=density_label)
        _add_focal_fly_marker(ax, ppm,
                              body_length_mm=body_length_mm,
                              body_width_mm=body_width_mm)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.text(0.02, 0.02, f'n={n}', transform=ax.transAxes,
                color='white', fontsize=7, va='bottom')
        if title:
            ax.set_title(title, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9)
        if xlabel:
            ax.set_xlabel('x (mm)', fontsize=9)

    for cond_idx, cond_label in enumerate(condition_labels):
        full_row = cond_idx
        zoom_row = n_cond + cond_idx if zoom_lim is not None else None
        is_last_full = (cond_idx == n_cond - 1) and zoom_lim is None
        is_last_zoom = zoom_row is not None and zoom_row == n_rows - 1

        for col_idx, assay_type in enumerate(assay_types):
            ppm = ppm_dict.get(assay_type, 1)

            # ── full row ──
            ax = axes[full_row, col_idx]
            result = histograms.get((cond_label, assay_type, 'full'))
            if result is None:
                ax.set_visible(False)
            else:
                h, x_edges, y_edges = result
                n = n_points.get((cond_label, assay_type, 'full'), 0)
                bl, bw = body_size_mm[assay_type]
                _draw_panel(ax, h, x_edges, y_edges, x_lim, n, ppm, vmax_full,
                            body_length_mm=bl, body_width_mm=bw,
                            title=assay_type if full_row == 0 else None,
                            ylabel=f'{cond_label}\ny (mm)' if col_idx == 0 else None,
                            xlabel=is_last_full and zoom_lim is None)

            # ── zoom row ──
            if zoom_row is not None:
                axz = axes[zoom_row, col_idx]
                zresult = histograms.get((cond_label, assay_type, 'zoom'))
                if zresult is None:
                    axz.set_visible(False)
                else:
                    hz, xze, yze = zresult
                    nz = n_points.get((cond_label, assay_type, 'zoom'), 0)
                    zoom_ylabel = (f'{cond_label} (±{zoom_mm}mm)\ny (mm)'
                                   if col_idx == 0 else None)
                    bl, bw = body_size_mm[assay_type]
                    _draw_panel(axz, hz, xze, yze, zoom_lim, nz, ppm, vmax_zoom,
                                body_length_mm=bl, body_width_mm=bw,
                                title=assay_type if zoom_row == n_cond else None,
                                ylabel=zoom_ylabel,
                                xlabel=is_last_zoom)

    mtag = ' (KDE)' if method == 'kde' else ''
    fig.suptitle(f'Relative position density across assay types{mtag}', fontsize=13)
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        msuf = '_kde' if method == 'kde' else ''
        savepath = os.path.join(save_dir,
                                f'relative_position_density_across_assays_{action_str}{msuf}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, axes


def plot_relative_position_density_by_velocity_bin_across_assays(
        binned_assay_dfs, ppm_dict,
        focal_flies_map=None,
        n_grid=100, vmax_percentile=99,
        zoom_mm=10, draw_focal_ellipse=True,
        save_dir=None, figsize=None, method='hist'):
    '''
    2D relative position density of the target fly in the focal fly's reference
    frame, with one row per velocity bin and one column per assay type. Every panel
    is the zoomed ±zoom_mm view, on a shared color scale so bins/assays compare.

    Arguments:
        binned_assay_dfs -- ordered list of (bin_label, {assay_type: df}), where each
                            df is already filtered to that velocity bin (e.g. from
                            tutil.filter_pursuit_frames). Matches the convention used
                            by plot_metric_by_velocity_bin_across_assays.
        ppm_dict         -- dict mapping assay type to pixels-per-mm value

    Keyword Arguments:
        focal_flies_map  -- dict mapping triad_type to focal fly id list (default: None)
        n_grid           -- grid bins per axis within the zoom window (default: 100)
        vmax_percentile  -- percentile for clipping the shared colormap (default: 99)
        zoom_mm          -- half-width of the view in mm (default: 10)
        draw_focal_ellipse -- draw focal body ellipse at origin from median
                              major/minor axis length (default: True)
        save_dir         -- directory to save figure (default: None)
        figsize          -- figure size tuple (default: auto)

    Returns:
        fig, axes
    '''
    bin_labels = [bl for bl, _ in binned_assay_dfs]
    assay_types = sorted({at for _, adfs in binned_assay_dfs for at in adfs})
    n_rows, n_cols = len(bin_labels), len(assay_types)
    if n_rows == 0 or n_cols == 0:
        return None, None
    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    zoom_lim = (-zoom_mm, zoom_mm)

    def _prep(df, assay_type):
        '''Focal-fly-filtered df for this assay (or None).'''
        if df is None or len(df) == 0:
            return None
        triad_type = df['triad_type'].iloc[0] if 'triad_type' in df.columns else None
        focal_flies = (focal_flies_map.get(triad_type)
                       if focal_flies_map and triad_type else None)
        return _filter_by_focal_fly(df, focal_flies)

    # Histograms per (bin, assay) over the zoom window
    histograms, n_points = {}, {}
    for bin_label, adfs in binned_assay_dfs:
        for assay_type in assay_types:
            ppm = ppm_dict.get(assay_type, 1)
            plot_df = _prep(adfs.get(assay_type), assay_type)
            if plot_df is None or len(plot_df) == 0:
                histograms[(bin_label, assay_type)] = None
                continue
            h, xe, ye, n = _density_fn(method)(
                plot_df, zoom_lim, zoom_lim, n_grid, ppm,
                dedup_cols=['frame', 'pair', 'acquisition'])
            histograms[(bin_label, assay_type)] = (h, xe, ye)
            n_points[(bin_label, assay_type)] = n

    first_ppm = next(iter(ppm_dict.values()))
    unit_str = '% / mm²' if first_ppm is not None else 'prob. density'

    # Per-assay focal body size (median major/minor axis over focal flies → mm). Body
    # size is morphology, not behavior, so pool ALL velocity bins (≈ all frames) rather
    # than reading it off a single bin.
    body_size_mm = {}
    for assay_type in assay_types:
        bl_bw = (None, None)
        if draw_focal_ellipse:
            dfs = [adfs[assay_type] for _, adfs in binned_assay_dfs
                   if adfs.get(assay_type) is not None and len(adfs[assay_type])]
            if dfs:
                allf = pd.concat(dfs, ignore_index=True)
                focal_flies = (focal_flies_map.get(allf['triad_type'].iloc[0])
                               if (focal_flies_map and 'triad_type' in allf.columns) else None)
                bl_bw = _focal_body_size_mm(allf, ppm_dict.get(assay_type, 1), focal_flies)
        body_size_mm[assay_type] = bl_bw

    # Shared color scale across every panel
    valid = [v for v in histograms.values() if v is not None]
    if valid:
        nonzero = np.concatenate([h[h > 0] for h, _, _ in valid])
        vmax = float(np.percentile(nonzero, vmax_percentile)) if len(nonzero) else 1.0
    else:
        vmax = 1.0
    print(f"Global vmax ({vmax_percentile}th percentile): {vmax:.4f} {unit_str}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    density_label = f'{unit_str} (>{vmax:.3f} clipped at {vmax_percentile}th pct)'
    for r, bin_label in enumerate(bin_labels):
        for c, assay_type in enumerate(assay_types):
            ax = axes[r, c]
            result = histograms.get((bin_label, assay_type))
            if result is None:
                ax.set_visible(False)
                continue
            h, xe, ye = result
            ppm = ppm_dict.get(assay_type, 1)
            n = n_points.get((bin_label, assay_type), 0)
            im = ax.imshow(np.ma.masked_where(h.T <= 0, h.T), origin='lower', aspect='equal', cmap=_density_cmap(),
                           extent=[xe[0], xe[-1], ye[0], ye[-1]], vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, label=density_label)
            bl, bw = body_size_mm[assay_type]
            _add_focal_fly_marker(ax, ppm, body_length_mm=bl, body_width_mm=bw)
            ax.set_xlim(zoom_lim); ax.set_ylim(zoom_lim)
            ax.set_aspect('equal')
            ax.text(0.02, 0.02, f'n={n}', transform=ax.transAxes,
                    color='white', fontsize=7, va='bottom')
            if r == 0:
                ax.set_title(assay_type, fontsize=11)
            if c == 0:
                ax.set_ylabel(f'{bin_label}\ny (mm)', fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('x (mm)', fontsize=9)

    mtag = ' (KDE)' if method == 'kde' else ''
    fig.suptitle(f'Relative target-position density by velocity bin (±{zoom_mm} mm){mtag}',
                 fontsize=13)
    plt.tight_layout()

    if save_dir is not None:
        msuf = '_kde' if method == 'kde' else ''
        savepath = os.path.join(
            save_dir, f'relative_position_density_by_velocity_bin_pm{zoom_mm:g}mm{msuf}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, axes
