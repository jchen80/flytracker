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
from scipy.stats import gaussian_kde, binned_statistic_2d
import pandas as pd
import seaborn as sns


from analyses.triad.src import util as tutil


def _filter_by_focal_fly(df, focal_fly_ids):
    '''Filter df to rows where the pair for which metrics have been calculated includes the focal fly as in focal_fly_ids.
    Assumes df contains a 'pair' column with format 'flyid1_flyid2' and that metrics are calculated for flyid1 as focal fly.
    '''
    if focal_fly_ids is None:
        return df
    mask = df['pair'].apply(lambda p: int(p.split('_')[0]) in focal_fly_ids)
    return df[mask]


def _select_action_frames(df, action_col):
    '''Return all rows whose (acquisition, frame) pair contains at least one fly performing action_col.
    Uses acquisition+frame jointly so frame numbers from different videos don't collide.
    '''
    active = df.loc[df[action_col] != -1, ['acquisition', 'frame']].drop_duplicates()
    active_idx = pd.MultiIndex.from_arrays([active['acquisition'], active['frame']])
    df_idx = pd.MultiIndex.from_arrays([df['acquisition'], df['frame']])
    return df[df_idx.isin(active_idx)]


def _exclude_action_frames(df, action_col):
    '''Return all rows whose (acquisition, frame) pair contains no fly performing action_col.
    Complement of _select_action_frames — uses acquisition+frame jointly to avoid cross-video collisions.
    '''
    active = df.loc[df[action_col] != -1, ['acquisition', 'frame']].drop_duplicates()
    active_idx = pd.MultiIndex.from_arrays([active['acquisition'], active['frame']])
    df_idx = pd.MultiIndex.from_arrays([df['acquisition'], df['frame']])
    return df[~df_idx.isin(active_idx)]


def _filter_to_target_pairs(df, action_col):
    '''Keep only rows where the pair matches the focal fly (id) and its assigned target for action_col.
    Works regardless of which fly appears first in the pair string.
    Only keeps rows where a target is assigned (target != -1).
    '''
    target_col = f'{action_col}_target'
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found, skipping target pair filtering.")
        return df

    pair_fly0 = df['pair'].str.split('_').str[0].astype(int)
    pair_fly1 = df['pair'].str.split('_').str[1].astype(int)
    fly_id = df['id'].astype(int)
    target_id = pd.to_numeric(df[target_col], errors='coerce').fillna(-1).astype(int)

    has_target = target_id != -1
    pair_matches = (
        ((pair_fly0 == fly_id) & (pair_fly1 == target_id)) |
        ((pair_fly1 == fly_id) & (pair_fly0 == target_id))
    )
    return df[has_target & pair_matches]


def _build_condition_panels(df, action_cols):
    '''
    Build list of (label, panel_df) tuples for all frames + actions + non-actions.

    Returns:
        panels -- list of (label, df) tuples
    '''
    panels = [('all frames', df)]
    if action_cols is not None:
        for action in action_cols:
            if action not in df.columns:
                print(f"Action '{action}' not found, skipping.")
                continue
            action_df = _select_action_frames(df, action)
            if len(action_df) == 0:
                print(f"No annotated frames for '{action}', skipping.")
                continue
            panels.append((action, action_df))
            non_action_df = _exclude_action_frames(df, action)
            if len(non_action_df) > 0:
                panels.append((f'non-{action} (all pairs)', non_action_df))
    return panels


def _compute_position_histogram(panel_df, x_lim, y_lim, n_grid, ppm,
                                 dedup_cols=None, log_scale=False):
    '''
    Compute 2D histogram of targ_rel_pos_x/y for a panel df.
    Note: log_scale removed — scaling is handled by the plotting functions.
    '''
    if dedup_cols is None:
        dedup_cols = ['frame', 'pair']
    dedup = panel_df.drop_duplicates(dedup_cols)
    x = dedup['targ_rel_pos_x'].dropna().values
    y = dedup['targ_rel_pos_y'].dropna().values
    if ppm is not None:
        x = x / ppm
        y = y / ppm
    h, x_edges, y_edges = np.histogram2d(x, y, bins=n_grid,
                                          range=[x_lim, y_lim],
                                          density=True)
    if ppm is not None:
        h = h * 100
    return h, x_edges, y_edges, len(x)


def _compute_position_kde(panel_df, x_lim, y_lim, n_grid, ppm,
                          dedup_cols=None, log_scale=False, mask_frac=0.02,
                          max_points=20000):
    '''
    Gaussian-KDE smoothed counterpart to _compute_position_histogram: same inputs
    and return shape (h, x_edges, y_edges, n), but the density is a smooth KDE of
    targ_rel_pos_x/y evaluated on the n_grid×n_grid cell-center grid (same extent /
    %·mm⁻² scaling as the histogram). Cells below mask_frac of the peak are set to 0
    so the plotter masks them to the panel background, matching the histogram's
    empty-bin look. Returns zeros if there are too few / degenerate points.

    KDE evaluation is O(n_grid² · n_points), so the points are randomly subsampled
    to max_points (the density estimate is unchanged for a representative sample);
    without this a full per-frame panel (~10⁵-10⁶ points) effectively hangs. The
    reported n is the true (pre-subsample) point count.
    '''
    if dedup_cols is None:
        dedup_cols = ['frame', 'pair']
    dedup = panel_df.drop_duplicates(dedup_cols)
    xy = dedup[['targ_rel_pos_x', 'targ_rel_pos_y']].dropna().to_numpy()
    x_edges = np.linspace(x_lim[0], x_lim[1], n_grid + 1)
    y_edges = np.linspace(y_lim[0], y_lim[1], n_grid + 1)
    n_full = len(xy)
    if n_full < 3:
        return np.zeros((n_grid, n_grid)), x_edges, y_edges, n_full
    if n_full > max_points:                       # subsample so the KDE eval stays fast
        xy = xy[np.random.default_rng(0).choice(n_full, max_points, replace=False)]

    x, y = xy[:, 0], xy[:, 1]
    if ppm is not None:
        x, y = x / ppm, y / ppm
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    # indexing='ij' -> (nx, ny), matching np.histogram2d's h orientation (plotters do h.T)
    gx, gy = np.meshgrid(xc, yc, indexing='ij')
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        h = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    except np.linalg.LinAlgError:                 # singular covariance (collinear points)
        return np.zeros((n_grid, n_grid)), x_edges, y_edges, n_full
    if ppm is not None:
        h = h * 100                               # match the histogram's %·mm⁻² scaling
    if mask_frac > 0 and h.max() > 0:
        h = np.where(h < mask_frac * h.max(), 0.0, h)   # floor the faint tail -> light bg
    return h, x_edges, y_edges, n_full


def _density_cmap():
    """inferno with masked (zero-density) bins drawn black, so each density panel's
    empty space reads as black (its conventional look) regardless of the figure theme.
    Pair with np.ma.masked_where(h <= 0, h) so zero-count bins use this 'bad' color."""
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('black')
    return cmap


def _compute_2d_density(panel_df, col_x, col_y, x_lim, y_lim, n_grid,
                        method='hist', dedup_cols=None, ppm=None,
                        mask_frac=0.02, max_points=20000):
    '''
    2D density of two arbitrary columns -- a column-parameterized generalization of
    _compute_position_histogram / _compute_position_kde (which are hardcoded to
    targ_rel_pos_x/y).

    method 'hist' -> np.histogram2d(density=True); 'kde' -> gaussian_kde evaluated on
    the cell-center grid (subsampled to max_points, faint tail < mask_frac*peak floored
    to 0 so empty space masks to the panel background, matching the position KDE).
    When ppm is given, x/y are divided by ppm (px->mm) and the density rescaled ×100
    (%·mm⁻²) as in the position helpers; pass ppm=None for unit-agnostic axes (e.g.
    degrees). dedup_cols, if given, drops duplicate rows on those columns first (the
    fov pivot is already one row per focal-frame, so callers pass None). Returns
    (h, x_edges, y_edges, n) with h shape (n_grid, n_grid), oriented like
    np.histogram2d (callers plot h.T).
    '''
    dedup = (panel_df.drop_duplicates([c for c in dedup_cols if c in panel_df.columns])
             if dedup_cols else panel_df)
    xy = dedup[[col_x, col_y]].dropna().to_numpy()
    x_edges = np.linspace(x_lim[0], x_lim[1], n_grid + 1)
    y_edges = np.linspace(y_lim[0], y_lim[1], n_grid + 1)
    n_full = len(xy)
    if n_full < 3:
        return np.zeros((n_grid, n_grid)), x_edges, y_edges, n_full
    x, y = xy[:, 0].astype(float), xy[:, 1].astype(float)
    if ppm is not None:
        x, y = x / ppm, y / ppm

    if method == 'kde':
        if n_full > max_points:                   # subsample so the KDE eval stays fast
            sel = np.random.default_rng(0).choice(n_full, max_points, replace=False)
            x, y = x[sel], y[sel]
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        gx, gy = np.meshgrid(xc, yc, indexing='ij')   # 'ij' matches histogram2d orientation
        try:
            h = gaussian_kde(np.vstack([x, y]))(
                np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        except np.linalg.LinAlgError:             # singular covariance (collinear points)
            return np.zeros((n_grid, n_grid)), x_edges, y_edges, n_full
        if ppm is not None:
            h = h * 100
        if mask_frac > 0 and h.max() > 0:
            h = np.where(h < mask_frac * h.max(), 0.0, h)
        return h, x_edges, y_edges, n_full

    h, x_edges, y_edges = np.histogram2d(x, y, bins=n_grid, range=[x_lim, y_lim],
                                         density=True)
    if ppm is not None:
        h = h * 100
    return h, x_edges, y_edges, n_full


def _compute_2d_binned_stat(panel_df, col_x, col_y, col_val, x_lim, y_lim, n_grid,
                            stat='mean', min_count=20):
    '''
    2D binned statistic of col_val over (col_x, col_y). With a 0/1 flag and stat='mean'
    this gives p(event) per cell (e.g. p(courtship) over the θ-θ space). Returns
    (stat_grid, count_grid, x_edges, y_edges); grids are (n_grid, n_grid) oriented like
    np.histogram2d (callers plot .T). Cells with count < min_count are set to NaN so the
    plotter masks sparsely-occupied positions. stat 'mean'|'sum'; 'count' ignores col_val.
    '''
    need = [col_x, col_y] + ([col_val] if (col_val and stat != 'count') else [])
    d = panel_df[need].dropna()
    rng = [list(x_lim), list(y_lim)]
    x, y = d[col_x].to_numpy(), d[col_y].to_numpy()
    counts, x_edges, y_edges, _ = binned_statistic_2d(
        x, y, None, statistic='count', bins=n_grid, range=rng)
    if stat == 'count':
        grid = counts.astype(float)
    else:
        grid, _, _, _ = binned_statistic_2d(
            x, y, d[col_val].to_numpy(), statistic=stat, bins=n_grid, range=rng)
        grid = grid.astype(float)
    if min_count and min_count > 0:
        grid[counts < min_count] = np.nan
    return grid, counts, x_edges, y_edges


def _compute_metric_histogram(panel_df, metric, bin_edges,
                               dedup_cols=None):
    '''
    Compute 1D histogram of a metric for a panel df.

    Returns:
        counts -- normalized histogram counts
        n      -- number of points used
    '''
    if dedup_cols is None:
        dedup_cols = ['frame', 'pair']
    dedup = panel_df.drop_duplicates(dedup_cols)
    vals = dedup[metric].dropna()
    counts, _ = np.histogram(vals, bins=bin_edges, density=True)
    return counts, len(vals)


def _focal_body_size_mm(df, ppm, focal_flies=None):
    '''Median focal-fly body footprint (length_mm, width_mm) from major_axis_len /
    minor_axis_len (px) ÷ ppm; (None, None) if the columns/ppm/data are unavailable.

    focal_flies restricts to those ids (via the 'id' column) when given. De-duplicates
    on (acquisition, frame, id) so each fly-frame counts once.
    '''
    if (df is None or len(df) == 0 or not ppm
            or 'major_axis_len' not in df.columns or 'minor_axis_len' not in df.columns):
        return (None, None)
    rows = df[df['id'].isin(focal_flies)] if focal_flies else df
    if len(rows) == 0:
        return (None, None)
    rows = rows.drop_duplicates(subset=['acquisition', 'frame', 'id'])
    return (rows['major_axis_len'].median() / ppm,
            rows['minor_axis_len'].median() / ppm)


def _add_focal_fly_marker(ax, ppm, body_length_mm=None, body_width_mm=None):
    '''Add focal fly position marker to an axis.

    If body_length_mm and body_width_mm are given, draws an ellipse representing
    the fly's body footprint (major axis along +x, since data is ego-centric with
    focal fly facing right). Otherwise falls back to a dot + heading arrow.
    '''
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)
    ax.axvline(0, color='white', lw=0.5, alpha=0.3)

    if body_length_mm is not None and body_width_mm is not None:
        ellipse = Ellipse(xy=(0, 0), width=body_length_mm, height=body_width_mm,
                          angle=0, facecolor='white', edgecolor='white',
                          alpha=0.6, zorder=5)
        ax.add_patch(ellipse)
        ax.scatter(0, 0, s=30, color='white', zorder=6, marker='o')
    else:
        arrow_len = 2 if ppm is not None else 40
        ax.scatter(0, 0, s=150, color='white', zorder=5, marker='o')
        ax.annotate('', xy=(arrow_len, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))
