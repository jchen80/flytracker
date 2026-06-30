"""Egocentric 'two targets on the FOV' plots for the triad module.

For each focal male during courtship there are two possible targets: the *pursued*
target (the assigned courtship_target) and the *other* (the remaining non-focal fly).
These plotters visualize where both sit on the male's egocentric field of view, across
assay scenarios (e.g. MFF / MMF-2M / MMF-1M × species):

  - plot_target_fov_density_across_assays      : 2D egocentric position density of each
        target (rows = pursued/other, cols = assay types); focal at origin facing +x.
  - plot_target_theta_joint_density_across_assays : joint density of signed theta-error
        to the pursued vs the other target (one panel per assay type).

Both build their per-assay tables with util.get_courtship_target_fov and reuse the
2D-density kernel and styling helpers in _helpers.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from analyses.triad.src import util as tutil
import libs.plotting as putil

from ._helpers import (_compute_2d_density, _compute_2d_binned_stat, _density_cmap,
                       _add_focal_fly_marker, _focal_body_size_mm)


def _prob_cmap():
    """viridis with masked (NaN) cells drawn in the panel background, for p-maps where
    low-occupancy cells are masked (so p=0 (dark viridis) stays distinct from masked)."""
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(plt.rcParams.get('axes.facecolor', '#262626'))
    return cmap


def _diverging_cmap():
    """RdBu_r centered at 0 with masked cells in the panel background, for signed maps
    (e.g. mean dθ-error/dt: red = +, blue = −)."""
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(plt.rcParams.get('axes.facecolor', '#262626'))
    return cmap


def _set_theta_ticks(ax, lim=180, step=90):
    """Put ticks every `step` degrees on both axes of a θ–θ panel (e.g. ±180 → -180,
    -90, 0, 90, 180), so x and y share the same spacing."""
    ticks = np.arange(-(int(lim) // step) * step, int(lim) + 1, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def _fov_tables(assay_dfs, focal_flies_map=None, action_col='courtship'):
    """{assay_key: get_courtship_target_fov(df)} for non-empty assays (focal-fly filtered)."""
    out = {}
    for key in sorted(assay_dfs):
        fov = tutil.get_courtship_target_fov(assay_dfs[key], action_col=action_col)
        if fov.empty:
            continue
        if focal_flies_map is not None and 'triad_type' in fov.columns:
            focal = focal_flies_map.get(fov['triad_type'].iloc[0])
            if focal:
                fov = fov[fov['id'].isin(focal)]
        if not fov.empty:
            out[key] = fov
    return out


def _shared_vmax(grids, vmax_percentile):
    """Percentile-clipped vmax over the nonzero density of all panels (shared scale)."""
    nz = [g[0][g[0] > 0].ravel() for g in grids if (g[0] > 0).any()]
    return float(np.percentile(np.concatenate(nz), vmax_percentile)) if nz else 1.0


def plot_target_fov_density_across_assays(assay_dfs, ppm_dict, focal_flies_map=None,
                                          roles=('pursued', 'other'), method='hist',
                                          n_grid=100, vmax_percentile=99, zoom_mm=None,
                                          action_col='courtship', save_dir=None,
                                          figsize=None):
    """2D egocentric density of each target on the focal male's FOV during courtship.

    Rows = roles (pursued, other), cols = assay types. Focal fly at origin facing +x,
    y-up, axes in mm. Density via _compute_2d_density on (<role>_x, <role>_y) with the
    assay's ppm (so axes are physical mm); color scale shared across all panels.
    method 'hist' or 'kde'. Saves target_fov_density{_kde}.png if save_dir given.
    """
    fov = _fov_tables(assay_dfs, focal_flies_map, action_col)
    if not fov:
        print("[warn] no courtship-target FOV data to plot")
        return None, None
    assay_types = sorted(fov.keys())
    ppm_any = next((ppm_dict.get(k) for k in assay_types if ppm_dict.get(k)), None)

    if zoom_mm is not None:
        lim = float(zoom_mm)
    else:
        allxy = np.vstack([f[[f'{r}_x', f'{r}_y']].to_numpy()
                           for f in fov.values() for r in roles])
        r = float(np.nanpercentile(np.abs(allxy), vmax_percentile))
        lim = r / ppm_any if ppm_any else r
    x_lim = y_lim = (-lim, lim)

    nrows, ncols = len(roles), len(assay_types)
    if figsize is None:
        figsize = (3.1 * ncols, 3.1 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    cmap = _density_cmap()

    grids = {}
    for j, key in enumerate(assay_types):
        for i, role in enumerate(roles):
            grids[(i, j)] = _compute_2d_density(
                fov[key], f'{role}_x', f'{role}_y', x_lim, y_lim, n_grid,
                method=method, dedup_cols=None, ppm=ppm_dict.get(key))
    vmax = _shared_vmax(grids.values(), vmax_percentile)

    # focal body footprint per assay: median major/minor axis length (px) -> mm via ppm,
    # restricted to focal flies (same convention as the generate_plots density grids).
    body_size_mm = {}
    for key in assay_types:
        adf = assay_dfs.get(key)
        focal = (focal_flies_map.get(adf['triad_type'].iloc[0])
                 if (focal_flies_map and adf is not None and not adf.empty
                     and 'triad_type' in adf.columns) else None)
        body_size_mm[key] = _focal_body_size_mm(adf, ppm_dict.get(key), focal)

    im = None
    for j, key in enumerate(assay_types):
        for i, role in enumerate(roles):
            ax = axes[i][j]
            h, xe, ye, n = grids[(i, j)]
            im = ax.imshow(h.T, origin='lower',
                           aspect='equal', cmap=cmap,
                           extent=[xe[0], xe[-1], ye[0], ye[-1]], vmin=0, vmax=vmax)
            bl, bw = body_size_mm.get(key, (None, None))
            _add_focal_fly_marker(ax, ppm_dict.get(key),
                                  body_length_mm=bl, body_width_mm=bw)
            ax.set_xlim(x_lim); ax.set_ylim(y_lim)
            ax.set_title(f'{key}\n(n={n})' if i == 0 else f'(n={n})',
                         fontsize=10 if i == 0 else 8)
            if j == 0:
                ax.set_ylabel(f'{role} target\ny (mm)')
            if i == nrows - 1:
                ax.set_xlabel('x (mm)')
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                     label='% / mm²')
    fig.suptitle('Target positions on focal male FOV during courtship'
                 + (' (KDE)' if method == 'kde' else ''), fontsize=13)

    if save_dir is not None:
        msuf = '_kde' if method == 'kde' else ''
        path = os.path.join(save_dir, f'target_fov_density{msuf}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def plot_target_theta_joint_density_across_assays(assay_dfs, focal_flies_map=None,
                                                  method='hist', n_grid=100, lim=180,
                                                  vmax_percentile=99,
                                                  action_col='courtship', save_dir=None,
                                                  figsize=None):
    """Joint density of signed theta-error to the pursued vs the other target during
    courtship. One panel per assay type; x = θ to pursued, y = θ to other (degrees,
    −lim…lim). Reference lines at 0 (both axes) and the y=x diagonal. method 'hist' or
    'kde'. Saves target_theta_joint_density{_kde}.png if save_dir given.
    """
    fov = _fov_tables(assay_dfs, focal_flies_map, action_col)
    if not fov:
        print("[warn] no courtship-target FOV data to plot")
        return None, None
    assay_types = sorted(fov.keys())
    x_lim = y_lim = (-lim, lim)
    ncols = min(3, len(assay_types))                 # wrap into a roughly-square grid
    nrows = int(np.ceil(len(assay_types) / ncols))
    if figsize is None:
        figsize = (3.6 * ncols, 3.9 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    axflat = axes.ravel()
    cmap = _density_cmap()

    grids = {k: _compute_2d_density(fov[k], 'pursued_theta_deg', 'other_theta_deg',
                                    x_lim, y_lim, n_grid, method=method,
                                    dedup_cols=None, ppm=None)
             for k in assay_types}
    vmax = _shared_vmax(grids.values(), vmax_percentile)

    im = None
    for idx, key in enumerate(assay_types):
        ax = axflat[idx]
        h, xe, ye, n = grids[key]
        im = ax.imshow(h.T, origin='lower', aspect='equal',
                       cmap=cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]], vmin=0, vmax=vmax)
        ax.axhline(0, color='white', lw=0.5, alpha=0.4)
        ax.axvline(0, color='white', lw=0.5, alpha=0.4)
        ax.plot([-lim, lim], [-lim, lim], color='white', lw=0.6, alpha=0.35, ls='--')
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        _set_theta_ticks(ax, lim)
        ax.set_title(f'{key}\n(n={n})', fontsize=10)
        if idx // ncols == nrows - 1:
            ax.set_xlabel('θ to pursued (deg)')
        if idx % ncols == 0:
            ax.set_ylabel('θ to other (deg)')
    for ax in axflat[len(assay_types):]:
        ax.axis('off')
    if im is not None:
        fig.colorbar(im, ax=axflat.tolist(), fraction=0.04, pad=0.02, label='density')
    fig.suptitle('Signed θ-error to pursued vs other target during courtship'
                 + (' (KDE)' if method == 'kde' else ''), fontsize=13)

    if save_dir is not None:
        msuf = '_kde' if method == 'kde' else ''
        path = os.path.join(save_dir, f'target_theta_joint_density{msuf}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def plot_target_courtship_prob_maps(assay_dfs, sex_map, focal_flies_map=None,
                                    n_grid=50, lim=180, min_count=20,
                                    action_col='courtship', save_dir=None, figsize=None):
    """p(courtship of each target) over the joint (θ to target x, θ to target y) space,
    per assay. Rows = p(court target x) / p(court target y); cols = assay types. Targets
    are sex-labeled (util.get_target_pair_fov: MMF=female/male, MFF=two females by id).
    Cells with < min_count occupancy frames are masked. Saves
    target_theta_courtship_prob.png if save_dir given.
    """
    fov = {}
    for key in sorted(assay_dfs):
        t = tutil.get_target_pair_fov(assay_dfs[key], sex_map,
                                      focal_flies_map=focal_flies_map, action_col=action_col)
        if not t.empty:
            fov[key] = t
    if not fov:
        print("[warn] no target-pair FOV data for courtship-prob maps")
        return None, None
    assay_types = sorted(fov.keys())
    x_lim = y_lim = (-lim, lim)
    role_cols = ['court_x', 'court_y']

    grids = {}
    for j, key in enumerate(assay_types):
        for i, col in enumerate(role_cols):
            g, _cnt, xe, ye = _compute_2d_binned_stat(
                fov[key], 'theta_x_deg', 'theta_y_deg', col, x_lim, y_lim, n_grid,
                stat='mean', min_count=min_count)
            grids[(i, j)] = (g, xe, ye)
    finite = [g[0][np.isfinite(g[0])].ravel() for g in grids.values() if np.isfinite(g[0]).any()]
    vmax = float(np.nanpercentile(np.concatenate(finite), 99)) if finite else 1.0

    nrows, ncols = 2, len(assay_types)
    if figsize is None:
        figsize = (3.3 * ncols, 6.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    cmap = _prob_cmap()
    im = None
    for j, key in enumerate(assay_types):
        xl, yl = fov[key]['x_label'].iloc[0], fov[key]['y_label'].iloc[0]
        for i in range(nrows):
            ax = axes[i][j]
            g, xe, ye = grids[(i, j)]
            im = ax.imshow(np.ma.masked_invalid(g.T), origin='lower', aspect='equal',
                           cmap=cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                           vmin=0, vmax=vmax)
            ax.axhline(0, color='white', lw=0.4, alpha=0.3)
            ax.axvline(0, color='white', lw=0.4, alpha=0.3)
            ax.plot([-lim, lim], [-lim, lim], color='white', lw=0.5, alpha=0.25, ls='--')
            ax.set_xlim(x_lim); ax.set_ylim(y_lim)
            _set_theta_ticks(ax, lim)
            if i == 0:
                ax.set_title(f'{key}\n(x: {xl}, y: {yl})', fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel('θ to target x (deg)')
    target_lbl = ['target x', 'target y']
    for i in range(nrows):
        axes[i][0].set_ylabel(f'p(court {target_lbl[i]})\nθ to target y (deg)')
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                     label='p(courtship | position)')
    fig.suptitle('p(courtship of each target) over the target θ–θ occupancy space',
                 fontsize=13)

    if save_dir is not None:
        path = os.path.join(save_dir, 'target_theta_courtship_prob.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def plot_target_switch_points(assay_dfs, sex_map, focal_flies_map=None, triad='MFF',
                              n_grid=50, lim=180, action_col='courtship',
                              switch_source='manual', save_dir=None, figsize=None):
    """Where manually-annotated switches land in the joint (θ to target x, θ to target y)
    space, MFF only. Rows = switches→target x / switches→target y; cols = MFF assays.
    Each panel: faint occupancy density background + overlaid switch-event points (events
    whose NEW target is that row's target). Skips with a warning if no manual switches.
    Saves target_theta_switch_points.png if save_dir given.
    """
    keys = sorted(k for k, d in assay_dfs.items() if d['triad_type'].iloc[0] == triad)
    if not keys:
        print(f"[warn] no {triad} assays for switch-points plot")
        return None, None

    data = {}                      # key -> (fov_df, switch_df_with_xy)
    for key in keys:
        fovdf = tutil.get_target_pair_fov(assay_dfs[key], sex_map,
                                          focal_flies_map=focal_flies_map, action_col=action_col)
        if fovdf.empty:
            continue
        sw = tutil.get_switch_frame_vectors(assay_dfs[key], action_col=action_col,
                                            switch_source=switch_source)
        if not sw.empty:
            sw = sw.merge(fovdf[['acquisition', 'frame', 'id', 'theta_x_deg',
                                 'theta_y_deg', 'x_id', 'y_id']],
                          on=['acquisition', 'frame', 'id'], how='inner')
        data[key] = (fovdf, sw)
    if not data or all(sw.empty for _, sw in data.values()):
        print(f"[warn] no manual switches found for {triad}; skipping switch-points plot")
        return None, None

    keys = sorted(data.keys())
    x_lim = y_lim = (-lim, lim)
    nrows, ncols = 2, len(keys)
    if figsize is None:
        figsize = (3.3 * ncols + 0.5, 6.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    bg_cmap = plt.get_cmap('Greys')
    pt_color = '#FFD11A'
    row_lbl = ['target x', 'target y']
    for j, key in enumerate(keys):
        fovdf, sw = data[key]
        xl, yl = fovdf['x_label'].iloc[0], fovdf['y_label'].iloc[0]
        occ, xe, ye, _ = _compute_2d_density(fovdf, 'theta_x_deg', 'theta_y_deg',
                                             x_lim, y_lim, n_grid, method='hist',
                                             dedup_cols=None, ppm=None)
        to_x = (sw['new_target'] == sw['x_id']) if not sw.empty else None
        for i in range(nrows):
            ax = axes[i][j]
            ax.imshow(occ.T, origin='lower', aspect='equal',
                      cmap=bg_cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]], alpha=0.55)
            n_ev = 0
            if not sw.empty:
                sub = sw[to_x] if i == 0 else sw[~to_x]
                n_ev = len(sub)
                ax.scatter(sub['theta_x_deg'], sub['theta_y_deg'], s=28, color=pt_color,
                           edgecolor='black', linewidths=0.4, alpha=0.9, zorder=3)
            ax.axhline(0, color='white', lw=0.4, alpha=0.3)
            ax.axvline(0, color='white', lw=0.4, alpha=0.3)
            ax.plot([-lim, lim], [-lim, lim], color='white', lw=0.5, alpha=0.25, ls='--')
            ax.set_xlim(x_lim); ax.set_ylim(y_lim)
            _set_theta_ticks(ax, lim)
            if i == 0:
                ax.set_title(f'{key}\n(x: {xl}, y: {yl})', fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel('θ to target x (deg)')
            if j == 0:
                ax.set_ylabel(f'switches → {row_lbl[i]}\nθ to target y (deg)')
            ax.text(0.03, 0.97, f'n={n_ev}', transform=ax.transAxes, va='top', ha='left',
                    color='white', fontsize=8)
    fig.suptitle(f'{triad}: manually-annotated switch locations in the target θ–θ space',
                 fontsize=13)

    if save_dir is not None:
        path = os.path.join(save_dir, 'target_theta_switch_points.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def _partner_metric_lookup(df, metric_col):
    """(acquisition, frame, id, _partner) -> metric_col, for joining a per-partner
    quantity (e.g. target_ang_vel_fov_signed_deg) onto switch events by target id."""
    d = df[['acquisition', 'frame', 'id', 'pair', metric_col]].copy()
    ps = d['pair'].str.split('_')
    f0 = ps.str[0].astype(int)
    f1 = ps.str[1].astype(int)
    d['_partner'] = np.where(f0 == d['id'].values, f1, f0)
    return d.drop(columns='pair')


def _switch_fov_data(assay_dfs, sex_map, focal_flies_map=None, triad='MFF',
                     action_col='courtship', switch_source='manual',
                     metric_col='target_ang_vel_fov_signed_deg'):
    """{key: (fovdf, sw)} for `triad` assays. fovdf = all-frames target-pair FOV (the
    θ_x/θ_y occupancy + sex labels); sw = switch events merged with the focal's bearing
    to each target (theta_x_deg/theta_y_deg) and the FOV angular velocity of the NEW vs
    OLD target (fov_new / fov_old / fov_diff = new − old), with a new_is_x flag."""
    keys = sorted(k for k, d in assay_dfs.items() if d['triad_type'].iloc[0] == triad)
    data = {}
    for key in keys:
        df = assay_dfs[key]
        fovdf = tutil.get_target_pair_fov(df, sex_map, focal_flies_map=focal_flies_map,
                                          action_col=action_col)
        if fovdf.empty:
            continue
        sw = tutil.get_switch_frame_vectors(df, action_col=action_col,
                                            switch_source=switch_source)
        if not sw.empty:
            mcols = ['acquisition', 'frame', 'id', 'theta_x_deg', 'theta_y_deg', 'x_id', 'y_id']
            if 'ang_vel_fly_degs' in fovdf.columns:    # male turning at the switch frame
                mcols.append('ang_vel_fly_degs')
            sw = sw.merge(fovdf[mcols], on=['acquisition', 'frame', 'id'], how='inner')
        if not sw.empty:
            sw['new_is_x'] = sw['new_target'] == sw['x_id']
            if metric_col in df.columns:
                lut = _partner_metric_lookup(df, metric_col)
                sw = sw.merge(lut.rename(columns={'_partner': 'new_target',
                                                  metric_col: 'fov_new'}),
                              on=['acquisition', 'frame', 'id', 'new_target'], how='left')
                sw = sw.merge(lut.rename(columns={'_partner': 'old_target',
                                                  metric_col: 'fov_old'}),
                              on=['acquisition', 'frame', 'id', 'old_target'], how='left')
                sw['fov_diff'] = sw['fov_new'] - sw['fov_old']
        data[key] = (fovdf, sw)
    return data


def _courtship_occupancy(fovdf):
    """The (θ_x, θ_y) rows during courtship (focal courting either target), for the grey
    switch-map background; falls back to all rows if no courtship frames are present."""
    court = fovdf[(fovdf.get('court_x', 0) == 1) | (fovdf.get('court_y', 0) == 1)]
    return court if not court.empty else fovdf


def plot_switch_points_combined(assay_dfs, sex_map, focal_flies_map=None, triad='MFF',
                                n_grid=50, lim=180, action_col='courtship',
                                switch_source='manual', save_dir=None, figsize=None):
    """All manually-annotated switches on a SINGLE joint (θ to target x, θ to target y)
    panel per assay, colored by which target was switched TO: pink = new target is x,
    blue = new target is y. Grey background = courtship-frame (θ_x, θ_y) occupancy density.
    One row, cols = MFF assays. Saves target_theta_switch_combined.png.
    """
    data = _switch_fov_data(assay_dfs, sex_map, focal_flies_map=focal_flies_map,
                            triad=triad, action_col=action_col, switch_source=switch_source)
    if not data or all(sw.empty for _, sw in data.values()):
        print(f"[warn] no manual switches found for {triad}; skipping combined switch plot")
        return None, None

    keys = sorted(data.keys())
    x_lim = y_lim = (-lim, lim)
    ncols = len(keys)
    if figsize is None:
        figsize = (3.5 * ncols + 0.5, 3.9)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    axes = axes[0]
    bg_cmap = plt.get_cmap('Greys')
    COL_X, COL_Y = '#FF3DA5', '#3D7DFF'      # pink = new target x, blue = new target y
    for ax, key in zip(axes, keys):
        fovdf, sw = data[key]
        xl, yl = fovdf['x_label'].iloc[0], fovdf['y_label'].iloc[0]
        occ, xe, ye, _ = _compute_2d_density(_courtship_occupancy(fovdf), 'theta_x_deg',
                                             'theta_y_deg', x_lim, y_lim, n_grid,
                                             method='hist', dedup_cols=None, ppm=None)
        ax.imshow(occ.T, origin='lower', aspect='equal',
                  cmap=bg_cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]], alpha=0.55)
        n_x = n_y = 0
        if not sw.empty:
            sx, sy = sw[sw['new_is_x']], sw[~sw['new_is_x']]
            n_x, n_y = len(sx), len(sy)
            ax.scatter(sx['theta_x_deg'], sx['theta_y_deg'], s=30, color=COL_X,
                       edgecolor='black', linewidths=0.4, alpha=0.9, zorder=3,
                       label=f'→ {xl} (n={n_x})')
            ax.scatter(sy['theta_x_deg'], sy['theta_y_deg'], s=30, color=COL_Y,
                       edgecolor='black', linewidths=0.4, alpha=0.9, zorder=3,
                       label=f'→ {yl} (n={n_y})')
        ax.axhline(0, color=COL_Y, lw=1.0, alpha=0.8)   # y=0 (target y ahead) → blue
        ax.axvline(0, color=COL_X, lw=1.0, alpha=0.8)   # x=0 (target x ahead) → pink
        ax.plot([-lim, lim], [-lim, lim], color='0.5', lw=0.5, alpha=0.35, ls='--')
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        _set_theta_ticks(ax, lim)
        ax.set_title(f'{key}\n(x: {xl}, y: {yl})', fontsize=9)
        ax.set_xlabel('θ to target x (deg)')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
    axes[0].set_ylabel('θ to target y (deg)')
    fig.suptitle(f'{triad}: switches colored by new target (grey = courtship occupancy)',
                 fontsize=13)

    if save_dir is not None:
        path = os.path.join(save_dir, 'target_theta_switch_combined.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def plot_switch_points_by_angvel(assay_dfs, sex_map, focal_flies_map=None, triad='MFF',
                                 n_grid=50, lim=180, action_col='courtship',
                                 switch_source='manual', vlim_percentile=95,
                                 save_dir=None, figsize=None):
    """All manually-annotated switches on a SINGLE joint (θ to target x, θ to target y)
    panel per assay, colored by the focal's angular velocity (ang_vel_fly, deg/s, signed)
    at the switch frame. One row, cols = MFF assays; grey background = courtship occupancy.
    ang_vel_fly is target-independent so this is a single map. Saves
    target_theta_switch_angvel.png.
    """
    data = _switch_fov_data(assay_dfs, sex_map, focal_flies_map=focal_flies_map,
                            triad=triad, action_col=action_col, switch_source=switch_source)
    if not data or all(sw.empty for _, sw in data.values()):
        print(f"[warn] no manual switches found for {triad}; skipping angvel switch plot")
        return None, None
    if not any((not sw.empty) and ('ang_vel_fly_degs' in sw.columns) for _, sw in data.values()):
        print(f"[warn] 'ang_vel_fly_degs' not available; skipping angvel switch plot")
        return None, None

    keys = sorted(data.keys())
    x_lim = y_lim = (-lim, lim)
    allv = np.concatenate([sw['ang_vel_fly_degs'].dropna().to_numpy()
                           for _, sw in data.values()
                           if (not sw.empty) and 'ang_vel_fly_degs' in sw.columns]
                          or [np.array([0.0])])
    vmax = float(np.nanpercentile(np.abs(allv), vlim_percentile)) if allv.size else 1.0
    vmax = vmax or 1.0
    ncols = len(keys)
    if figsize is None:
        figsize = (3.5 * ncols + 0.6, 3.9)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    axes = axes[0]
    bg_cmap = plt.get_cmap('Greys')
    cmap = _diverging_cmap()
    sc = None
    for ax, key in zip(axes, keys):
        fovdf, sw = data[key]
        xl, yl = fovdf['x_label'].iloc[0], fovdf['y_label'].iloc[0]
        occ, xe, ye, _ = _compute_2d_density(_courtship_occupancy(fovdf), 'theta_x_deg',
                                             'theta_y_deg', x_lim, y_lim, n_grid,
                                             method='hist', dedup_cols=None, ppm=None)
        ax.imshow(occ.T, origin='lower', aspect='equal',
                  cmap=bg_cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]], alpha=0.55)
        n_ev = 0
        if (not sw.empty) and 'ang_vel_fly_degs' in sw.columns:
            sub = sw.dropna(subset=['ang_vel_fly_degs'])
            n_ev = len(sub)
            if n_ev:
                sc = ax.scatter(sub['theta_x_deg'], sub['theta_y_deg'], s=32,
                                c=sub['ang_vel_fly_degs'], cmap=cmap, vmin=-vmax, vmax=vmax,
                                edgecolor='black', linewidths=0.4, zorder=3)
        ax.axhline(0, color='0.5', lw=0.4, alpha=0.5)
        ax.axvline(0, color='0.5', lw=0.4, alpha=0.5)
        ax.plot([-lim, lim], [-lim, lim], color='0.5', lw=0.5, alpha=0.35, ls='--')
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        _set_theta_ticks(ax, lim)
        ax.set_title(f'{key}\n(x: {xl}, y: {yl})', fontsize=9)
        ax.set_xlabel('θ to target x (deg)')
        ax.text(0.03, 0.97, f'n={n_ev}', transform=ax.transAxes, va='top', ha='left',
                color='black', fontsize=8)
    axes[0].set_ylabel('θ to target y (deg)')
    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02,
                     label='ang_vel_fly at switch (deg/s, signed)')
    fig.suptitle(f'{triad}: switches colored by fly angular velocity (grey = courtship occupancy)',
                 fontsize=13)

    if save_dir is not None:
        path = os.path.join(save_dir, 'target_theta_switch_angvel.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def _draw_switch_value_maps(data, colorings, cmap, triad='MFF', n_grid=50, lim=180,
                            save_dir=None, figsize=None):
    """Core for the new/old/Δ switch-value maps. For each coloring (dict with keys
    col, suf, vmin, vmax, cbar_label, title_desc) draw a 2-row (switches → target x /
    switches → target y) × assays figure of switch points colored by data[key][1][col]
    on `cmap` over the courtship-occupancy grey background. Saves
    target_theta_switch_{suf}.png. Returns [(col, fig)]."""
    keys = sorted(data.keys())
    x_lim = y_lim = (-lim, lim)
    bg_cmap = plt.get_cmap('Greys')
    COL_X, COL_Y = '#FF3DA5', '#3D7DFF'      # pink = target x axis, blue = target y axis
    row_lbl = ['target x', 'target y']

    out = []
    for spec in colorings:
        col, suf = spec['col'], spec['suf']
        vmin, vmax = spec['vmin'], spec['vmax']
        nrows, ncols = 2, len(keys)
        fs = figsize or (3.3 * ncols + 0.8, 6.8)
        fig, axes = plt.subplots(nrows, ncols, figsize=fs, squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        sc = None
        for j, key in enumerate(keys):
            fovdf, sw = data[key]
            xl, yl = fovdf['x_label'].iloc[0], fovdf['y_label'].iloc[0]
            occ, xe, ye, _ = _compute_2d_density(_courtship_occupancy(fovdf), 'theta_x_deg',
                                                 'theta_y_deg', x_lim, y_lim, n_grid,
                                                 method='hist', dedup_cols=None, ppm=None)
            for i in range(nrows):
                ax = axes[i][j]
                ax.imshow(occ.T, origin='lower',
                          aspect='equal', cmap=bg_cmap,
                          extent=[xe[0], xe[-1], ye[0], ye[-1]], alpha=0.5)
                n_ev = 0
                if (not sw.empty) and col in sw.columns:
                    sub = sw[sw['new_is_x']] if i == 0 else sw[~sw['new_is_x']]
                    sub = sub.dropna(subset=[col])
                    n_ev = len(sub)
                    if n_ev:
                        sc = ax.scatter(sub['theta_x_deg'], sub['theta_y_deg'], s=34,
                                        c=sub[col], cmap=cmap, vmin=vmin, vmax=vmax,
                                        edgecolor='black', linewidths=0.4, zorder=3)
                ax.axhline(0, color=COL_Y, lw=1.0, alpha=0.8)   # y=0 (target y ahead) → blue
                ax.axvline(0, color=COL_X, lw=1.0, alpha=0.8)   # x=0 (target x ahead) → pink
                ax.plot([-lim, lim], [-lim, lim], color='0.5', lw=0.5, alpha=0.35, ls='--')
                ax.set_xlim(x_lim); ax.set_ylim(y_lim)
                _set_theta_ticks(ax, lim)
                if i == 0:
                    ax.set_title(f'{key}\n(x: {xl}, y: {yl})', fontsize=9)
                if i == nrows - 1:
                    ax.set_xlabel('θ to target x (deg)')
                if j == 0:
                    ax.set_ylabel(f'switches → {row_lbl[i]}\nθ to target y (deg)')
                ax.text(0.03, 0.97, f'n={n_ev}', transform=ax.transAxes, va='top',
                        ha='left', color='black', fontsize=8)
        if sc is not None:
            fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                         label=spec['cbar_label'])
        fig.suptitle(f"{triad}: switch locations colored by {spec['title_desc']}",
                     fontsize=13)

        if save_dir is not None:
            path = os.path.join(save_dir, f'target_theta_switch_{suf}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved to {path}")
        out.append((col, fig))
    return out


def plot_switch_points_by_motion(assay_dfs, sex_map, focal_flies_map=None, triad='MFF',
                                 n_grid=50, lim=180, action_col='courtship',
                                 switch_source='manual',
                                 metric_col='target_ang_vel_fov_signed_deg',
                                 vlim_percentile=95, save_dir=None, figsize=None):
    """Switches in the joint (θ to target x, θ to target y) space, kept on two separate
    rows (switches → target x / switches → target y), with switch points colored by the
    target's FOV angular velocity (target_ang_vel_fov_signed_deg; + progressive / −
    regressive, prog_regr colormap) of the:
        new target   -> target_theta_switch_fovnew.png
        old target   -> target_theta_switch_fovold.png
        difference   -> target_theta_switch_fovdiff.png   (new − old; + = new more
                        progressive than old)
    Cols = MFF assays. Grey background = courtship-frame occupancy. Returns the list of
    (column, fig) produced.
    """
    data = _switch_fov_data(assay_dfs, sex_map, focal_flies_map=focal_flies_map,
                            triad=triad, action_col=action_col, switch_source=switch_source,
                            metric_col=metric_col)
    if not data or all(sw.empty for _, sw in data.values()):
        print(f"[warn] no manual switches found for {triad}; skipping motion switch plots")
        return []
    if not any((not sw.empty) and ('fov_new' in sw.columns) for _, sw in data.values()):
        print(f"[warn] '{metric_col}' not available; skipping motion switch plots")
        return []

    # signed metric -> symmetric scale about 0 (shared vmax for new/old; own for Δ)
    def _sym_vmax(cols):
        vals = [sw[c].dropna().to_numpy() for _, sw in data.values()
                for c in cols if (not sw.empty) and c in sw.columns]
        allv = np.concatenate(vals) if vals else np.array([0.0])
        v = float(np.nanpercentile(np.abs(allv), vlim_percentile)) if allv.size else 1.0
        return v or 1.0

    no_vmax = _sym_vmax(['fov_new', 'fov_old'])
    d_vmax = _sym_vmax(['fov_diff'])
    colorings = [
        {'col': 'fov_new', 'suf': 'fovnew', 'vmin': -no_vmax, 'vmax': no_vmax,
         'title_desc': 'new target FOV ang. vel.',
         'cbar_label': 'new target FOV ang. vel. (deg/s; + prog / − reg)'},
        {'col': 'fov_old', 'suf': 'fovold', 'vmin': -no_vmax, 'vmax': no_vmax,
         'title_desc': 'old target FOV ang. vel.',
         'cbar_label': 'old target FOV ang. vel. (deg/s; + prog / − reg)'},
        {'col': 'fov_diff', 'suf': 'fovdiff', 'vmin': -d_vmax, 'vmax': d_vmax,
         'title_desc': 'Δ FOV ang. vel. (new − old)',
         'cbar_label': 'Δ FOV ang. vel. (new − old, deg/s; + = new more prog)'},
    ]
    return _draw_switch_value_maps(data, colorings, putil.prog_regr_cmap(), triad=triad,
                                   n_grid=n_grid, lim=lim, save_dir=save_dir,
                                   figsize=figsize)


def plot_switch_points_by_distance(assay_dfs, sex_map, focal_flies_map=None, triad='MFF',
                                   n_grid=50, lim=180, action_col='courtship',
                                   switch_source='manual',
                                   metric_col='dist_to_other_body_adj',
                                   vlim_percentile=98, save_dir=None, figsize=None):
    """Same layout as plot_switch_points_by_motion (2 rows: switches → target x / → y,
    cols = MFF assays) but switch points colored by the focal–target distance
    (dist_to_other_body_adj) on a blue→red diverging scale of the:
        new target   -> target_theta_switch_distnew.png   (blue = near, red = far)
        old target   -> target_theta_switch_distold.png   (shared scale with new)
        difference   -> target_theta_switch_distdiff.png  (new − old; blue = new closer,
                        red = new farther; symmetric about 0)
    Grey background = courtship-frame occupancy. Returns the list of (column, fig).
    """
    data = _switch_fov_data(assay_dfs, sex_map, focal_flies_map=focal_flies_map,
                            triad=triad, action_col=action_col, switch_source=switch_source,
                            metric_col=metric_col)
    if not data or all(sw.empty for _, sw in data.values()):
        print(f"[warn] no manual switches found for {triad}; skipping distance switch plots")
        return []
    if not any((not sw.empty) and ('fov_new' in sw.columns) for _, sw in data.values()):
        print(f"[warn] '{metric_col}' not available; skipping distance switch plots")
        return []

    def _pool(cols):
        vals = [sw[c].dropna().to_numpy() for _, sw in data.values()
                for c in cols if (not sw.empty) and c in sw.columns]
        return np.concatenate(vals) if vals else np.array([0.0])

    # new/old are positive distances -> shared low/high range (blue=near, red=far);
    # the difference is signed -> symmetric about 0.
    no = _pool(['fov_new', 'fov_old'])
    lo = float(np.nanpercentile(no, 100 - vlim_percentile)) if no.size else 0.0
    hi = float(np.nanpercentile(no, vlim_percentile)) if no.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    d = _pool(['fov_diff'])
    d_vmax = float(np.nanpercentile(np.abs(d), vlim_percentile)) if d.size else 1.0
    d_vmax = d_vmax or 1.0

    cmap = plt.get_cmap('RdBu_r')                      # blue = low/near, red = high/far
    colorings = [
        {'col': 'fov_new', 'suf': 'distnew', 'vmin': lo, 'vmax': hi,
         'title_desc': 'new target distance',
         'cbar_label': 'new target distance (body-adj, px; blue = near, red = far)'},
        {'col': 'fov_old', 'suf': 'distold', 'vmin': lo, 'vmax': hi,
         'title_desc': 'old target distance',
         'cbar_label': 'old target distance (body-adj, px; blue = near, red = far)'},
        {'col': 'fov_diff', 'suf': 'distdiff', 'vmin': -d_vmax, 'vmax': d_vmax,
         'title_desc': 'Δ distance (new − old)',
         'cbar_label': 'Δ distance (new − old, body-adj px; blue = new closer)'},
    ]
    return _draw_switch_value_maps(data, colorings, cmap, triad=triad, n_grid=n_grid,
                                   lim=lim, save_dir=save_dir, figsize=figsize)


def plot_target_turn_maps(assay_dfs, sex_map, focal_flies_map=None, n_grid=50, lim=180,
                          min_count=20, courtship_only=False, action_col='courtship',
                          save_dir=None, figsize=None):
    """Mean signed fly angular velocity (ang_vel_fly, deg/s — raw rotation direction,
    + = one way, − = the other) over the joint (θ to target x, θ to target y) space, one
    panel per assay. ang_vel_fly is a single per-frame value (target-independent), so this
    is a single map rather than per-target panels; read 'turns toward x vs y' from whether
    the structure is vertical (organized by θ_x → tracking target x) or horizontal
    (θ_y → target y). Diverging colormap centered at 0. courtship_only restricts to frames
    where the focal is courting either target. Cells with < min_count frames masked. Saves
    target_theta_angvel_{courtship|allframes}.png.
    """
    fov = {}
    for key in sorted(assay_dfs):
        t = tutil.get_target_pair_fov(assay_dfs[key], sex_map,
                                      focal_flies_map=focal_flies_map, action_col=action_col)
        if t.empty or 'ang_vel_fly_degs' not in t.columns:
            continue
        if courtship_only:
            t = t[(t['court_x'] == 1) | (t['court_y'] == 1)]
        if not t.empty:
            fov[key] = t
    if not fov:
        print("[warn] no ang_vel_fly data for angular-velocity maps")
        return None, None
    assay_types = sorted(fov.keys())
    x_lim = y_lim = (-lim, lim)

    grids = {}
    for key in assay_types:
        grids[key] = _compute_2d_binned_stat(
            fov[key], 'theta_x_deg', 'theta_y_deg', 'ang_vel_fly_degs',
            x_lim, y_lim, n_grid, stat='mean', min_count=min_count)
    finite = [np.abs(g[0][np.isfinite(g[0])]).ravel()
              for g in grids.values() if np.isfinite(g[0]).any()]
    vmax = float(np.nanpercentile(np.concatenate(finite), 99)) if finite else 1.0
    vmax = vmax or 1.0

    # rows = species, cols = scenario (fixed order) → always 2 rows for the standard
    # species set, regardless of how many scenarios (4-way all-frames or 8-way courtship).
    species = sorted({k.split('_', 1)[0] for k in assay_types})
    scen_order = ['MFF', 'MMF', 'MMF_1M', 'MMF_2M', 'MMF_2M_F']
    scenarios = [s for s in scen_order if any(f'{sp}_{s}' in fov for sp in species)]
    scenarios += sorted({k.split('_', 1)[1] for k in assay_types} - set(scenarios))
    nrows, ncols = len(species), len(scenarios)
    if figsize is None:
        figsize = (3.4 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    cmap = _diverging_cmap()
    im = None
    for i, sp in enumerate(species):
        for j, sc in enumerate(scenarios):
            ax = axes[i][j]
            key = f'{sp}_{sc}'
            if key not in fov:
                ax.axis('off')
                continue
            xl, yl = fov[key]['x_label'].iloc[0], fov[key]['y_label'].iloc[0]
            g, _c, xe, ye = grids[key]
            im = ax.imshow(np.ma.masked_invalid(g.T), origin='lower', aspect='equal',
                           cmap=cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                           vmin=-vmax, vmax=vmax)
            ax.axhline(0, color='0.5', lw=0.4, alpha=0.6)
            ax.axvline(0, color='0.5', lw=0.4, alpha=0.6)
            ax.plot([-lim, lim], [-lim, lim], color='0.5', lw=0.5, alpha=0.4, ls='--')
            ax.set_xlim(x_lim); ax.set_ylim(y_lim)
            _set_theta_ticks(ax, lim)
            ax.set_title(f'{key}\n(x: {xl}, y: {yl})', fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel('θ to target x (deg)')
            if j == 0:
                ax.set_ylabel('θ to target y (deg)')
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                     label='mean ang_vel_fly (deg/s, signed by rotation direction)')
    scope = 'courtship frames' if courtship_only else 'all frames'
    fig.suptitle(f'Fly angular velocity (signed) over the target θ–θ space ({scope})',
                 fontsize=13)

    if save_dir is not None:
        suf = 'courtship' if courtship_only else 'allframes'
        path = os.path.join(save_dir, f'target_theta_angvel_{suf}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def plot_courtship_theta_vs_angvel_scatter(assay_dfs, sex_map, focal_flies_map=None,
                                           lim=180, xlim_display=90,
                                           action_col='courtship', assay_colors=None,
                                           max_points=8000, yclip_percentile=99,
                                           min_example_points=3000, vel_range=None,
                                           vel_label=None, save_suffix='',
                                           save_dir=None, figsize=None):
    """Scatter of θ-error to the PURSUED target (deg) vs the focal's angular velocity
    (ang_vel_fly, deg/s) over all courtship frames, with a per-panel OLS regression.

    The pursued target's bearing is θ_x where the courted target is x (court_x==1) and
    θ_y where it is y (court_y==1) — i.e. the bearing to whichever fly the focal is
    courting that frame. Laid out 2×3: rows = species, cols = {MFF, MMF-1M, MMF-2M}
    (pass the 6-way split assay_dfs). The regression is fit on courtship points with
    |θ-error| ≤ xlim_display (large-bearing frames excluded); the x-axis is shown over
    ±xlim_display and y over a shared symmetric display range. slope, Pearson r, p and n
    are annotated. All points are drawn light grey; one randomly-sampled acquisition per
    panel (with ≥ min_example_points in-range frames; else the one with the most in-range
    frames) is highlighted in the assay color, and the regression line is a lighter shade
    of it. Colors come from assay_colors keyed by the base species_triad, with MMF-1M
    lightened (matching the multiplicity plot). The example acquisition name is captioned
    under each panel. Saves courtship_theta_vs_angvel_scatter{save_suffix}.png.

    vel_range=(lo, hi) (mm/s, either bound None) restricts to courtship frames where the
    focal fly's own speed (vel_mm_s) is in that range — for the focal-velocity-binned
    version; vel_label annotates the bin in the title.
    """
    keep_cols = ['acquisition', 'theta_pursued', 'ang_vel_fly_degs']
    data = {}
    for key in sorted(assay_dfs):
        t = tutil.get_target_pair_fov(assay_dfs[key], sex_map,
                                      focal_flies_map=focal_flies_map, action_col=action_col)
        if t.empty or 'ang_vel_fly_degs' not in t.columns:
            continue
        c = t[(t['court_x'] == 1) | (t['court_y'] == 1)].copy()
        if c.empty:
            continue
        c['theta_pursued'] = np.where(c['court_x'] == 1, c['theta_x_deg'], c['theta_y_deg'])
        if vel_range is not None:                       # focal-speed bin filter
            if 'vel_mm_s' not in c.columns:
                continue
            lo, hi = vel_range
            m = c['vel_mm_s'].notna()
            if lo is not None:
                m &= c['vel_mm_s'] >= lo
            if hi is not None:
                m &= c['vel_mm_s'] <= hi
            c = c[m]
        c = c[keep_cols].dropna()
        if not c.empty:
            data[key] = c
    if not data:
        print(f"[warn] no courtship ang_vel data for scatter{save_suffix}")
        return None, None

    # 2x3 grid: rows = species, cols = scenario in a fixed order
    species = sorted({k.split('_', 1)[0] for k in data})
    scen_order = ['MFF', 'MMF_1M', 'MMF_2M', 'MMF_2M_F']
    scenarios = [s for s in scen_order if any(f'{sp}_{s}' in data for sp in species)]
    scenarios += sorted({k.split('_', 1)[1] for k in data} - set(scenarios))
    nrows, ncols = len(species), len(scenarios)
    if figsize is None:
        figsize = (3.7 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)

    # shared, symmetric y display range from pooled |ang_vel| (drops glitch outliers)
    ally = np.concatenate([d['ang_vel_fly_degs'].to_numpy() for d in data.values()])
    yl = float(np.nanpercentile(np.abs(ally), yclip_percentile)) or 1.0
    rng = np.random.default_rng(0)
    xstep = 30 if xlim_display <= 60 else 45
    xticks = np.arange(-xlim_display, xlim_display + 1, xstep)

    for i, sp in enumerate(species):
        for j, sc in enumerate(scenarios):
            ax = axes[i][j]
            d = data.get(f'{sp}_{sc}')
            if d is None or d.empty:
                ax.axis('off')
                continue
            # assay color: 2M & MFF = full base; 1M = lighter (matches the multiplicity
            # plot, which uses lighten(base, 0.45) for the 1-male share).
            base = (assay_colors or {}).get(f"{sp}_{sc.split('_')[0]}") or '#D1495B'
            color = putil.lighten(base, 0.45) if sc.endswith('_1M') else base
            line_color = putil.lighten(color, 0.30)        # lighter than points → visible
            x = d['theta_pursued'].to_numpy()
            y = d['ang_vel_fly_degs'].to_numpy()
            acq = d['acquisition'].to_numpy()
            # restrict to |θ-error| <= xlim_display: fit and scatter both exclude
            # large-bearing frames.
            fit = np.abs(x) <= xlim_display
            xf, yf, af = x[fit], y[fit], acq[fit]
            # bulk light-grey scatter (subsample only for drawing)
            xs, ys = xf, yf
            if len(xf) > max_points:
                idx = rng.choice(len(xf), max_points, replace=False)
                xs, ys = xf[idx], yf[idx]
            ax.scatter(xs, ys, s=5, alpha=0.18, color='0.75',
                       edgecolors='none', rasterized=True, zorder=2)
            # highlight one randomly-sampled acquisition (in the assay color) that has at
            # least min_example_points in-range frames; fall back to the largest. Drawn
            # subsampled so a big example doesn't bury the grey bulk.
            uacq, counts = np.unique(af, return_counts=True)
            pick = None
            if len(uacq):
                elig = uacq[counts >= min_example_points]
                if len(elig):
                    pick = elig[int(rng.integers(len(elig)))]
                else:
                    pick = uacq[int(np.argmax(counts))]
            if pick is not None:
                hidx = np.where(af == pick)[0]
                n_hl = len(hidx)
                draw = hidx if n_hl <= 3000 else rng.choice(hidx, 3000, replace=False)
                ax.scatter(xf[draw], yf[draw], s=7, alpha=0.5, color=color,
                           edgecolors='none', rasterized=True, zorder=3)
            # regression (lighter shade of the points) over the in-range points
            if len(xf) > 2 and np.ptp(xf) > 0:
                res = linregress(xf, yf)
                xx = np.array([-xlim_display, xlim_display])
                ax.plot(xx, res.intercept + res.slope * xx, color=line_color, lw=2.4,
                        zorder=4)
                ax.text(0.03, 0.97,
                        f'slope={res.slope:.3f}\nr={res.rvalue:.2f}, p={res.pvalue:.1g}\n'
                        f'n={len(xf)}', transform=ax.transAxes, va='top', ha='left',
                        fontsize=7, color='black',
                        bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.7))
            ax.axhline(0, color='0.5', lw=0.5, alpha=0.6)
            ax.axvline(0, color='0.5', lw=0.5, alpha=0.6)
            ax.set_xlim(-xlim_display, xlim_display); ax.set_ylim(-yl, yl)
            ax.set_xticks(xticks)
            ax.set_title(f'{sp}_{sc}', fontsize=10)
            # example-acquisition name as a caption UNDER each panel (not on the plot)
            if pick is not None:
                ax.set_xlabel(f'ex: {pick}  (n={n_hl})', fontsize=6, color=color)
    title = 'Courtship: θ-error to pursued target vs fly angular velocity'
    if vel_label:
        title += f'  (focal speed {vel_label} mm/s)'
    fig.suptitle(title, fontsize=13)
    fig.supxlabel('θ-error to pursued target (deg)', fontsize=11)
    fig.supylabel('ang_vel_fly (deg/s)', fontsize=11)

    if save_dir is not None:
        path = os.path.join(save_dir, f'courtship_theta_vs_angvel_scatter{save_suffix}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes


def plot_courtship_angvel_slope_per_acquisition(assay_dfs, sex_map, focal_flies_map=None,
                                                xlim_display=90, action_col='courtship',
                                                assay_colors=None, min_points_per_acq=500,
                                                save_dir=None, figsize=None):
    """Per-acquisition OLS slope of fly angular velocity (deg/s) regressed on θ-error to
    the PURSUED target (deg), over courtship frames with |θ| ≤ xlim_display, compared
    across the 6-way assays. One point per acquisition (those with < min_points_per_acq
    in-range frames are dropped), with the across-acquisition mean ± SEM overlaid. Same
    θ/ang_vel definition and assay-color convention (2M & MFF full, MMF-1M lightened) as
    plot_courtship_theta_vs_angvel_scatter; the slope summarizes how strongly the fly
    turns toward the pursued target (1/s). Saves
    courtship_angvel_slope_per_acquisition.png.
    """
    slopes = {}                          # assay key -> {acquisition: slope}
    for key in sorted(assay_dfs):
        t = tutil.get_target_pair_fov(assay_dfs[key], sex_map,
                                      focal_flies_map=focal_flies_map, action_col=action_col)
        if t.empty or 'ang_vel_fly_degs' not in t.columns:
            continue
        c = t[(t['court_x'] == 1) | (t['court_y'] == 1)].copy()
        if c.empty:
            continue
        c['theta_pursued'] = np.where(c['court_x'] == 1, c['theta_x_deg'], c['theta_y_deg'])
        c = c[['acquisition', 'theta_pursued', 'ang_vel_fly_degs']].dropna()
        c = c[np.abs(c['theta_pursued']) <= xlim_display]
        per = {}
        for acq, g in c.groupby('acquisition'):
            xv = g['theta_pursued'].to_numpy()
            yv = g['ang_vel_fly_degs'].to_numpy()
            if len(g) >= min_points_per_acq and np.ptp(xv) > 0:
                per[acq] = float(linregress(xv, yv).slope)
        if per:
            slopes[key] = per
    if not slopes:
        print("[warn] no per-acquisition slopes to plot")
        return None, None

    species = sorted({k.split('_', 1)[0] for k in slopes})
    scen_order = ['MFF', 'MMF_1M', 'MMF_2M', 'MMF_2M_F']
    scenarios = [s for s in scen_order if any(f'{sp}_{s}' in slopes for sp in species)]
    scenarios += sorted({k.split('_', 1)[1] for k in slopes} - set(scenarios))
    keys = [f'{sp}_{sc}' for sp in species for sc in scenarios if f'{sp}_{sc}' in slopes]

    if figsize is None:
        figsize = (max(6.0, 1.2 * len(keys)), 5)
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    for i, key in enumerate(keys):
        sp, sc = key.split('_', 1)
        base = (assay_colors or {}).get(f"{sp}_{sc.split('_')[0]}") or '#888888'
        color = putil.lighten(base, 0.45) if sc.endswith('_1M') else base
        vals = np.array(list(slopes[key].values()))
        jit = (rng.random(len(vals)) - 0.5) * 0.25
        ax.scatter(i + jit, vals, s=42, color=color, alpha=0.85,
                   edgecolor='black', linewidths=0.4, zorder=3)
        m = float(np.nanmean(vals))
        se = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.plot([i - 0.28, i + 0.28], [m, m], color=color, lw=3, zorder=4)
        ax.errorbar(i, m, yerr=se, color=color, capsize=4, lw=1.5, zorder=4)
    ax.axhline(0, color='0.5', lw=0.8, ls='--', alpha=0.7)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f'{k}\n(n={len(slopes[k])})' for k in keys],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('per-acquisition slope: ang_vel_fly vs θ-error to pursued (1/s)')
    ax.set_title('Courtship turn-toward-target slope per acquisition across assays',
                 fontsize=12)
    ax.margins(x=0.08)
    plt.tight_layout()

    if save_dir is not None:
        path = os.path.join(save_dir, 'courtship_angvel_slope_per_acquisition.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, ax


# Focal-speed bins (mm/s) for the θ→ang-vel slope-vs-speed plots; last bound None = open.
ANGVEL_SLOPE_VEL_BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, None)]


def plot_courtship_angvel_slope_by_velocity(assay_dfs, sex_map, focal_flies_map=None,
                                            mode='pooled', vel_bins=None, xlim_display=60,
                                            vel_source='focal', action_col='courtship',
                                            assay_colors=None, min_points=500, save_dir=None,
                                            save_suffix='', figsize=None):
    """Turn-toward-target slope as a function of linear speed (focal fly's own, or the
    pursued target's).

    The slope (1/s) is the OLS coefficient of fly angular velocity (ang_vel_fly_degs,
    deg/s) regressed on θ-error to the PURSUED target (deg), over courtship frames with
    |θ| ≤ xlim_display — the same definition as plot_courtship_theta_vs_angvel_scatter.
    Frames are binned by speed per `vel_bins` (default 0–4, 4–8, 8–12, 12–16, 16+ mm/s).
    One panel per species (shared y-axis), x = speed bin, one line per assay scenario
    {MFF, MMF-1M, MMF-2M}.

    vel_source:
      'focal'  -- bin by the focal fly's own speed (vel_mm_s).
      'target' -- bin by the PURSUED target's speed (looked up from each fly's own `vel`).

    mode:
      'pooled'  -- one slope per (species, assay, bin) fit over ALL in-range frames
                   pooled across acquisitions; error bar = 1 SE of the fitted slope.
      'per_acq' -- a slope fit per (species, assay, bin, acquisition) with ≥ min_points
                   in-range frames; every acquisition is a point (jittered), with the
                   across-acquisition mean ± SEM overlaid and means connected across bins.

    Saves courtship_angvel_slope_by_velocity_{mode}_{vel_source}{save_suffix}.png.
    """
    if vel_source not in ('focal', 'target'):
        raise ValueError(f"vel_source must be 'focal' or 'target', got {vel_source!r}")
    vel_bins = vel_bins or ANGVEL_SLOPE_VEL_BINS
    scen_order = ['MFF', 'MMF_1M', 'MMF_2M']
    bin_labels = [f'{lo:g}–{hi:g}' if hi is not None else f'{lo:g}+' for lo, hi in vel_bins]
    vel_word = 'pursued target' if vel_source == 'target' else 'focal'

    # in-range courtship points (|θ|≤xlim) per requested assay scenario. `binvel` is the
    # speed (mm/s) used for the velocity binning — the focal fly's or the pursued target's.
    pts = {}
    for key in sorted(assay_dfs):
        if key.split('_', 1)[-1] not in scen_order:
            continue
        df0 = assay_dfs[key]
        t = tutil.get_target_pair_fov(df0, sex_map,
                                      focal_flies_map=focal_flies_map, action_col=action_col)
        if t.empty or 'ang_vel_fly_degs' not in t.columns or 'vel_mm_s' not in t.columns:
            continue
        c = t[(t['court_x'] == 1) | (t['court_y'] == 1)].copy()
        if c.empty:
            continue
        c['theta_pursued'] = np.where(c['court_x'] == 1, c['theta_x_deg'], c['theta_y_deg'])
        if vel_source == 'target':
            if 'vel' not in df0.columns:
                continue
            # speed of the pursued target = each fly's own `vel`, joined on its id
            c['_tid'] = np.where(c['court_x'] == 1, c['x_id'], c['y_id'])
            vl = (df0[['acquisition', 'frame', 'id', 'vel']].dropna()
                  .drop_duplicates(['acquisition', 'frame', 'id'])
                  .rename(columns={'id': '_tid', 'vel': 'binvel'}))
            c = c.merge(vl, on=['acquisition', 'frame', '_tid'], how='left')
        else:
            c['binvel'] = c['vel_mm_s']
        c = c[['acquisition', 'theta_pursued', 'ang_vel_fly_degs', 'binvel']].dropna()
        c = c[np.abs(c['theta_pursued']) <= xlim_display]
        if not c.empty:
            pts[key] = c
    if not pts:
        print(f"[warn] no courtship ang_vel/speed data for slope-by-velocity ({mode})")
        return None, None

    species = sorted({k.split('_', 1)[0] for k in pts})
    scenarios = [s for s in scen_order if any(f'{sp}_{s}' in pts for sp in species)]

    def _bin_mask(v, lohi):
        lo, hi = lohi
        m = v >= lo
        return m & (v < hi) if hi is not None else m

    # results[key] = per-bin: pooled -> {slope, se, n}|None ; per_acq -> np.array(acq slopes)
    results, allv = {}, []
    for sp in species:
        for sc in scenarios:
            key = f'{sp}_{sc}'
            if key not in pts:
                continue
            d, perbin = pts[key], []
            for lohi in vel_bins:
                b = d[_bin_mask(d['binvel'].to_numpy(), lohi)]
                if mode == 'pooled':
                    if len(b) > 2 and np.ptp(b['theta_pursued']) > 0:
                        r = linregress(b['theta_pursued'], b['ang_vel_fly_degs'])
                        perbin.append({'slope': float(r.slope), 'se': float(r.stderr),
                                       'n': len(b)})
                        allv += [r.slope - r.stderr, r.slope + r.stderr]
                    else:
                        perbin.append(None)
                else:
                    acc = [float(linregress(g['theta_pursued'], g['ang_vel_fly_degs']).slope)
                           for _, g in b.groupby('acquisition')
                           if len(g) >= min_points and np.ptp(g['theta_pursued']) > 0]
                    perbin.append(np.array(acc))
                    allv += acc
            results[key] = perbin
    if not allv:
        print(f"[warn] no slopes for slope-by-velocity ({mode})")
        return None, None

    lo_y, hi_y = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = 0.08 * ((hi_y - lo_y) or 1.0)
    ylim = (lo_y - pad, hi_y + pad)

    x = np.arange(len(vel_bins))
    offs = (np.linspace(-0.2, 0.2, len(scenarios)) if mode == 'per_acq'
            else np.zeros(len(scenarios)))
    fig, axes = plt.subplots(1, len(species), squeeze=False, sharey=True,
                             figsize=figsize or (4.8 * len(species), 4.8))
    axes = axes[0]
    rng = np.random.default_rng(0)
    for ai, sp in enumerate(species):
        ax = axes[ai]
        for si, sc in enumerate(scenarios):
            key = f'{sp}_{sc}'
            if key not in results:
                continue
            base = (assay_colors or {}).get(f"{sp}_{sc.split('_')[0]}") or '#888888'
            color = putil.lighten(base, 0.45) if sc.endswith('_1M') else base
            xs, perbin = x + offs[si], results[key]
            if mode == 'pooled':
                ys = [c['slope'] if c else np.nan for c in perbin]
                es = [c['se'] if c else np.nan for c in perbin]
                ax.errorbar(xs, ys, yerr=es, color=color, marker='o', ms=6, lw=2,
                            capsize=3, label=sc, zorder=3)
            else:
                means = []
                for bi, acc in enumerate(perbin):
                    if len(acc):
                        jit = (rng.random(len(acc)) - 0.5) * 0.12
                        ax.scatter(xs[bi] + jit, acc, s=20, color=color, alpha=0.55,
                                   edgecolor='none', zorder=2)
                        m = float(np.nanmean(acc))
                        se = (float(np.nanstd(acc, ddof=1) / np.sqrt(len(acc)))
                              if len(acc) > 1 else 0.0)
                        ax.errorbar(xs[bi], m, yerr=se, color=color, marker='o', ms=7,
                                    capsize=4, elinewidth=1.5, lw=0, zorder=4)
                        means.append(m)
                    else:
                        means.append(np.nan)
                ax.plot(xs, means, color=color, lw=2, label=sc, zorder=3)
        ax.axhline(0, color='0.5', lw=0.8, ls='--', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_xlabel(f'{vel_word} speed bin (mm/s)')
        ax.set_title(sp, fontsize=11)
        ax.set_ylim(ylim)
        ax.legend(fontsize=8, title='assay')
    axes[0].set_ylabel('slope: ang_vel_fly vs θ-error to pursued (1/s)')
    ttl = 'per-acquisition fits' if mode == 'per_acq' else 'pooled fit'
    fig.suptitle(f'Turn-toward-target slope vs {vel_word} speed ({ttl}, |θ|≤{xlim_display:g}°)',
                 fontsize=13)
    plt.tight_layout()

    if save_dir is not None:
        path = os.path.join(
            save_dir,
            f'courtship_angvel_slope_by_velocity_{mode}_{vel_source}{save_suffix}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig, axes
