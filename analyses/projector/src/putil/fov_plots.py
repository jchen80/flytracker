"""Egocentric "two dots on the FOV" plots for the projector dot-assay.

The projector analog of the triad FOV / θ–θ occupancy maps (analyses/triad/src/putil/
fov_plots.py). A single real male chases TWO projected dots (inner + outer) under varying
LED intensities. During CHASING frames this visualizes, in the male's egocentric field of
view, where the pursued dot vs the other dot sit, the joint (θ_inner, θ_outer) occupancy,
that occupancy colored by FOV motion / male turning, and where switches between dots land.

Inner/outer dots play the role of the triad's two targets (x = inner, y = outer). Every
plot has two modes: 'aggregate' (pool valid LED intensities; panels = species) and 'split'
(one figure per species; panels = that species' LED intensities).

Reuses the schema-free low-level helpers from triad putil/_helpers and libs.plotting; the
data builders here produce a tidy per-frame table from the projector's wide schema.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.triad.src.putil._helpers import (
    _compute_2d_density, _compute_2d_binned_stat, _density_cmap, _add_focal_fly_marker)
import libs.plotting as putil
from .. import led_metadata as lm

LIM = 180


# ── style helpers (mirror triad putil/fov_plots) ──────────────────────────────

def _set_theta_ticks(ax, lim=LIM, step=90):
    """Ticks every `step` degrees on both axes (±180 → -180,-90,0,90,180)."""
    ticks = np.arange(-(int(lim) // step) * step, int(lim) + 1, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def _diverging_cmap():
    """RdBu_r with masked cells drawn in the panel background, for signed mean maps."""
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(plt.rcParams.get('axes.facecolor', '#262626'))
    return cmap


def _prob_cmap():
    """viridis with masked (low-count) cells drawn in the panel background, for p-maps."""
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(plt.rcParams.get('axes.facecolor', '#262626'))
    return cmap


def _shared_density_vmax(grids, pct=99):
    """Percentile-clipped vmax over the nonzero density of all panels (shared scale)."""
    nz = [g[g > 0].ravel() for g in grids if (g > 0).any()]
    return float(np.percentile(np.concatenate(nz), pct)) if nz else 1.0


def _sym_vmax(values, pct=99):
    """Symmetric vmax = pct-th percentile of |finite values| (>=tiny)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.nanpercentile(np.abs(v), pct)) or 1.0) if v.size else 1.0


def _save_fig(fig, save_dir, name):
    if save_dir is not None:
        path = os.path.join(save_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")


def _theta_lines(ax, lim=LIM, diagonal=True):
    ax.axhline(0, color='0.5', lw=0.4, alpha=0.6)
    ax.axvline(0, color='0.5', lw=0.4, alpha=0.6)
    if diagonal:
        ax.plot([-lim, lim], [-lim, lim], color='0.5', lw=0.5, alpha=0.35, ls='--')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    _set_theta_ticks(ax, lim)


# ── builders ──────────────────────────────────────────────────────────────────

def get_dot_pair_fov(df, chasing_only=True, tie_break='score'):
    """One row per frame with the male's egocentric view of the inner & outer dots.

    Columns:
        assay, frame, species, led_intensity, fps, px_per_mm
        theta_inner_deg, theta_outer_deg  -- signed bearing (deg, y-up CCW+) = −deg(angle_to)
        inner_x/inner_y, outer_x/outer_y  -- ego position (mm; x = dist·cos, y = −dist·sin)
        chasing_inner, chasing_outer      -- 0/1
        fovmotion_inner, fovmotion_outer  -- signed target FOV motion (deg/s, +prog/−reg)
        ang_vel_fly_degs                  -- male angular velocity (deg/s, y-up CCW+)
        pursued/other ∈ {inner,outer}, theta_pursued/other_deg, pursued/other ego x/y,
        fovmotion_pursued/fovmotion_other
    pursued = chased dot; both chased ⇒ tie-break by higher chasing score ('score') else
    dropped ('require_one'); chasing_only keeps frames where ≥1 dot is chased.

    θ-sign / ego note: angle_to_*dot is a bearing in IMAGE coords (y-DOWN). Flipping y to
    y-up gives theta = −deg(angle) and ego y = −dist·sin(angle) (matches
    analyze_switches.switch_vectors). ang_vel (rad/frame, image frame) is likewise negated
    so + = CCW in the displayed y-up frame, consistent with +θ = dot on the fly's left.
    """
    ai = df['angle_to_innerdot'].to_numpy(float)
    ao = df['angle_to_outerdot'].to_numpy(float)
    di = df['dist_to_innerdot_mm'].to_numpy(float)     # centroid distance (NOT headdist)
    do = df['dist_to_outerdot_mm'].to_numpy(float)
    out = pd.DataFrame({
        'assay': df['assay'].to_numpy(), 'frame': df['frame'].to_numpy(),
        'species': df['species'].to_numpy(), 'led_intensity': df['led_intensity'].to_numpy(),
        'fps': df['fps'].to_numpy(), 'px_per_mm': df['px_per_mm'].to_numpy(),
        'theta_inner_deg': -np.degrees(ai), 'theta_outer_deg': -np.degrees(ao),
        'inner_x': di * np.cos(ai), 'inner_y': -di * np.sin(ai),
        'outer_x': do * np.cos(ao), 'outer_y': -do * np.sin(ao),
        'chasing_inner': df['chasing_innerdot'].to_numpy(),
        'chasing_outer': df['chasing_outerdot'].to_numpy(),
        'fovmotion_inner': df['target_ang_vel_fov_innerdot_signed_deg'].to_numpy(float),
        'fovmotion_outer': df['target_ang_vel_fov_outerdot_signed_deg'].to_numpy(float),
    })
    if 'ang_vel' in df.columns:
        out['ang_vel_fly_degs'] = -(df['ang_vel'].to_numpy(float)
                                    * df['fps'].to_numpy(float) * 180.0 / np.pi)

    ci = out['chasing_inner'].to_numpy() == 1
    co = out['chasing_outer'].to_numpy() == 1
    pursued = np.full(len(out), None, dtype=object)
    pursued[ci & ~co] = 'inner'
    pursued[co & ~ci] = 'outer'
    both = ci & co
    if tie_break == 'score' and both.any():
        si = (df['chasing_innerdot_score'].to_numpy(float)
              if 'chasing_innerdot_score' in df.columns else np.zeros(len(df)))
        so = (df['chasing_outerdot_score'].to_numpy(float)
              if 'chasing_outerdot_score' in df.columns else np.zeros(len(df)))
        inner_win = np.nan_to_num(si, nan=-np.inf) >= np.nan_to_num(so, nan=-np.inf)
        pursued[both & inner_win] = 'inner'
        pursued[both & ~inner_win] = 'outer'
    out['pursued'] = pursued

    if chasing_only:
        out = out[ci | co].reset_index(drop=True)

    is_inner = (out['pursued'] == 'inner').to_numpy()
    out['other'] = np.where(out['pursued'].to_numpy() == 'inner', 'outer',
                            np.where(out['pursued'].to_numpy() == 'outer', 'inner', None))
    out['theta_pursued_deg'] = np.where(is_inner, out['theta_inner_deg'], out['theta_outer_deg'])
    out['theta_other_deg'] = np.where(is_inner, out['theta_outer_deg'], out['theta_inner_deg'])
    out['pursued_x'] = np.where(is_inner, out['inner_x'], out['outer_x'])
    out['pursued_y'] = np.where(is_inner, out['inner_y'], out['outer_y'])
    out['other_x'] = np.where(is_inner, out['outer_x'], out['inner_x'])
    out['other_y'] = np.where(is_inner, out['outer_y'], out['inner_y'])
    out['fovmotion_pursued'] = np.where(is_inner, out['fovmotion_inner'], out['fovmotion_outer'])
    out['fovmotion_other'] = np.where(is_inner, out['fovmotion_outer'], out['fovmotion_inner'])
    return out


# ── mode → figure/panel specs ─────────────────────────────────────────────────

def _figure_specs(tbl, mode):
    """[(suffix, cols, species_or_None)]; cols = [(label, subdf)].
    aggregate → one figure, panels = species. split → one figure per species, panels =
    that species' valid LED intensities."""
    sps = sorted(pd.Series(tbl['species']).dropna().unique())
    if mode == 'aggregate':
        return [('', [(sp, tbl[tbl['species'] == sp]) for sp in sps], None)]
    specs = []
    for sp in sps:
        t = tbl[tbl['species'] == sp]
        leds = sorted(pd.Series(t['led_intensity']).dropna().unique())
        cols = [(f'{led:g}%', t[t['led_intensity'] == led]) for led in leds]
        specs.append((f'_by_led_{sp}', cols, sp))
    return specs


def _suptitle(base, sp):
    return base + (f' — {sp}' if sp else '')


# ── (1) egocentric pursued/other density ──────────────────────────────────────

def plot_fov_density(tbl, mode='aggregate', n_grid=80, vmax_pct=99, save_dir=None):
    """2D egocentric density (mm) of the pursued dot and the other dot during chasing.
    Rows = {pursued, other}; cols = species (aggregate) or LED (split). Focal at origin
    facing +x. Saves fov_density[ _by_led_{species}].png."""
    roles = ['pursued', 'other']
    # keep only panels with data; in split mode there is one spec (figure) per species
    specs = [(suffix, [(lab, sub) for lab, sub in cols if len(sub) > 2], sp)
             for suffix, cols, sp in _figure_specs(tbl, mode)]
    specs = [s for s in specs if s[1]]
    if not specs:
        print('[warn] no chasing data for fov_density')
        return []
    # ONE spatial window + ONE color scale across every panel in this call, so species
    # (and, in split mode, the per-species figures) are directly comparable.
    allxy = np.vstack([sub[[f'{r}_x', f'{r}_y']].to_numpy()
                       for _, cols, _ in specs for _, sub in cols for r in roles])
    lim = float(np.nanpercentile(np.abs(allxy), vmax_pct)) or 1.0
    grids = {}
    for si, (suffix, cols, sp) in enumerate(specs):
        for lab, sub in cols:
            for r in roles:
                grids[(si, r, lab)] = _compute_2d_density(
                    sub, f'{r}_x', f'{r}_y', (-lim, lim), (-lim, lim), n_grid,
                    method='hist', ppm=None)
    vmax = _shared_density_vmax([g[0] for g in grids.values()])
    cmap = _density_cmap()
    out = []
    for si, (suffix, cols, sp) in enumerate(specs):
        ncols = len(cols)
        fig, axes = plt.subplots(2, ncols, figsize=(3.1 * ncols, 6.4), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        im = None
        for j, (lab, sub) in enumerate(cols):
            for i, r in enumerate(roles):
                ax = axes[i][j]
                h, xe, ye, n = grids[(si, r, lab)]
                im = ax.imshow(h.T, origin='lower', aspect='equal', cmap=cmap,
                               extent=[xe[0], xe[-1], ye[0], ye[-1]], vmin=0, vmax=vmax)
                _add_focal_fly_marker(ax, 1.0)         # origin dot+arrow (mm axes); no ellipse
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                if i == 0:
                    ax.set_title(f'{lab}\n(n={n})', fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'{r} dot\ny (mm)')
                if i == 1:
                    ax.set_xlabel('x (mm)')
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label='density')
        fig.suptitle(_suptitle('Dot positions on male FOV during chasing', sp), fontsize=13)
        _save_fig(fig, save_dir, f'fov_density{suffix}.png')
        out.append(fig)
    return out


# ── (2) joint θ_inner–θ_outer density ─────────────────────────────────────────

def plot_theta_joint_density(tbl, mode='aggregate', lim=LIM, n_grid=60, save_dir=None):
    """Joint density of (θ to inner dot, θ to outer dot) during chasing. Cols = species
    (aggregate) or LED (split). Saves theta_joint_density[ _by_led_{species}].png."""
    out = []
    for suffix, cols, sp in _figure_specs(tbl, mode):
        cols = [(lab, sub) for lab, sub in cols if len(sub) > 2]
        if not cols:
            continue
        grids = {lab: _compute_2d_density(sub, 'theta_inner_deg', 'theta_outer_deg',
                                          (-lim, lim), (-lim, lim), n_grid, method='hist')
                 for lab, sub in cols}
        vmax = _shared_density_vmax([g[0] for g in grids.values()])
        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 4.0), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        cmap = _density_cmap(); im = None
        for j, (lab, sub) in enumerate(cols):
            ax = axes[0][j]
            h, xe, ye, n = grids[lab]
            im = ax.imshow(h.T, origin='lower', aspect='equal', cmap=cmap,
                           extent=[xe[0], xe[-1], ye[0], ye[-1]], vmin=0, vmax=vmax)
            _theta_lines(ax, lim)
            ax.set_title(f'{lab}\n(n={n})', fontsize=10)
            ax.set_xlabel('θ to inner dot (deg)')
        axes[0][0].set_ylabel('θ to outer dot (deg)')
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02, label='density')
        fig.suptitle(_suptitle('Inner vs outer dot bearing during chasing', sp), fontsize=13)
        _save_fig(fig, save_dir, f'theta_joint_density{suffix}.png')
        out.append(fig)
    if not out:
        print('[warn] no chasing data for theta_joint_density')
    return out


def plot_courting_joint_prob(df, mode='aggregate', save_dir=None):
    """Joint probability of the two courting states: P(chasing inner ∈ {0,1}, chasing
    outer ∈ {0,1}) over ALL frames — a 2×2 table per panel showing how often the male
    courts neither dot, the inner dot only, the outer dot only, or both at once. Cols =
    species (aggregate) or LED (split). Each cell is annotated with its probability; the
    title compares observed P(both) to P(inner)·P(outer) (the independence baseline, so
    you can see whether courting the two dots co-occurs more/less than chance). Saves
    courting_joint_prob[ _by_led_{species}].png."""
    if not {'chasing_innerdot', 'chasing_outerdot'}.issubset(df.columns):
        print('[warn] chasing_innerdot/outerdot missing; skipping courting_joint_prob')
        return []
    cmap = _prob_cmap()
    out = []
    for suffix, cols, sp in _figure_specs(df, mode):
        cols = [(lab, sub) for lab, sub in cols if len(sub) > 0]
        if not cols:
            continue
        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols + 0.6, 3.6), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        im = None
        for j, (lab, sub) in enumerate(cols):
            ax = axes[0][j]
            ci = sub['chasing_innerdot'].to_numpy() == 1
            co = sub['chasing_outerdot'].to_numpy() == 1
            n = len(sub)
            # rows = court outer (no/yes, bottom→top); cols = court inner (no/yes)
            P = np.array([[(~ci & ~co).sum(), (ci & ~co).sum()],
                          [(~ci & co).sum(), (ci & co).sum()]], float) / max(n, 1)
            im = ax.imshow(P, origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='equal')
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, f'{P[r, c] * 100:.1f}%', ha='center', va='center',
                            color=('white' if P[r, c] < 0.55 else 'black'), fontsize=11)
            ax.set_xticks([0, 1]); ax.set_xticklabels(['no', 'yes'])
            ax.set_yticks([0, 1]); ax.set_yticklabels(['no', 'yes'])
            ax.set_xlabel('court inner')
            if j == 0:
                ax.set_ylabel('court outer')
            p_in, p_out, p_both = ci.mean(), co.mean(), (ci & co).mean()
            ax.set_title(f'{lab} (n={n})\nP(both)={p_both * 100:.1f}%  '
                         f'(indep {p_in * p_out * 100:.1f}%)', fontsize=8)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02,
                         label='joint probability')
        fig.suptitle(_suptitle('Joint probability of courting inner vs outer dot', sp),
                     fontsize=13)
        _save_fig(fig, save_dir, f'courting_joint_prob{suffix}.png')
        out.append(fig)
    if not out:
        print('[warn] no data for courting_joint_prob')
    return out


# ── (3a) θ–θ colored by per-dot FOV motion (inner & outer maps) ───────────────

def plot_theta_fovmotion_maps(tbl, mode='aggregate', lim=LIM, n_grid=50, n_grid_split=30,
                              min_count=15, save_dir=None):
    """Mean signed target FOV motion (deg/s, +prog/−reg) over the (θ_inner, θ_outer) space
    during chasing — rows = {inner dot, outer dot} FOV motion; cols = species (aggregate) or
    LED (split). Split mode uses a coarser grid (n_grid_split) so per-LED cells pool enough
    frames to clear min_count. Saves theta_fovmotion_maps[ _by_led_{species}].png."""
    rows = [('fovmotion_inner', 'inner dot'), ('fovmotion_outer', 'outer dot')]
    cmap = putil.prog_regr_cmap()
    ng = n_grid_split if mode == 'split' else n_grid
    out = []
    for suffix, cols, sp in _figure_specs(tbl, mode):
        cols = [(lab, sub) for lab, sub in cols if len(sub) > min_count]
        if not cols:
            continue
        grids = {(col, lab): _compute_2d_binned_stat(sub, 'theta_inner_deg', 'theta_outer_deg',
                                                     col, (-lim, lim), (-lim, lim), ng,
                                                     stat='mean', min_count=min_count)
                 for col, _ in rows for lab, sub in cols}
        finite = [np.abs(g[0][np.isfinite(g[0])]) for g in grids.values()
                  if np.isfinite(g[0]).any()]
        vmax = float(np.nanpercentile(np.concatenate(finite), 99)) if finite else 1.0
        vmax = vmax or 1.0
        ncols = len(cols)
        fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 7.0), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        im = None
        for i, (col, rlab) in enumerate(rows):
            for j, (lab, sub) in enumerate(cols):
                ax = axes[i][j]
                g, _c, xe, ye = grids[(col, lab)]
                im = ax.imshow(np.ma.masked_invalid(g.T), origin='lower', aspect='equal',
                               cmap=cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                               vmin=-vmax, vmax=vmax)
                _theta_lines(ax, lim)
                if i == 0:
                    ax.set_title(f'{lab}', fontsize=10)
                if i == 1:
                    ax.set_xlabel('θ to inner dot (deg)')
                if j == 0:
                    ax.set_ylabel(f'{rlab} FOV motion\nθ to outer dot (deg)')
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                         label='mean target FOV motion (deg/s; + prog / − reg)')
        fig.suptitle(_suptitle('Target FOV motion over the dot θ–θ space during chasing', sp),
                     fontsize=13)
        _save_fig(fig, save_dir, f'theta_fovmotion_maps{suffix}.png')
        out.append(fig)
    if not out:
        print('[warn] no chasing data for theta_fovmotion_maps')
    return out


# ── (3b) θ–θ colored by male angular velocity ─────────────────────────────────

def plot_theta_angvel_map(tbl, mode='aggregate', lim=LIM, n_grid=50, n_grid_split=30,
                          min_count=15, save_dir=None):
    """Mean male angular velocity (deg/s, signed) over the (θ_inner, θ_outer) space during
    chasing. Cols = species (aggregate) or LED (split). Split mode uses a coarser grid
    (n_grid_split) so per-LED cells pool enough frames to clear min_count. Saves
    theta_angvel_map[ _by_led_{species}].png."""
    if 'ang_vel_fly_degs' not in tbl.columns:
        print('[warn] ang_vel_fly_degs missing; skipping theta_angvel_map')
        return []
    cmap = _diverging_cmap()
    ng = n_grid_split if mode == 'split' else n_grid
    out = []
    for suffix, cols, sp in _figure_specs(tbl, mode):
        cols = [(lab, sub) for lab, sub in cols if len(sub) > min_count]
        if not cols:
            continue
        grids = {lab: _compute_2d_binned_stat(sub, 'theta_inner_deg', 'theta_outer_deg',
                                              'ang_vel_fly_degs', (-lim, lim), (-lim, lim),
                                              ng, stat='mean', min_count=min_count)
                 for lab, sub in cols}
        finite = [np.abs(g[0][np.isfinite(g[0])]) for g in grids.values()
                  if np.isfinite(g[0]).any()]
        vmax = float(np.nanpercentile(np.concatenate(finite), 99)) if finite else 1.0
        vmax = vmax or 1.0
        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 4.0), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        im = None
        for j, (lab, sub) in enumerate(cols):
            ax = axes[0][j]
            g, _c, xe, ye = grids[lab]
            im = ax.imshow(np.ma.masked_invalid(g.T), origin='lower', aspect='equal',
                           cmap=cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                           vmin=-vmax, vmax=vmax)
            _theta_lines(ax, lim)
            ax.set_title(f'{lab}', fontsize=10)
            ax.set_xlabel('θ to inner dot (deg)')
        axes[0][0].set_ylabel('θ to outer dot (deg)')
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                         label='mean ang_vel (deg/s, signed)')
        fig.suptitle(_suptitle('Male angular velocity over the dot θ–θ space during chasing', sp),
                     fontsize=13)
        _save_fig(fig, save_dir, f'theta_angvel_map{suffix}.png')
        out.append(fig)
    if not out:
        print('[warn] no chasing data for theta_angvel_map')
    return out


