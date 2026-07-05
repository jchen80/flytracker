"""All switch-related plots for the projector dot-assay, in one place.

Two families:
  - θ–θ FOV switch maps (the triad-style occupancy analogs): switch points over the
    chasing (θ_inner, θ_outer) occupancy colored by new dot / FOV motion / male turning,
    and p(switch | courtship at position) maps. These have 'aggregate' / 'split' (per-LED)
    modes and reuse the FOV table builders + style helpers in fov_plots.py.
  - Lab-frame / trajectory switch plots: switch rate vs LED, egocentric old→new target
    vectors, fly poses at the switch, peri-switch time-courses, sampled trajectories
    (several reuse the triad renderer analyses.triad.src.putil.switch_plots).

Shared low-level helpers come from plotting.py (_save, _hue_colors, TRACK_COLORS) and
fov_plots.py (θ-style helpers); switch-event extraction lives in analyze_switches.py.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import libs.plotting as putil
from .. import analyze_switches as asw
from .. import calibration as cal
from .plotting import _save, _hue_colors, _hue_label, TRACK_COLORS
from .fov_plots import (LIM, _diverging_cmap, _prob_cmap, _sym_vmax, _save_fig,
                        _theta_lines, _figure_specs, _suptitle)
from analyses.triad.src.putil._helpers import (_compute_2d_density, _compute_2d_binned_stat)


# ── switch-event tables (from the projector wide schema) ──────────────────────

def get_switch_fov(df):
    """Per switch (switching==1, switch_target set): assay, frame, species, led_intensity,
    new_dot, old_dot, theta_inner_deg/theta_outer_deg (at the switch frame),
    fovmotion_inner/fovmotion_outer (signed target FOV motion of each dot) and the new/old
    aliases fovmotion_new/fovmotion_old, fovmotion_diff = new − old, fovmotion_absdiff =
    |new| − |old| (signed magnitude difference; + = new dot's FOV motion larger), the
    egocentric mm positions of each dot (inner_x/inner_y, outer_x/outer_y; x = forward,
    +y = focal's left) and their new/old aliases (new_x/new_y, old_x/old_y),
    ang_vel_fly_degs (male turning, y-up CCW+). Empty if no switches. Computed directly
    from df (NOT restricted to chasing frames)."""
    if 'switching' not in df.columns or 'switch_target' not in df.columns:
        return pd.DataFrame()
    sw = df[(df['switching'] == 1) & df['switch_target'].notna()].copy()
    if sw.empty:
        return pd.DataFrame()
    ai = sw['angle_to_innerdot'].to_numpy(float)
    ao = sw['angle_to_outerdot'].to_numpy(float)
    di = sw['dist_to_innerdot_mm'].to_numpy(float)     # centroid distance (NOT headdist)
    do = sw['dist_to_outerdot_mm'].to_numpy(float)
    new = sw['switch_target'].astype(str).str.lower().to_numpy()
    is_new_inner = new == 'inner'
    fi = sw['target_ang_vel_fov_innerdot_signed_deg'].to_numpy(float)
    fo = sw['target_ang_vel_fov_outerdot_signed_deg'].to_numpy(float)
    # egocentric positions (mm); same y-up convention as get_dot_pair_fov / switch_vectors
    inner_x, inner_y = di * np.cos(ai), -di * np.sin(ai)
    outer_x, outer_y = do * np.cos(ao), -do * np.sin(ao)
    res = pd.DataFrame({
        'assay': sw['assay'].to_numpy(), 'frame': sw['frame'].to_numpy(),
        'species': sw['species'].to_numpy(), 'led_intensity': sw['led_intensity'].to_numpy(),
        'new_dot': np.where(is_new_inner, 'inner', 'outer'),
        'old_dot': np.where(is_new_inner, 'outer', 'inner'),
        'theta_inner_deg': -np.degrees(ai), 'theta_outer_deg': -np.degrees(ao),
        'inner_x': inner_x, 'inner_y': inner_y, 'outer_x': outer_x, 'outer_y': outer_y,
        'new_x': np.where(is_new_inner, inner_x, outer_x),
        'new_y': np.where(is_new_inner, inner_y, outer_y),
        'old_x': np.where(is_new_inner, outer_x, inner_x),
        'old_y': np.where(is_new_inner, outer_y, inner_y),
        'fovmotion_inner': fi, 'fovmotion_outer': fo,
        'fovmotion_new': np.where(is_new_inner, fi, fo),
        'fovmotion_old': np.where(is_new_inner, fo, fi),
    })
    res['fovmotion_diff'] = res['fovmotion_new'] - res['fovmotion_old']
    # difference of MAGNITUDES (signed): + = new dot's FOV motion larger than old's
    res['fovmotion_absdiff'] = res['fovmotion_new'].abs() - res['fovmotion_old'].abs()
    if 'ang_vel' in sw.columns:
        res['ang_vel_fly_degs'] = -(sw['ang_vel'].to_numpy(float)
                                    * sw['fps'].to_numpy(float) * 180.0 / np.pi)
    return res


def _theta_switch_table(df, chasing_only=True):
    """Per-frame θ to inner/outer dot + switch indicators, for p(switch | courtship at θ–θ)
    maps: is_switch (any), is_switch_inner / is_switch_outer (by dot switched TO). With
    chasing_only (default), restricts to CHASING frames so the denominator is courtship
    time at each bearing (switches outside chasing are excluded from both num & denom)."""
    d = df
    if (chasing_only and 'chasing_innerdot' in df.columns
            and 'chasing_outerdot' in df.columns):
        d = df[(df['chasing_innerdot'] == 1) | (df['chasing_outerdot'] == 1)]
    sw = d['switching'].to_numpy() if 'switching' in d.columns else np.zeros(len(d))
    tgt = (d['switch_target'].astype(str).str.lower().to_numpy()
           if 'switch_target' in d.columns else np.full(len(d), 'none'))
    is_sw = (sw == 1)
    return pd.DataFrame({
        'species': d['species'].to_numpy(),
        'led_intensity': d['led_intensity'].to_numpy(),
        'theta_inner_deg': -np.degrees(d['angle_to_innerdot'].to_numpy(float)),
        'theta_outer_deg': -np.degrees(d['angle_to_outerdot'].to_numpy(float)),
        'is_switch': is_sw.astype(float),
        'is_switch_inner': (is_sw & (tgt == 'inner')).astype(float),
        'is_switch_outer': (is_sw & (tgt == 'outer')).astype(float),
    })


def _switch_figure_specs(tbl, sw, mode):
    """Like fov_plots._figure_specs but each col carries (label, bg_subdf, switch_subdf)."""
    sps = sorted(pd.Series(tbl['species']).dropna().unique())
    def _sw(mask_col, val):
        return sw[sw[mask_col] == val] if (sw is not None and not sw.empty) else sw
    if mode == 'aggregate':
        cols = [(sp, tbl[tbl['species'] == sp], _sw('species', sp)) for sp in sps]
        return [('', cols, None)]
    specs = []
    for sp in sps:
        t = tbl[tbl['species'] == sp]
        s_sp = _sw('species', sp)
        leds = sorted(pd.Series(t['led_intensity']).dropna().unique())
        cols = []
        for led in leds:
            s = (s_sp[s_sp['led_intensity'] == led]
                 if (s_sp is not None and not s_sp.empty) else s_sp)
            cols.append((f'{led:g}%', t[t['led_intensity'] == led], s))
        specs.append((f'_by_led_{sp}', cols, sp))
    return specs


def _switch_bg(ax, bg, lim, n_grid):
    occ, xe, ye, _ = _compute_2d_density(bg, 'theta_inner_deg', 'theta_outer_deg',
                                         (-lim, lim), (-lim, lim), n_grid, method='hist')
    ax.imshow(occ.T, origin='lower', aspect='equal', cmap=plt.get_cmap('Greys'),
              extent=[xe[0], xe[-1], ye[0], ye[-1]], alpha=0.55)


# ── θ–θ switch-point maps (over the chasing occupancy) ────────────────────────

def plot_switch_points_by_newdot(tbl, sw, mode='aggregate', lim=LIM, n_grid=50, save_dir=None):
    """Switch events on the (θ_inner, θ_outer) chasing-occupancy background, colored by the
    dot switched TO (inner=orange, outer=green). Cols = species (aggregate) or LED (split).
    Saves switch_points_by_newdot[ _by_led_{species}].png."""
    if sw is None or sw.empty:
        print('[warn] no switches; skipping switch_points_by_newdot')
        return []
    out = []
    for suffix, cols, sp in _switch_figure_specs(tbl, sw, mode):
        cols = [c for c in cols if len(c[1]) > 2]
        if not cols:
            continue
        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols + 0.5, 4.0), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        for j, (lab, bg, s) in enumerate(cols):
            ax = axes[0][j]
            _switch_bg(ax, bg, lim, n_grid)
            ni = no = 0
            if s is not None and not s.empty:
                si, so = s[s['new_dot'] == 'inner'], s[s['new_dot'] == 'outer']
                ni, no = len(si), len(so)
                ax.scatter(si['theta_inner_deg'], si['theta_outer_deg'], s=28,
                           color=TRACK_COLORS['innerdot'], edgecolor='black', linewidths=0.4,
                           alpha=0.9, zorder=3, label=f'→ inner (n={ni})')
                ax.scatter(so['theta_inner_deg'], so['theta_outer_deg'], s=28,
                           color=TRACK_COLORS['outerdot'], edgecolor='black', linewidths=0.4,
                           alpha=0.9, zorder=3, label=f'→ outer (n={no})')
            _theta_lines(ax, lim)
            ax.set_title(f'{lab}', fontsize=10)
            ax.set_xlabel('θ to inner dot (deg)')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.85)
        axes[0][0].set_ylabel('θ to outer dot (deg)')
        fig.suptitle(_suptitle('Switches by new dot (grey = chasing occupancy)', sp), fontsize=13)
        _save_fig(fig, save_dir, f'switch_points_by_newdot{suffix}.png')
        out.append(fig)
    return out


def plot_switch_points_by_motion(tbl, sw, mode='aggregate', lim=LIM, n_grid=50,
                                 vlim_pct=95, save_dir=None):
    """Switches in the (θ_inner, θ_outer) space, rows = {switches→inner, switches→outer},
    colored by the target FOV motion of the new dot / old dot / Δ(new−old) [signed,
    prog_regr] and |new|−|old| [signed magnitude difference, RdBu_r] — 4 figures. Cols =
    species (aggregate) or LED (split). Saves switch_points_by_motion_{fovnew,fovold,
    fovdiff,fovabsdiff}[ _by_led_{species}].png."""
    if sw is None or sw.empty:
        print('[warn] no switches; skipping switch_points_by_motion')
        return []
    if 'fovmotion_new' not in sw.columns:
        print('[warn] FOV-motion columns missing; skipping switch_points_by_motion')
        return []
    rows = [('inner', 'switches → inner'), ('outer', 'switches → outer')]
    # kind: 'prog' = signed, symmetric prog_regr (toward/away of optic flow); 'mag' = signed
    # magnitude difference, symmetric neutral diverging (RdBu_r).
    colorings = [('fovmotion_new', 'new dot FOV motion', 'fovnew', 'prog'),
                 ('fovmotion_old', 'old dot FOV motion', 'fovold', 'prog'),
                 ('fovmotion_diff', 'Δ FOV motion (new − old)', 'fovdiff', 'prog'),
                 ('fovmotion_absdiff', '|new| − |old| FOV motion', 'fovabsdiff', 'mag')]
    out = []
    for col, desc, suf, kind in colorings:
        if col not in sw.columns:
            continue
        vmax = _sym_vmax(sw[col].to_numpy(), vlim_pct)
        vmin = -vmax
        if kind == 'prog':
            cmap = putil.prog_regr_cmap()
            cbar = f'{desc} (deg/s; + prog / − reg)'
        else:
            cmap = _diverging_cmap()
            cbar = f'{desc} (deg/s; + = new larger)'
        for suffix, cols, sp in _switch_figure_specs(tbl, sw, mode):
            cols = [c for c in cols if len(c[1]) > 2]
            if not cols:
                continue
            ncols = len(cols)
            fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols + 0.6, 7.0), squeeze=False,
                                     sharex=True, sharey=True, constrained_layout=True)
            sc = None
            for j, (lab, bg, s) in enumerate(cols):
                for i, (nd, rlab) in enumerate(rows):
                    ax = axes[i][j]
                    _switch_bg(ax, bg, lim, n_grid)
                    n_ev = 0
                    if s is not None and not s.empty:
                        sub = s[s['new_dot'] == nd].dropna(subset=[col])
                        n_ev = len(sub)
                        if n_ev:
                            sc = ax.scatter(sub['theta_inner_deg'], sub['theta_outer_deg'], s=34,
                                            c=sub[col], cmap=cmap, vmin=vmin, vmax=vmax,
                                            edgecolor='black', linewidths=0.4, zorder=3)
                    _theta_lines(ax, lim)
                    if i == 0:
                        ax.set_title(f'{lab}', fontsize=10)
                    if i == 1:
                        ax.set_xlabel('θ to inner dot (deg)')
                    if j == 0:
                        ax.set_ylabel(f'{rlab}\nθ to outer dot (deg)')
                    ax.text(0.03, 0.97, f'n={n_ev}', transform=ax.transAxes, va='top',
                            ha='left', color='black', fontsize=8)
            if sc is not None:
                fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label=cbar)
            fig.suptitle(_suptitle(f'Switch locations colored by {desc}', sp), fontsize=13)
            _save_fig(fig, save_dir, f'switch_points_by_motion_{suf}{suffix}.png')
            out.append(fig)
    return out


def plot_switch_points_by_angvel(tbl, sw, mode='aggregate', lim=LIM, n_grid=50,
                                 vlim_pct=95, save_dir=None):
    """Switch events on the chasing-occupancy background, colored by the male's angular
    velocity (deg/s, signed) at the switch frame. Single map; cols = species (aggregate) or
    LED (split). Saves switch_points_by_angvel[ _by_led_{species}].png."""
    if sw is None or sw.empty or 'ang_vel_fly_degs' not in sw.columns:
        print('[warn] no switches / ang_vel; skipping switch_points_by_angvel')
        return []
    cmap = _diverging_cmap()
    vmax = _sym_vmax(sw['ang_vel_fly_degs'].to_numpy(), vlim_pct)
    out = []
    for suffix, cols, sp in _switch_figure_specs(tbl, sw, mode):
        cols = [c for c in cols if len(c[1]) > 2]
        if not cols:
            continue
        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols + 0.6, 4.0), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        sc = None
        for j, (lab, bg, s) in enumerate(cols):
            ax = axes[0][j]
            _switch_bg(ax, bg, lim, n_grid)
            n_ev = 0
            if s is not None and not s.empty:
                sub = s.dropna(subset=['ang_vel_fly_degs'])
                n_ev = len(sub)
                if n_ev:
                    sc = ax.scatter(sub['theta_inner_deg'], sub['theta_outer_deg'], s=32,
                                    c=sub['ang_vel_fly_degs'], cmap=cmap, vmin=-vmax, vmax=vmax,
                                    edgecolor='black', linewidths=0.4, zorder=3)
            _theta_lines(ax, lim)
            ax.set_title(f'{lab}', fontsize=10)
            ax.set_xlabel('θ to inner dot (deg)')
            ax.text(0.03, 0.97, f'n={n_ev}', transform=ax.transAxes, va='top', ha='left',
                    color='black', fontsize=8)
        axes[0][0].set_ylabel('θ to outer dot (deg)')
        if sc is not None:
            fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                         label='male ang_vel at switch (deg/s, signed)')
        fig.suptitle(_suptitle('Switch locations colored by male angular velocity', sp),
                     fontsize=13)
        _save_fig(fig, save_dir, f'switch_points_by_angvel{suffix}.png')
        out.append(fig)
    return out


# ── 4×4 dot-configuration "modes" (position × FOV-motion of the two dots) ─────

# Each frame/switch is assigned a position combo (sign of θ to each dot; left = θ ≥ 0,
# right = θ < 0) and a motion combo (sign of each dot's FOV motion; prog = +, reg = −).
# The two together give a 4×4 grid of dot-configuration "modes".
_POS_LABELS = ['inner L · outer L', 'inner L · outer R', 'inner R · outer L', 'inner R · outer R']
_MOT_LABELS = ['inner prog\nouter prog', 'inner prog\nouter reg',
               'inner reg\nouter prog', 'inner reg\nouter reg']
# compact 16-mode labels for the bar charts (position code · motion code)
_MODE_SHORT = [f'{p}·{m}' for p in ['LL', 'LR', 'RL', 'RR'] for m in ['pp', 'pr', 'rp', 'rr']]


def _mode_indices(theta_inner, theta_outer, fov_inner, fov_outer):
    """(pos_idx, mot_idx, valid): row in 0–3 (position combo) and col in 0–3 (motion
    combo) for each event, plus a finite-input mask. Encoding matches _POS/_MOT_LABELS:
        pos = 2·(θ_inner < 0) + (θ_outer < 0)     # left = θ ≥ 0, right = θ < 0
        mot = 2·(fov_inner < 0) + (fov_outer < 0)  # prog = ≥ 0, reg = < 0
    """
    ti = np.asarray(theta_inner, float); to = np.asarray(theta_outer, float)
    fi = np.asarray(fov_inner, float); fo = np.asarray(fov_outer, float)
    valid = np.isfinite(ti) & np.isfinite(to) & np.isfinite(fi) & np.isfinite(fo)
    pos = (2 * (ti < 0) + (to < 0)).astype(int)
    mot = (2 * (fi < 0) + (fo < 0)).astype(int)
    return pos, mot, valid


def _mode_grid(theta_inner, theta_outer, fov_inner, fov_outer):
    """4×4 count grid (rows = position combo, cols = motion combo) of events."""
    pos, mot, valid = _mode_indices(theta_inner, theta_outer, fov_inner, fov_outer)
    H, _, _ = np.histogram2d(pos[valid], mot[valid], bins=[np.arange(5) - 0.5] * 2)
    return H


def _mode_flat(theta_inner, theta_outer, fov_inner, fov_outer, nmodes=16):
    """Per-event flat mode index in 0–15 (= pos·4 + mot), finite events only."""
    pos, mot, valid = _mode_indices(theta_inner, theta_outer, fov_inner, fov_outer)
    return (pos * 4 + mot)[valid]


def plot_switch_modes_positions(sw, save_dir=None, dist_pct=97, figsize=None):
    """For each of the 16 dot-configuration modes (position combo × FOV-motion combo),
    draw the egocentric old→new switch vectors: an arrow from the abandoned (old) dot to
    the switched-to (new) dot, colored by the new dot (inner/outer), with the focal male
    at the origin facing +x. One figure per species; rows = position combo, cols = motion
    combo. Saves switch_modes_positions_{species}.png."""
    if sw is None or sw.empty or 'new_x' not in sw.columns:
        print('[warn] no switches / ego positions; skipping switch_modes_positions')
        return []
    out = []
    for sp in sorted(pd.Series(sw['species']).dropna().unique()):
        s = sw[sw['species'] == sp]
        pos, mot, valid = _mode_indices(s['theta_inner_deg'], s['theta_outer_deg'],
                                        s['fovmotion_inner'], s['fovmotion_outer'])
        s = s[valid]; pos, mot = pos[valid], mot[valid]
        if s.empty:
            continue
        xy = np.abs(s[['old_x', 'old_y', 'new_x', 'new_y']].to_numpy().ravel())
        xy = xy[np.isfinite(xy)]
        lim = max(float(np.nanpercentile(xy, dist_pct)) if xy.size else 10.0, 1.0)
        fig, axes = plt.subplots(4, 4, figsize=figsize or (12.5, 13), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        for r in range(4):
            for c in range(4):
                ax = axes[r][c]
                g = s[(pos == r) & (mot == c)]
                # focal fly at origin facing +x (drawn dark for the light theme)
                ax.axhline(0, color='0.7', lw=0.5, zorder=0)
                ax.axvline(0, color='0.7', lw=0.5, zorder=0)
                ax.scatter(0, 0, s=70, color='0.15', marker='o', zorder=5)
                ax.annotate('', xy=(0.18 * lim, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='0.15', lw=1.6), zorder=5)
                for nd, col in (('inner', TRACK_COLORS['innerdot']),
                                ('outer', TRACK_COLORS['outerdot'])):
                    gg = g[g['new_dot'] == nd]
                    if gg.empty:
                        continue
                    ax.quiver(gg['old_x'], gg['old_y'], gg['new_x'] - gg['old_x'],
                              gg['new_y'] - gg['old_y'], angles='xy', scale_units='xy',
                              scale=1, color=col, alpha=0.5, width=0.006, zorder=3)
                    ax.scatter(gg['old_x'], gg['old_y'], s=18, facecolor='none',
                               edgecolor=col, linewidths=0.8, alpha=0.7, zorder=4)
                    ax.scatter(gg['new_x'], gg['new_y'], s=18, color=col, alpha=0.85, zorder=4)
                ax.text(0.03, 0.97, f'n={len(g)}', transform=ax.transAxes, va='top',
                        ha='left', fontsize=8)
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.set_aspect('equal')
                if r == 0:
                    ax.set_title(_MOT_LABELS[c], fontsize=9)
                if r == 3:
                    ax.set_xlabel('x (mm, forward)')
                if c == 0:
                    ax.set_ylabel(f'{_POS_LABELS[r]}\ny (mm, left)', fontsize=9)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=TRACK_COLORS['innerdot'], lw=2, label='→ inner'),
                   Line2D([0], [0], color=TRACK_COLORS['outerdot'], lw=2, label='→ outer'),
                   Line2D([0], [0], marker='o', mfc='none', mec='0.4', ls='none', label='old dot'),
                   Line2D([0], [0], marker='o', color='0.4', ls='none', label='new dot')]
        fig.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
        fig.suptitle(f'Switch vectors (old → new) by dot-configuration mode — {sp} '
                     f'(n={len(s)})', fontsize=13)
        _save_fig(fig, save_dir, f'switch_modes_positions_{sp}.png')
        out.append(fig)
    if not out:
        print('[warn] no valid switches for switch_modes_positions')
    return out


def _common_modes(tbl, min_frac=0.02):
    """Flat mode indices (0–15) where EVERY species spends > min_frac of its chasing frames,
    plus {species: per-mode chasing fraction array}. Modes ordered ascending."""
    species = sorted(pd.Series(tbl['species']).dropna().unique())
    fr = {}
    for sp in species:
        t = tbl[tbl['species'] == sp]
        g = _mode_grid(t['theta_inner_deg'], t['theta_outer_deg'],
                       t['fovmotion_inner'], t['fovmotion_outer'])
        fr[sp] = g.ravel() / g.sum() if g.sum() else np.zeros(16)
    keep = [m for m in range(16) if all(fr[sp][m] > min_frac for sp in species)]
    return keep, fr


def plot_switch_modes_oldnew(tbl, sw, color_by='species', min_frac=0.02, dist_pct=99.5,
                             min_lim=25.0, save_dir=None, figsize=None):
    """Old→new egocentric switch positions, restricted to the dot-configuration modes where
    EVERY species spends > min_frac (default 2%) of its chasing time — the modes the flies
    actually occupy in both species — and further split within each mode by the dot switched
    TO (inner vs outer). Rows = species, cols = (mode × → inner/outer). Focal male at the
    origin facing +x (white marker); OLD (abandoned) target = open circle, NEW (switched-to)
    target = filled circle (no border), joined by a light-grey line (triad style). `color_by`:
        'species'   -- old grey open circle, new = filled species color
        'newdot'    -- old grey open circle, new = filled inner(orange)/outer(green) color
        'fovmotion' -- old = filled square / new = filled circle, both colored by that
                       target's FOV motion (deg/s, prog/reg)
    Saves switch_modes_oldnew_{color_by}.png."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    if sw is None or sw.empty or 'new_x' not in sw.columns:
        print('[warn] no switches / ego positions; skipping switch_modes_oldnew')
        return None
    keep, fr = _common_modes(tbl, min_frac)
    if not keep:
        print(f'[warn] no modes exceed {min_frac:.0%} chasing in all species; skipping')
        return None
    species = sorted(pd.Series(tbl['species']).dropna().unique())
    pos, mot, valid = _mode_indices(sw['theta_inner_deg'], sw['theta_outer_deg'],
                                    sw['fovmotion_inner'], sw['fovmotion_outer'])
    s_all = sw[valid].copy()
    s_all['_mode'] = (pos * 4 + mot)[valid]
    s_all = s_all[s_all['_mode'].isin(keep)]
    if s_all.empty:
        print('[warn] no switches fall in the common modes; skipping switch_modes_oldnew')
        return None
    xy = np.abs(s_all[['old_x', 'old_y', 'new_x', 'new_y']].to_numpy().ravel())
    xy = xy[np.isfinite(xy)]
    lim = max(float(np.nanpercentile(xy, dist_pct)) if xy.size else 10.0, min_lim)
    cmap = putil.prog_regr_cmap()
    vmax = max(_sym_vmax(np.concatenate([s_all['fovmotion_new'].to_numpy(),
                                         s_all['fovmotion_old'].to_numpy()]), 95), 1e-6)
    norm = Normalize(-vmax, vmax)
    # columns: each common mode split into → inner and → outer
    col_specs = [(m, nd) for m in keep for nd in ('inner', 'outer')]

    def _draw_line(ax, ox, oy, nx, ny):
        segs = [[(a, b), (c, d)] for a, b, c, d in zip(ox, oy, nx, ny)]
        ax.add_collection(LineCollection(segs, colors='0.8', linewidths=0.5,
                                         alpha=0.25, zorder=1))

    nrow, ncol = len(species), len(col_specs)
    # height matches the per-column width (~1.9 in) so the equal-aspect panels don't leave
    # a vertical gap between the species rows; +1.6 for the suptitle and x tick labels
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True, sharey=True,
                             figsize=figsize or (1.9 * ncol + 1.0, 1.9 * nrow + 1.6),
                             constrained_layout=True)
    # pull the species rows together (equal-aspect panels otherwise leave a vertical gap)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.0)
    sc_fov = None
    for i, sp in enumerate(species):
        for j, (m, nd) in enumerate(col_specs):
            ax = axes[i][j]
            g = s_all[(s_all['species'] == sp) & (s_all['_mode'] == m)
                      & (s_all['new_dot'] == nd)]
            ax.axhline(0, color='0.85', lw=0.4, zorder=0)
            ax.axvline(0, color='0.85', lw=0.4, zorder=0)
            ax.scatter(0, 0, s=55, color='white', edgecolor='0.4', linewidths=0.8, zorder=6)
            ax.annotate('', xy=(0.17 * lim, 0), xytext=(0, 0), zorder=6,
                        arrowprops=dict(arrowstyle='->', color='0.4', lw=1.2))
            if not g.empty:
                ox, oy = g['old_x'].to_numpy(), g['old_y'].to_numpy()
                nx, ny = g['new_x'].to_numpy(), g['new_y'].to_numpy()
                _draw_line(ax, ox, oy, nx, ny)
                if color_by == 'fovmotion':
                    # old = filled square, new = filled circle; both colored by FOV motion
                    ax.scatter(ox, oy, s=11, marker='s', c=g['fovmotion_old'].to_numpy(),
                               cmap=cmap, norm=norm, linewidths=0, alpha=0.85, zorder=4)
                    sc_fov = ax.scatter(nx, ny, s=9, c=g['fovmotion_new'].to_numpy(),
                                        cmap=cmap, norm=norm, linewidths=0, alpha=0.85, zorder=5)
                else:
                    new_col = (putil.species_color(sp) if color_by == 'species'
                               else TRACK_COLORS[f'{nd}dot'])
                    ax.scatter(ox, oy, s=8, facecolors='none', edgecolors='gray',
                               linewidths=0.6, alpha=0.8, zorder=4)          # old: grey open
                    ax.scatter(nx, ny, s=8, color=new_col, linewidths=0, alpha=0.8,
                               zorder=5)                                     # new: filled, no border
            ax.text(0.03, 0.97, f'n={len(g)}', transform=ax.transAxes, va='top', ha='left',
                    fontsize=7.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
            if i == 0:
                ax.set_title(f'{_MODE_SHORT[m]}\n→ {nd}', fontsize=8)
            if i == nrow - 1:
                ax.set_xlabel('x (mm)', fontsize=8)
            if j == 0:
                ax.set_ylabel(f'{sp}\ny (mm)', fontsize=9)
    # anchor the top row to its lower edge and the bottom row to its upper edge so the
    # equal-aspect squares meet in the middle instead of leaving a gap between species
    for j in range(ncol):
        axes[0][j].set_anchor('S')
        axes[nrow - 1][j].set_anchor('N')
    if color_by == 'fovmotion':
        if sc_fov is not None:
            fig.colorbar(sc_fov, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                         label='target FOV motion (deg/s; + prog / − reg)\nold = square, new = circle')
        ttl = 'colored by old/new target FOV motion'
    elif color_by == 'newdot':
        handles = [Line2D([0], [0], marker='o', color=TRACK_COLORS['innerdot'], ls='none',
                          label='new → inner'),
                   Line2D([0], [0], marker='o', color=TRACK_COLORS['outerdot'], ls='none',
                          label='new → outer'),
                   Line2D([0], [0], marker='o', mfc='none', mec='gray', ls='none', label='old')]
        fig.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
        ttl = 'new colored by dot switched to'
    else:
        handles = [Line2D([0], [0], marker='o', mfc='none', mec='gray', ls='none', label='old'),
                   Line2D([0], [0], marker='o', color='0.35', ls='none', label='new (species color)')]
        fig.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
        ttl = 'old grey · new = species color'
    fig.suptitle(f'Old → new switch positions in common modes, split by new dot ({ttl})\n'
                 f'(modes with > {min_frac:.0%} chasing in both species; '
                 f'LL/LR/RL/RR = θ-position inner·outer, pp/pr/rp/rr = prog/reg FOV motion)',
                 fontsize=11)
    _save_fig(fig, save_dir, f'switch_modes_oldnew_{color_by}.png')
    return fig


def plot_switch_mode_heatmaps(tbl, sw, save_dir=None, figsize=None):
    """How chasing time and switches distribute over the 16 dot-configuration modes, and the
    per-mode switch rate. Per species (rows), three 4×4 heatmaps (cols):

      1. chasing occupancy   -- fraction of CHASING frames falling in each mode (the baseline)
      2. switch share        -- fraction of SWITCH events falling in each mode
      3. p(switch | chasing) -- switches / chasing-frame in each mode (rate vs baseline)

    Comparing (1) vs (2) shows whether switches concentrate in modes beyond what the chasing
    baseline predicts; (3) makes that explicit as a per-frame rate. Rows = position combo
    (θ sign of each dot), cols = motion combo (FOV-motion sign of each dot). Saves
    switch_mode_heatmaps.png."""
    if tbl is None or tbl.empty:
        print('[warn] no chasing data; skipping switch_mode_heatmaps')
        return None
    species = sorted(pd.Series(tbl['species']).dropna().unique())
    if not species:
        return None
    cmap = _prob_cmap()
    grids = {}
    for sp in species:
        t = tbl[tbl['species'] == sp]
        ch = _mode_grid(t['theta_inner_deg'], t['theta_outer_deg'],
                        t['fovmotion_inner'], t['fovmotion_outer'])
        if sw is not None and not sw.empty:
            s = sw[sw['species'] == sp]
            sg = _mode_grid(s['theta_inner_deg'], s['theta_outer_deg'],
                            s['fovmotion_inner'], s['fovmotion_outer'])
        else:
            sg = np.zeros((4, 4))
        ch_frac = ch / ch.sum() if ch.sum() else np.full((4, 4), np.nan)
        sw_frac = sg / sg.sum() if sg.sum() else np.full((4, 4), np.nan)
        p_sw = np.where(ch > 0, sg / np.where(ch > 0, ch, 1), np.nan)
        grids[sp] = {'chasing occupancy': (ch_frac, ch, '%'),
                     'switch share': (sw_frac, sg, '%'),
                     'p(switch | chasing)': (p_sw, sg, 'p')}
    metrics = ['chasing occupancy', 'switch share', 'p(switch | chasing)']
    # shared color scale per metric (column) across species
    vmax = {m: max([np.nanmax(grids[sp][m][0]) if np.isfinite(grids[sp][m][0]).any() else 0
                    for sp in species] + [1e-9]) for m in metrics}

    nsp = len(species)
    fig, axes = plt.subplots(nsp, 3, figsize=figsize or (13, 4.3 * nsp), squeeze=False,
                             constrained_layout=True)
    for i, sp in enumerate(species):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            grid, counts, kind = grids[sp][m]
            im = ax.imshow(np.ma.masked_invalid(grid), origin='upper', cmap=cmap,
                           vmin=0, vmax=vmax[m], aspect='equal')
            for r in range(4):
                for c in range(4):
                    if not np.isfinite(grid[r, c]):
                        continue
                    txt = (f'{grid[r, c] * 100:.1f}%\n({int(counts[r, c])})' if kind == '%'
                           else f'{grid[r, c] * 100:.2f}%\n({int(counts[r, c])} sw)')
                    ax.text(c, r, txt, ha='center', va='center', fontsize=7.5,
                            color=('white' if grid[r, c] < 0.6 * vmax[m] else 'black'))
            ax.set_xticks(range(4)); ax.set_xticklabels(_MOT_LABELS, fontsize=7)
            ax.set_yticks(range(4)); ax.set_yticklabels(_POS_LABELS, fontsize=8)
            ax.set_title(f'{sp} — {m}', fontsize=10)
            cb_lab = 'fraction of frames/events' if kind == '%' else 'switches / chasing frame'
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cb_lab)
    fig.suptitle('Switches vs chasing baseline across dot-configuration modes', fontsize=13)
    _save_fig(fig, save_dir, 'switch_mode_heatmaps.png')
    return fig


def plot_switch_mode_bars(tbl, sw, by_led=False, common_only=False, split_newdot=False,
                          min_frac=0.02, save_dir=None, figsize=None):
    """Per-mode bar charts over the dot-configuration modes, one column per species, four rows:
      1. fraction of CHASING frames in each mode (a true share of all chasing frames)
      2. chasing frames in each mode per ACQUISITION (count ÷ #acquisitions)
      3. switches in each mode per ACQUISITION (count ÷ #acquisitions)
      4. switches ÷ chasing frames in each mode = p(switch | chasing) within the mode
    Rows 2–3 are normalized by the number of acquisitions (per species, or per species×LED
    when by_led) so counts are comparable across species/LED. y-axes are shared per row.
    Row-4 cells with < 30 chasing frames are dropped (denominator too small).

      by_led       -- group bars by LED intensity within each mode.
      common_only  -- restrict the x-axis to the modes EVERY species occupies > min_frac of
                      chasing time (the ~6 common modes) instead of all 16.
      split_newdot -- split the switch bars (row 3) into → inner (filled) stacked under
                      → outer (outline); when not by_led these use the inner/outer colors.

    Saves switch_mode_bars[ _by_led][ _common][ _split].png."""
    if tbl is None or tbl.empty:
        print('[warn] no chasing data; skipping switch_mode_bars')
        return None
    species = sorted(pd.Series(tbl['species']).dropna().unique())
    if not species:
        return None
    if common_only:
        modes_idx, _ = _common_modes(tbl, min_frac)
        if not modes_idx:
            print(f'[warn] no modes exceed {min_frac:.0%} chasing in all species; skipping bars')
            return None
    else:
        modes_idx = list(range(16))
    labels = [_MODE_SHORT[m] for m in modes_idx]
    x = np.arange(len(modes_idx))

    # one-time switch mode index (so we can subset by new_dot / LED cheaply)
    s_all = None
    if sw is not None and not sw.empty:
        p, mo, va = _mode_indices(sw['theta_inner_deg'], sw['theta_outer_deg'],
                                  sw['fovmotion_inner'], sw['fovmotion_outer'])
        s_all = sw[va].copy(); s_all['_mode'] = (p * 4 + mo)[va]

    # acquisitions per (species, led) and per species, for the per-acq normalization
    nacq_led = tbl.groupby(['species', 'led_intensity'])['assay'].nunique()
    nacq_sp = tbl.groupby('species')['assay'].nunique()

    def _chase_frac(sub):
        v = np.bincount(_mode_flat(sub['theta_inner_deg'], sub['theta_outer_deg'],
                                   sub['fovmotion_inner'], sub['fovmotion_outer']),
                        minlength=16).astype(float)
        v = v / v.sum() if v.sum() else v
        return v[modes_idx]

    def _chase_cnt(sub):
        return np.bincount(_mode_flat(sub['theta_inner_deg'], sub['theta_outer_deg'],
                                      sub['fovmotion_inner'], sub['fovmotion_outer']),
                           minlength=16).astype(float)[modes_idx]

    def _swc(sub):
        return (np.bincount(sub['_mode'].to_numpy(), minlength=16).astype(float)[modes_idx]
                if sub is not None and not sub.empty else np.zeros(len(modes_idx)))

    def _sw_rate(sub, chase_cnt, min_cell=30):
        """switches ÷ chasing frames per mode; nan where the chasing denominator < min_cell."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(chase_cnt >= min_cell, _swc(sub) / chase_cnt, np.nan)

    fsz = figsize or (max(7.5, (1.0 if common_only else 0.55) * len(modes_idx) + 2)
                      * len(species), 13.5)
    fig, axes = plt.subplots(4, len(species), squeeze=False, sharex=True, sharey='row',
                             figsize=fsz)
    for j, sp in enumerate(species):
        t = tbl[tbl['species'] == sp]
        s = s_all[s_all['species'] == sp] if s_all is not None else None
        ax_fr, ax_cf, ax_sw, ax_pr = axes[0][j], axes[1][j], axes[2][j], axes[3][j]

        def _draw_sw(xpos, sub, width, base_color, nacq):
            """Switch bars (per acq) at xpos: stacked inner+outer if split_newdot, else one bar."""
            if not split_newdot:
                ax_sw.bar(xpos, _swc(sub) / nacq, width=width, color=base_color)
                return
            inner = _swc(sub[sub['new_dot'] == 'inner'] if sub is not None else None) / nacq
            outer = _swc(sub[sub['new_dot'] == 'outer'] if sub is not None else None) / nacq
            ci = base_color if by_led else TRACK_COLORS['innerdot']
            co = base_color if by_led else TRACK_COLORS['outerdot']
            ax_sw.bar(xpos, inner, width=width, color=ci)                       # → inner: filled
            ax_sw.bar(xpos, outer, width=width, bottom=inner, facecolor='none', # → outer: outline
                      edgecolor=co, linewidth=1.0)

        if by_led:
            leds = sorted(pd.Series(t['led_intensity']).dropna().unique())
            colors = _hue_colors(leds, putil.species_color(sp))
            w = 0.8 / max(len(leds), 1)
            for k, led in enumerate(leds):
                off = (k - (len(leds) - 1) / 2) * w
                tl = t[t['led_intensity'] == led]
                nacq = float(nacq_led.get((sp, led), 0)) or 1.0
                s_led = None if s is None else s[s['led_intensity'] == led]
                ax_fr.bar(x + off, _chase_frac(tl), width=w, color=colors[led], label=f'{led:g}%')
                ax_cf.bar(x + off, _chase_cnt(tl) / nacq, width=w, color=colors[led])
                _draw_sw(x + off, s_led, w, colors[led], nacq)
                ax_pr.bar(x + off, _sw_rate(s_led, _chase_cnt(tl)), width=w, color=colors[led])
            ax_fr.legend(title='LED', fontsize=7, ncol=2)
        else:
            nacq = float(nacq_sp.get(sp, 0)) or 1.0
            ax_fr.bar(x, _chase_frac(t), width=0.8, color=putil.species_color(sp))
            ax_cf.bar(x, _chase_cnt(t) / nacq, width=0.8, color=putil.species_color(sp))
            _draw_sw(x, s, 0.8, putil.species_color(sp), nacq)
            ax_pr.bar(x, _sw_rate(s, _chase_cnt(t)), width=0.8, color=putil.species_color(sp))

        n_sw = 0 if s is None else len(s)
        ax_fr.set_title(f'{sp} (chasing n={len(t)}, switches={n_sw}, '
                        f'acq={int(nacq_sp.get(sp, 0))})', fontsize=10)
        ax_pr.set_xticks(x)
        ax_pr.set_xticklabels(labels, rotation=90, fontsize=(8 if common_only else 6))
        if not common_only:
            for ax in (ax_fr, ax_cf, ax_sw, ax_pr):
                for b in range(4, 16, 4):
                    ax.axvline(b - 0.5, color='0.85', lw=0.8, zorder=0)
    # legend distinguishing the stacked inner/outer segments
    if split_newdot:
        from matplotlib.patches import Patch
        if by_led:
            h = [Patch(facecolor='0.5', label='→ inner (filled)'),
                 Patch(facecolor='none', edgecolor='0.4', label='→ outer (outline)')]
        else:
            h = [Patch(facecolor=TRACK_COLORS['innerdot'], label='→ inner (filled)'),
                 Patch(facecolor='none', edgecolor=TRACK_COLORS['outerdot'],
                       label='→ outer (outline)')]
        axes[2][-1].legend(handles=h, fontsize=7, loc='upper right')
    axes[0][0].set_ylabel('fraction of chasing frames')
    axes[1][0].set_ylabel('chasing frames / acquisition')
    axes[2][0].set_ylabel('switches / acquisition')
    axes[3][0].set_ylabel('switches / chasing frame')
    fig.suptitle('Chasing time & switches per dot-configuration mode'
                 + (' (by LED intensity)' if by_led else '')
                 + (' — common modes' if common_only else '')
                 + '\n(LL/LR/RL/RR = θ-position of inner·outer; pp/pr/rp/rr = prog/reg FOV motion)',
                 fontsize=12)
    fig.tight_layout()
    suf = (f'{"_by_led" if by_led else ""}{"_common" if common_only else ""}'
           f'{"_split" if split_newdot else ""}')
    _save_fig(fig, save_dir, f'switch_mode_bars{suf}.png')
    return fig


def plot_switch_prob_by_led(tbl, sw, common_only=True, min_frac=0.02, per_minute=True,
                            min_chasing_frames=150, save_dir=None, figsize=None):
    """p(switch | chasing) as a function of LED intensity, one line per dot-configuration
    mode, per species. For each (species, LED, mode): rate = switches / chasing-time in that
    cell — expressed per chasing-minute (per_minute, default) or as a per-frame probability.
    Poisson ±1 SE error bars (from the switch count); a dashed black line shows the pooled
    (all-modes) rate vs LED for reference. With common_only, only the modes every species
    occupies > min_frac of chasing time are drawn. Cells with < min_chasing_frames chasing
    frames are dropped (too few to estimate). Saves switch_prob_by_led[ _common].png."""
    if tbl is None or tbl.empty:
        print('[warn] no chasing data; skipping switch_prob_by_led')
        return None
    species = sorted(pd.Series(tbl['species']).dropna().unique())
    modes_idx = (_common_modes(tbl, min_frac)[0] if common_only else list(range(16)))
    if not modes_idx:
        print('[warn] no modes to plot; skipping switch_prob_by_led')
        return None
    fps = float(tbl['fps'].iloc[0])
    scale = fps * 60.0 if per_minute else 1.0     # switches/chasing-frame → per chasing-min

    tt = tbl.copy()
    p, mo, va = _mode_indices(tt['theta_inner_deg'], tt['theta_outer_deg'],
                              tt['fovmotion_inner'], tt['fovmotion_outer'])
    tt = tt[va]; tt['_mode'] = (p * 4 + mo)[va]
    ch = tt.groupby(['species', 'led_intensity', '_mode']).size()
    ch_all = tt.groupby(['species', 'led_intensity']).size()        # pooled over modes
    if sw is not None and not sw.empty:
        q, no, vb = _mode_indices(sw['theta_inner_deg'], sw['theta_outer_deg'],
                                  sw['fovmotion_inner'], sw['fovmotion_outer'])
        ss = sw[vb].copy(); ss['_mode'] = (q * 4 + no)[vb]
        sc = ss.groupby(['species', 'led_intensity', '_mode']).size()
        sc_all = ss.groupby(['species', 'led_intensity']).size()
    else:
        sc = sc_all = pd.Series(dtype=float)

    cmap = plt.get_cmap('turbo')
    mcolor = {m: cmap((i + 0.5) / len(modes_idx)) for i, m in enumerate(modes_idx)}

    def _rate(nsw, nframes):
        rate = nsw / nframes * scale
        se = np.sqrt(nsw) / nframes * scale
        return rate, se

    fig, axes = plt.subplots(1, len(species), squeeze=False, sharey=True,
                             figsize=figsize or (5.2 * len(species), 4.4))
    for j, sp in enumerate(species):
        ax = axes[0][j]
        leds = sorted(pd.Series(tt[tt['species'] == sp]['led_intensity']).dropna().unique())
        for m in modes_idx:
            xs, ys, es = [], [], []
            for led in leds:
                nframes = float(ch.get((sp, led, m), 0))
                if nframes < min_chasing_frames:
                    continue
                r, e = _rate(float(sc.get((sp, led, m), 0)), nframes)
                xs.append(led); ys.append(r); es.append(e)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker='o', ms=4, lw=1.5, capsize=2,
                            color=mcolor[m], label=_MODE_SHORT[m])
        # pooled (all modes) reference
        xs, ys = [], []
        for led in leds:
            nframes = float(ch_all.get((sp, led), 0))
            if nframes < min_chasing_frames:
                continue
            xs.append(led); ys.append(float(sc_all.get((sp, led), 0)) / nframes * scale)
        if xs:
            ax.plot(xs, ys, color='black', lw=2.0, ls='--', marker='s', ms=4,
                    label='all modes', zorder=5)
        ax.set_title(sp, fontsize=11)
        ax.set_xlabel('LED intensity (%)')
        if j == 0:
            ax.set_ylabel('switches per chasing-min' if per_minute
                          else 'p(switch | chasing frame)')
    axes[0][-1].legend(fontsize=7, ncol=2, title='mode', framealpha=0.9)
    fig.suptitle('p(switch | chasing) vs LED intensity, by dot-configuration mode'
                 + ('  (common modes)' if common_only else ''), fontsize=12)
    fig.tight_layout()
    _save_fig(fig, save_dir, f'switch_prob_by_led{"_common" if common_only else ""}.png')
    return fig


def _read_frame_rgb(cap, frame_idx):
    """Read one BGR frame from an open cv2 VideoCapture and return it as RGB, or None."""
    cap.set(1, int(frame_idx))           # 1 == cv2.CAP_PROP_POS_FRAMES
    ok, frame = cap.read()
    return frame[:, :, ::-1] if ok else None


def plot_mode_example_frames(df, assay_type_dir, n_examples=3, tail_sec=1.0, margin_px=45,
                             save_dir=None, seed=0, figsize=None):
    """Sample example CHASING frames in each of the 16 dot-configuration modes and show the
    real video frame (allocentric / image coordinates) cropped around the fly, with each
    dot's PAST `tail_sec` (default 1 s) trajectory overlaid (inner = orange, outer = green;
    open marker = tail start, filled = current) so the direction of motion is visible. The
    focal male is marked with its heading arrow. One figure per species; rows = the 16 modes
    (position combo × FOV-motion combo), cols = sampled examples. Video frames are resolved
    with calibration.assay_video. Saves mode_example_frames_{species}.png."""
    import cv2
    need = ['angle_to_innerdot', 'angle_to_outerdot',
            'target_ang_vel_fov_innerdot_signed_deg', 'target_ang_vel_fov_outerdot_signed_deg',
            'chasing_innerdot', 'chasing_outerdot', 'innerdot_x', 'innerdot_y',
            'outerdot_x', 'outerdot_y', 'fly_x', 'fly_y']
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f'[warn] missing {missing[:2]}...; skipping mode_example_frames')
        return []
    ci, co = TRACK_COLORS['innerdot'], TRACK_COLORS['outerdot']

    def _resolve_video(assay, assay_type):
        v = cal.assay_video(assay_type_dir, assay)
        if v is None and assay_type is not None:
            v = cal.assay_video(os.path.join(assay_type_dir, str(assay_type)), assay)
        return v

    rng = np.random.default_rng(seed)
    out = []
    for sp in sorted(pd.Series(df['species']).dropna().unique()):
        d = df[df['species'] == sp]
        by_assay = {a: g.sort_values('frame') for a, g in d.groupby('assay')}
        ch = d[(d['chasing_innerdot'] == 1) | (d['chasing_outerdot'] == 1)]
        pos, mot, valid = _mode_indices(
            -np.degrees(ch['angle_to_innerdot'].to_numpy(float)),
            -np.degrees(ch['angle_to_outerdot'].to_numpy(float)),
            ch['target_ang_vel_fov_innerdot_signed_deg'].to_numpy(float),
            ch['target_ang_vel_fov_outerdot_signed_deg'].to_numpy(float))
        ch = ch[valid].reset_index(drop=True)
        flat = (pos * 4 + mot)[valid]
        # pick example rows per mode, grouped by assay so each video opens once
        picks = {m: [] for m in range(16)}
        wanted = {}                          # assay -> set of frames to read
        for m in range(16):
            idx = np.where(flat == m)[0]
            if len(idx) == 0:
                continue
            for ii in rng.choice(idx, size=min(n_examples, len(idx)), replace=False):
                row = ch.iloc[ii]
                picks[m].append(row)
                wanted.setdefault(row['assay'], set()).add(int(row['frame']))
        # read the needed frames + crop around fly+dots over the tail window
        crops = {}                           # (assay, frame) -> (rgb_crop, x0, x1, y0, y1)
        for assay, frames in wanted.items():
            g = by_assay[assay]
            at = g['assay_type'].iloc[0] if 'assay_type' in g.columns else None
            video = _resolve_video(assay, at)
            if not video:
                print(f'    [skip] no video for {assay}')
                continue
            cap = cv2.VideoCapture(video)
            H = int(cap.get(4)) or 10**6     # 4 == FRAME_HEIGHT, 3 == FRAME_WIDTH
            W = int(cap.get(3)) or 10**6
            fps = float(g['fps'].iloc[0])
            half = int(round(tail_sec * fps))
            for f0 in frames:
                rgb = _read_frame_rgb(cap, f0)
                if rgb is None:
                    continue
                w = g[(g['frame'] > f0 - half) & (g['frame'] <= f0)]
                xs = np.concatenate([w[c].to_numpy(float) for c in
                                     ('innerdot_x', 'outerdot_x', 'fly_x')])
                ys = np.concatenate([w[c].to_numpy(float) for c in
                                     ('innerdot_y', 'outerdot_y', 'fly_y')])
                xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
                if not xs.size:
                    continue
                x0 = max(0, int(xs.min() - margin_px)); x1 = min(W, int(xs.max() + margin_px))
                y0 = max(0, int(ys.min() - margin_px)); y1 = min(H, int(ys.max() + margin_px))
                crops[(assay, f0)] = (rgb[y0:y1, x0:x1], x0, x1, y0, y1)
            cap.release()

        ncol = max(n_examples, 1)
        fig, axes = plt.subplots(16, ncol, squeeze=False,
                                 figsize=figsize or (3.0 * ncol + 1.4, 2.8 * 16),
                                 constrained_layout=True)
        for m in range(16):
            for c in range(ncol):
                ax = axes[m][c]
                ax.set_xticks([]); ax.set_yticks([])
                row = picks[m][c] if c < len(picks[m]) else None
                crop = crops.get((row['assay'], int(row['frame']))) if row is not None else None
                if crop is None:
                    ax.text(0.5, 0.5, '—', transform=ax.transAxes, ha='center', va='center',
                            color='0.6')
                else:
                    img, x0, x1, y0, y1 = crop
                    ax.imshow(img, extent=[x0, x1, y1, y0])
                    g = by_assay[row['assay']]
                    fps = float(g['fps'].iloc[0]); half = int(round(tail_sec * fps))
                    f0 = int(row['frame'])
                    w = g[(g['frame'] > f0 - half) & (g['frame'] <= f0)]
                    for dot, col in (('innerdot', ci), ('outerdot', co)):
                        dx, dy = w[f'{dot}_x'].to_numpy(float), w[f'{dot}_y'].to_numpy(float)
                        ax.plot(dx, dy, color=col, lw=1.6, alpha=0.9, zorder=3)
                        if dx.size:
                            ax.scatter(dx[0], dy[0], s=14, facecolor='none', edgecolor=col,
                                       linewidths=1.0, zorder=4)              # tail start
                            ax.scatter(dx[-1], dy[-1], s=55, color=col, edgecolor='white',
                                       linewidths=0.6, zorder=5)              # now
                    fr = w.iloc[-1]
                    fx, fy, th = fr['fly_x'], fr['fly_y'], fr.get('fly_theta', np.nan)
                    if np.isfinite(fx) and np.isfinite(fy):
                        ax.scatter(fx, fy, s=30, color='white', edgecolor='black',
                                   linewidths=0.6, zorder=6)
                        if np.isfinite(th):
                            L = 0.06 * max(x1 - x0, y1 - y0)
                            ax.annotate('', xy=(fx + L * np.cos(th), fy + L * np.sin(th)),
                                        xytext=(fx, fy), zorder=6,
                                        arrowprops=dict(arrowstyle='->', color='white', lw=1.4))
                    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)   # image y-down
                    ax.text(0.03, 0.97, f'{row["led_intensity"]:g}%  fr{f0}',
                            transform=ax.transAxes, va='top', ha='left', fontsize=6.5,
                            color='white')
                ax.set_aspect('equal')
                if c == 0:
                    ax.set_ylabel(f'{_MODE_SHORT[m]}\n{_POS_LABELS[m // 4]}', fontsize=6.5)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=ci, lw=2, label='inner dot'),
                   Line2D([0], [0], color=co, lw=2, label='outer dot'),
                   Line2D([0], [0], marker='o', mfc='none', mec='0.4', ls='none',
                          label=f'−{tail_sec:g}s'),
                   Line2D([0], [0], marker='o', color='0.4', ls='none', label='now')]
        fig.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
        fig.suptitle(f'Example chasing frames by dot-configuration mode — {sp}\n'
                     f'(real frame, allocentric; {tail_sec:g}s past dot trajectory overlaid)',
                     fontsize=12)
        _save_fig(fig, save_dir, f'mode_example_frames_{sp}.png')
        out.append(fig)
    if not out:
        print('[warn] no chasing data for mode_example_frames')
    return out


# ── p(switch | courtship at θ–θ position) maps (coarse bins) ──────────────────

# Conditioning set ("prior") for the switch-probability maps. Both are a binned mean
# of the per-frame switch indicator; they differ only in the denominator (the frame
# set the mean is taken over):
#   'chasing'   -- chasing frames only:  p(switch | chasing at that θ–θ bearing)
#   'occupancy' -- every frame the fly is at that bearing: p(switch | present here)
_SWITCH_PRIOR_SPEC = {
    'chasing':   {'chasing_only': True,  'fname': 'switch_prob_map',
                  'cbar': 'p(switch | chasing at position)',
                  'title': 'p(switch | chasing) over the dot θ–θ space',
                  'denom': 'chasing'},
    'occupancy': {'chasing_only': False, 'fname': 'switch_prob_occupancy_map',
                  'cbar': 'p(switch | present at position)',
                  'title': 'p(switch | occupancy) over the dot θ–θ space',
                  'denom': 'all'},
}


def plot_switch_prob_map(df, mode='aggregate', prior='chasing', lim=LIM, n_grid=20,
                         min_count=30, save_dir=None):
    """p(a switch occurs | the fly is at (θ_inner, θ_outer)) — binned mean of a per-frame
    switch indicator on coarse n_grid×n_grid bins. Cols = species (aggregate) or LED (split).

    `prior` sets the conditioning set (the denominator of the per-bin mean):
      'chasing'   -- over CHASING frames only → p(switch | chasing at that bearing)
                     (filename switch_prob_map).
      'occupancy' -- over ALL frames at that bearing → p(switch | present here)
                     (filename switch_prob_occupancy_map).
    Cells with < min_count frames in the chosen denominator are masked. Saves
    {fname}[ _by_led_{species}].png."""
    if prior not in _SWITCH_PRIOR_SPEC:
        raise ValueError(f"prior must be one of {list(_SWITCH_PRIOR_SPEC)}, got {prior!r}")
    spec = _SWITCH_PRIOR_SPEC[prior]
    if 'switching' not in df.columns or int((df['switching'] == 1).sum()) == 0:
        print(f'[warn] no switches; skipping switch_prob_map (prior={prior})')
        return []
    t = _theta_switch_table(df, chasing_only=spec['chasing_only'])
    cmap = _prob_cmap()
    out = []
    for suffix, cols, sp in _figure_specs(t, mode):
        cols = [(lab, sub) for lab, sub in cols if len(sub) > min_count]
        if not cols:
            continue
        grids = {lab: _compute_2d_binned_stat(sub, 'theta_inner_deg', 'theta_outer_deg',
                                              'is_switch', (-lim, lim), (-lim, lim), n_grid,
                                              stat='mean', min_count=min_count)
                 for lab, sub in cols}
        finite = [g[0][np.isfinite(g[0])] for g in grids.values() if np.isfinite(g[0]).any()]
        vmax = float(np.nanpercentile(np.concatenate(finite), 99)) if finite else 1.0
        vmax = vmax or 1e-6
        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 4.0), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        im = None
        for j, (lab, sub) in enumerate(cols):
            ax = axes[0][j]
            g, _c, xe, ye = grids[lab]
            im = ax.imshow(np.ma.masked_invalid(g.T), origin='lower', aspect='equal',
                           cmap=cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                           vmin=0, vmax=vmax)
            _theta_lines(ax, lim)
            ax.set_title(f'{lab}\n(switches={int(sub["is_switch"].sum())}, '
                         f'{spec["denom"]} frames={len(sub)})', fontsize=9)
            ax.set_xlabel('θ to inner dot (deg)')
        axes[0][0].set_ylabel('θ to outer dot (deg)')
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02,
                         label=spec['cbar'])
        fig.suptitle(_suptitle(spec['title'], sp), fontsize=13)
        _save_fig(fig, save_dir, f'{spec["fname"]}{suffix}.png')
        out.append(fig)
    return out


def plot_switch_prob_by_target(df, mode='aggregate', lim=LIM, n_grid=20, min_count=30,
                               save_dir=None):
    """p(switch to a specific dot | courtship at θ–θ position): rows = {→ inner, → outer},
    binned mean of the per-target switch indicator over CHASING frames (coarse bins, shared
    color scale). Cols = species (aggregate) or LED (split). Saves
    switch_prob_by_target[ _by_led_{species}].png."""
    if 'switching' not in df.columns or int((df['switching'] == 1).sum()) == 0:
        print('[warn] no switches; skipping switch_prob_by_target')
        return []
    rows = [('is_switch_inner', 'p(switch → inner)'), ('is_switch_outer', 'p(switch → outer)')]
    t = _theta_switch_table(df)
    cmap = _prob_cmap()
    out = []
    for suffix, cols, sp in _figure_specs(t, mode):
        cols = [(lab, sub) for lab, sub in cols if len(sub) > min_count]
        if not cols:
            continue
        grids = {(col, lab): _compute_2d_binned_stat(sub, 'theta_inner_deg', 'theta_outer_deg',
                                                     col, (-lim, lim), (-lim, lim), n_grid,
                                                     stat='mean', min_count=min_count)
                 for col, _ in rows for lab, sub in cols}
        finite = [g[0][np.isfinite(g[0])] for g in grids.values() if np.isfinite(g[0]).any()]
        vmax = float(np.nanpercentile(np.concatenate(finite), 99)) if finite else 1.0
        vmax = vmax or 1e-6
        ncols = len(cols)
        fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.2), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        im = None
        for i, (col, rlab) in enumerate(rows):
            for j, (lab, sub) in enumerate(cols):
                ax = axes[i][j]
                g, _c, xe, ye = grids[(col, lab)]
                im = ax.imshow(np.ma.masked_invalid(g.T), origin='lower', aspect='equal',
                               cmap=cmap, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                               vmin=0, vmax=vmax)
                _theta_lines(ax, lim)
                if i == 0:
                    ax.set_title(f'{lab}', fontsize=10)
                if i == 1:
                    ax.set_xlabel('θ to inner dot (deg)')
                if j == 0:
                    ax.set_ylabel(f'{rlab}\nθ to outer dot (deg)')
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                         label='p(switch → target | courtship at position)')
        fig.suptitle(_suptitle('p(switch to each dot | courtship) over the dot θ–θ space', sp),
                     fontsize=13)
        _save_fig(fig, save_dir, f'switch_prob_by_target{suffix}.png')
        out.append(fig)
    return out


# ── switch rate ───────────────────────────────────────────────────────────────

def plot_switch_rate(df, hue=None, per_minute=True, assay_type_dir=None,
                     save_name='switch_rate_vs_intensity.png'):
    """Switch rate (switches / courtship time) vs LED intensity, one panel per species.

    Mean +/- SEM across assays as a line+band over faint per-assay points. `hue`
    (e.g. 'target_direction') draws one colored line per value within each panel.
    """
    summary, per = asw.switch_rate(df, hue=hue, per_minute=per_minute)
    species = sorted(summary['species'].dropna().unique())
    n_by_sp = per.groupby('species')['assay'].nunique().to_dict() if not per.empty else {}
    sp_colors = {sp: putil.species_color(sp) for sp in species}
    hue_vals = sorted(summary[hue].dropna().unique()) if hue else []
    hue_colors = {sp: _hue_colors(hue_vals, sp_colors[sp]) for sp in species} if hue else {}
    unit = 'min' if per_minute else 's'

    fig, axes = plt.subplots(1, len(species), figsize=(5 * len(species), 4),
                             squeeze=False, sharey=True)
    for j, sp in enumerate(species):
        ax = axes[0][j]
        s_sp = summary[summary['species'] == sp]
        p_sp = per[per['species'] == sp]
        groups = [(v, hue_colors[sp][v], _hue_label(hue, v)) for v in hue_vals] if hue \
            else [(None, sp_colors[sp], None)]
        for v, color, label in groups:
            s = (s_sp[s_sp[hue] == v] if hue else s_sp).sort_values('led_intensity')
            p = p_sp[p_sp[hue] == v] if hue else p_sp
            ax.scatter(p['led_intensity'], p['switch_rate'], color=color, alpha=0.3, s=20, zorder=1)
            ax.errorbar(s['led_intensity'], s['mean'], yerr=s['sem'].fillna(0), marker='o',
                        color=color, capsize=3, label=label, zorder=2)
        ax.set_title(f'{sp} (n={n_by_sp.get(sp, 0)})')
        ax.set_xlabel('LED intensity (%)')
    axes[0][0].set_ylabel(f'switches / {unit} courtship')
    if hue and hue_vals:
        axes[0][-1].legend(title=hue, fontsize=8)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name)
    return fig, axes


def plot_switch_rate_per_courtship_min(df, save_dir=None, figsize=None):
    """Overall switch rate (switches per minute of COURTSHIP), pooled across all valid LED
    intensities — one point per acquisition + mean ± SEM per species. Reuses
    analyze_switches.switch_rate_per_assay (courtship time = the module's COURTSHIP_BEHAVIORS
    = chasing/orienting either dot; valid_led applied), then sums switches & courtship-sec
    over LED per assay. Saves switch_rate_per_courtship_min.png."""
    if 'switching' not in df.columns or int((df['switching'] == 1).sum()) == 0:
        print('[warn] no switches; skipping switch_rate_per_courtship_min')
        return None
    per_led = asw.switch_rate_per_assay(df, per_minute=True)
    if per_led is None or per_led.empty:
        print('[warn] no per-assay switch rates; skipping switch_rate_per_courtship_min')
        return None
    per = (per_led.groupby(['species', 'assay'], as_index=False)
           .agg(n_switches=('n_switches', 'sum'), courtship_sec=('courtship_sec', 'sum')))
    per['switch_rate'] = np.where(per['courtship_sec'] > 0,
                                  per['n_switches'] / per['courtship_sec'] * 60.0, np.nan)
    species = sorted(per['species'].dropna().unique())
    if figsize is None:
        figsize = (1.8 * len(species) + 1.6, 4.6)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, sp in enumerate(species):
        vals = per.loc[per['species'] == sp, 'switch_rate'].dropna().to_numpy()
        color = putil.species_color(sp)
        if not len(vals):
            continue
        jit = (rng.random(len(vals)) - 0.5) * 0.3
        ax.scatter(i + jit, vals, s=42, color=color, alpha=0.85,
                   edgecolor='black', linewidths=0.4, zorder=3)
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.plot([i - 0.28, i + 0.28], [m, m], color=color, lw=3, zorder=4)
        ax.errorbar(i, m, yerr=se, color=color, capsize=4, lw=1.5, zorder=4)
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels([f'{sp}\n(n={int(per.loc[per["species"] == sp, "switch_rate"].notna().sum())})'
                        for sp in species])
    ax.set_ylabel('switches / courtship-min')
    ax.set_ylim(bottom=0)
    ax.margins(x=0.2)
    ax.set_title('Overall switch rate per courtship-minute (pooled over LED intensities)',
                 fontsize=11)
    _save_fig(fig, save_dir, 'switch_rate_per_courtship_min.png')
    return fig


# ── egocentric old→new target vectors / tails / positions ─────────────────────

def plot_switch_vectors(df, groupby=('species',), old_theta_range=None, old_motion=None,
                        assay_type_dir=None,
                        save_name='switch_target_vectors.png', **kwargs):
    """Triad-style egocentric switch-target figure (3 rows: old / new / old->new,
    one column per group), reusing analyses.triad.src.putil.switch_plots.

    Builds old/new target vectors from the projector switch annotations and hands
    them to the existing triad plotter so the figure matches the triad
    'relative switch positions' look. `groupby` sets the columns (default species;
    pass ('species','led_intensity') for one column per species x LED intensity).
    old_theta_range -- (lo, hi) in degrees: keep only switches whose OLD target's
    |theta error| at the switch falls in [lo, hi) (None = all switches).
    old_motion -- 'progressive'/'regressive': keep only switches where the OLD
    target's retinal FOV bearing-rate is +/- at the switch (None = all).
    Saves `save_name` into {assay_type_dir}/figures/.
    """
    vector_dfs, ppm_dict = asw.switch_vectors(df, groupby=groupby,
                                              old_theta_range=old_theta_range,
                                              old_motion=old_motion)
    if not vector_dfs:
        print('[warn] no switch vectors to plot')
        return None, None
    from analyses.triad.src.putil import switch_plots as tsp
    # new target in the species/assay color, old target grey (caller can override)
    kwargs.setdefault('assay_colors',
                      {lbl: putil.courtship_color(lbl) for lbl in vector_dfs})
    kwargs.setdefault('old_color', putil.OLD_TARGET_COLOR)
    # triad's plotter hardcodes its filename when save_dir is set, so save here
    # instead (lets us write per-grouping names without overwriting)
    fig, axes = tsp.plot_switch_vectors_across_assays(vector_dfs, ppm_dict, **kwargs)
    _save(fig, assay_type_dir, save_name)
    return fig, axes


def plot_switch_motion_tails(df, groupby=('species',), window_sec=2.0, tail_sec=0.25,
                             vlim_percentile=95, assay_type_dir=None,
                             save_name='switch_motion_tails.png', **kwargs):
    """Old/new target position at each switch with a short lead-in tail colored by
    the retinal FOV motion (progressive=green / regressive=magenta), reusing triad's
    plot_switch_target_tail_colored_by_metric_across_assays. One column per group.

    vlim_percentile clips the diverging color scale to +/- that percentile of
    |metric| (default 95) so a few extreme bearing rates don't wash out the rest.
    """
    traj, ppm = asw.switch_target_trajectories(df, window_sec=window_sec, groupby=groupby)
    if not traj:
        print('[warn] no switch trajectories to plot')
        return None, None
    from analyses.triad.src.putil import switch_plots as tsp
    fig, axes = tsp.plot_switch_target_tail_colored_by_metric_across_assays(
        traj, ppm, tail_sec=tail_sec, vlim_percentile=vlim_percentile, **kwargs)
    _save(fig, assay_type_dir, save_name)
    return fig, axes


def plot_switch_positions_by_motion(df, groupby=('species',), window_sec=2.2,
                                    t_rel_points=((-0.5, '0.5 s before'), (0.0, 'at switch')),
                                    vlim_percentile=95, old_theta_range=None, old_motion=None,
                                    assay_type_dir=None,
                                    save_name='switch_positions_by_motion.png', **kwargs):
    """Old vs new target positions at switch, colored by retinal FOV motion
    (progressive/regressive), reusing triad's
    plot_switch_positions_colored_by_metric_across_assays.

    vlim_percentile clips the diverging color scale to +/- that percentile of
    |metric| (default 95).
    old_theta_range -- (lo, hi) in degrees: keep only switches whose OLD target's
    |theta error| at the switch falls in [lo, hi) (None = all switches).
    old_motion -- 'progressive'/'regressive': keep only switches where the OLD
    target's retinal FOV bearing-rate is +/- at the switch (None = all).
    """
    traj, ppm = asw.switch_target_trajectories(df, window_sec=window_sec, groupby=groupby,
                                               old_theta_range=old_theta_range,
                                               old_motion=old_motion)
    if not traj:
        print('[warn] no switch trajectories to plot')
        return None, None
    from analyses.triad.src.putil import switch_plots as tsp
    fig, axes = tsp.plot_switch_positions_colored_by_metric_across_assays(
        traj, ppm, t_rel_points=t_rel_points, vlim_percentile=vlim_percentile, **kwargs)
    _save(fig, assay_type_dir, save_name)
    return fig, axes


def plot_switch_vectors_by_motion(df, groupby=('species',), window_sec=0.5,
                                  metric_window_sec=0.25, color_by='new',
                                  vlim_percentile=95, panel_size=7.0,
                                  assay_type_dir=None,
                                  save_name='switch_vectors_by_motion.png'):
    """Large egocentric figure of old->new target vectors at the switch, the line
    colored by the retinal FOV motion (magenta=regressive / green=progressive).

    One big panel per group (e.g. Dmel vs Dyak). Focal fly at the origin facing +x;
    each switch is a thick line from the old target (open square) to the new target
    (filled circle), colored by the `color_by` target's signed FOV bearing-rate,
    averaged over the [-metric_window_sec, 0] s lead-in. Color scale shared across
    panels and clipped to +/- the vlim_percentile of |metric|.
    """
    from matplotlib.colors import Normalize
    from analyses.triad.src.putil import switch_plots as tsp

    traj, ppm = asw.switch_target_trajectories(df, window_sec=window_sec, groupby=groupby)
    if not traj:
        print('[warn] no switch vectors to plot')
        return None, None
    mcol = f'{color_by}_target_ang_vel_fov_signed_deg'

    data, mags, radii = {}, [], []
    for label in sorted(traj):
        g, p = traj[label], ppm[label]
        rows = []
        for _, ev in g.groupby(['acquisition', 'switch_frame']):
            at = ev[ev['frame_offset'] == 0]
            if at.empty:
                continue
            pre = ev[(ev['t_rel'] <= 0) & (ev['t_rel'] >= -metric_window_sec)]
            val = float(pre[mcol].mean())
            r = at.iloc[0]
            ox, oy = r['old_x_ego'] / p, r['old_y_ego'] / p
            nx, ny = r['x_ego'] / p, r['y_ego'] / p
            rows.append((ox, oy, nx, ny, val))
            mags.append(abs(val))
            radii += [np.hypot(ox, oy), np.hypot(nx, ny)]
        data[label] = rows

    mags = np.array([m for m in mags if np.isfinite(m)])
    vlim = float(np.percentile(mags, vlim_percentile)) if mags.size else 1.0
    vlim = vlim if vlim > 0 else 1.0
    norm = Normalize(-vlim, vlim)
    cmap = putil.prog_regr_cmap()
    max_r = (np.nanmax(radii) * 1.05) if radii else 1.0
    ring_step = tsp._nice_ring_step(max_r)
    ring_t = np.linspace(0, 2 * np.pi, 200)

    labels = sorted(traj)
    fig, axes = plt.subplots(1, len(labels), figsize=(panel_size * len(labels), panel_size),
                             squeeze=False, sharex=True, sharey=True)
    sc = None
    for j, label in enumerate(labels):
        ax = axes[0][j]
        for r in np.arange(ring_step, max_r + ring_step, ring_step):
            ax.plot(r * np.cos(ring_t), r * np.sin(ring_t), color='gray', lw=0.5,
                    alpha=0.3, ls='--', zorder=0)
            ax.text(0, r, f'{r:.0f}', fontsize=8, color='gray', ha='center', va='bottom', alpha=0.6)
        for ox, oy, nx, ny, val in data[label]:
            c = cmap(norm(val)) if np.isfinite(val) else 'gray'
            ax.plot([ox, nx], [oy, ny], color=c, lw=3, alpha=0.9, solid_capstyle='round', zorder=2)
            ax.plot(ox, oy, marker='s', mfc='none', mec=putil.OLD_TARGET_COLOR, mew=0.8, ms=7, zorder=2)
        if data[label]:
            nxs = [d[2] for d in data[label]]
            nys = [d[3] for d in data[label]]
            vals = [d[4] for d in data[label]]
            sc = ax.scatter(nxs, nys, c=vals, cmap=cmap, norm=norm, marker='o', s=90,
                            edgecolors='white', linewidths=0.5, zorder=3)
        ax.scatter(0, 0, s=90, color='white', marker='o', edgecolors='gray', linewidths=0.8, zorder=5)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.25)
        ax.axvline(0, color='gray', lw=0.4, alpha=0.25)
        ax.set_aspect('equal')
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_xlabel('x (mm)')
        if j == 0:
            ax.set_ylabel('y (mm)')
        ax.set_title(f'{label} (n={len(data[label])})', fontsize=14)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.046, pad=0.02)
        cb.set_label(f'{color_by} target ang vel in FOV (signed, +prog/−regr) (°/s)')
    fig.suptitle('old → new target vectors at switch, colored by retinal FOV motion '
                 '(magenta=regressive, green=progressive)', fontsize=13)
    _save(fig, assay_type_dir, save_name)
    return fig, axes


# ── fly pose at the switch (lab frame) ────────────────────────────────────────

def _prep_switch_fly_poses(df, species, use_mm):
    """Shared prep for the fly-pose plots: filter to species, drop NaNs, add the
    world/y-up coordinate, and return (poses, unit, target_colors, geom) where geom
    is (cx, cy, half, arrow_len) for one shared square window. None if no poses."""
    poses = asw.switch_fly_poses(df, use_mm=use_mm)
    if species is not None:
        poses = poses[poses['species'] == species]
    # only the pose columns are always required; the case masks (theta/motion) drop
    # NaN old-target values on their own, so the no-case plot works without FOV cols
    poses = poses.dropna(subset=['fly_x', 'fly_y', 'fly_theta'])
    if poses.empty:
        return None
    unit = 'mm' if (use_mm and 'fly_x_mm' in df.columns) else 'px'
    target_colors = {'inner': TRACK_COLORS['innerdot'], 'outer': TRACK_COLORS['outerdot']}
    # world/y-up coords with POSITIVE labels: reflect image-y within its range so
    # 'up' is up while keeping the original mm magnitudes (heading flips with it).
    y0 = poses['fly_y'].min() + poses['fly_y'].max()
    poses = poses.assign(fly_y_up=y0 - poses['fly_y'])
    xs, ys = poses['fly_x'], poses['fly_y_up']
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = 0.54 * max(xs.max() - xs.min(), ys.max() - ys.min())   # ~8% margin
    return poses, unit, target_colors, (cx, cy, half, 0.10 * half)


# how to color the fly-pose plots: 'target' (categorical inner/outer) or a
# continuous per-switch metric -- 'seq' (sequential, e.g. |theta error|) or 'div'
# (diverging prog/regressive FOV motion, symmetric about 0).
POSE_COLOR_MODES = {
    'target':    {'kind': 'cat', 'title': 'new target'},
    'old_theta': {'kind': 'seq', 'col': 'old_abs_theta_deg', 'cmap': 'viridis',
                  'vmin': 0, 'vmax': 180, 'label': 'old |θ error| (deg)',
                  'title': 'old |θ error|'},
    'new_theta': {'kind': 'seq', 'col': 'new_abs_theta_deg', 'cmap': 'viridis',
                  'vmin': 0, 'vmax': 180, 'label': 'new |θ error| (deg)',
                  'title': 'new |θ error|'},
    'old_fov':   {'kind': 'div', 'col': 'old_motion_signed',
                  'label': 'old target FOV (deg/s, +prog/−reg)',
                  'title': 'old target FOV motion'},
    'new_fov':   {'kind': 'div', 'col': 'new_motion_signed',
                  'label': 'new target FOV (deg/s, +prog/−reg)',
                  'title': 'new target FOV motion'},
    # LED intensity: DISCRETE levels, one equal step along the species-color gradient
    # per level (so 2->4 and 10->20 are the same color change), not value-proportional
    'led':       {'kind': 'levels', 'col': 'led_intensity', 'suffix': '%',
                  'title': 'LED intensity'},
}


def _pose_coloring(poses, color_by, base_color='#888888', vlim_percentile=95):
    """Resolve a color_by mode against the data: returns a dict the draw helper uses.

    'cat'    -- discrete categories with fixed colors (new target inner/outer).
    'levels' -- discrete ordered levels (LED intensity): one equal step along the
                `base_color` (species) gradient per level, not value-proportional.
    'seq'/'div' -- continuous metric (sequential cmap, or diverging prog/reg with a
                symmetric vlim_percentile limit)."""
    mode = POSE_COLOR_MODES[color_by]
    if mode['kind'] == 'cat':
        return {'kind': 'cat', 'col': 'switch_target', 'fmt': str, 'title': mode['title'],
                'value_colors': {'inner': TRACK_COLORS['innerdot'],
                                 'outer': TRACK_COLORS['outerdot']}}
    if mode['kind'] == 'levels':
        levels = sorted(poses[mode['col']].dropna().unique())
        grad = putil.color_gradient(base_color, len(levels))
        suffix = mode.get('suffix', '')
        return {'kind': 'cat', 'col': mode['col'], 'title': mode['title'],
                'fmt': lambda v: f'{v:g}{suffix}',
                'value_colors': {lv: grad[i] for i, lv in enumerate(levels)}}
    finite = poses[mode['col']].astype(float)
    finite = finite[np.isfinite(finite)]
    if mode['kind'] == 'div':
        vlim = float(np.percentile(finite.abs(), vlim_percentile)) if len(finite) else 1.0
        vlim = vlim if vlim > 0 else 1.0
        norm, cmap = plt.Normalize(-vlim, vlim), putil.prog_regr_cmap()
    else:                                   # sequential
        vmin = mode.get('vmin', float(finite.min()) if len(finite) else 0.0)
        vmax = mode.get('vmax', float(finite.max()) if len(finite) else 1.0)
        norm, cmap = plt.Normalize(vmin, vmax), plt.get_cmap(mode['cmap'])
    return {'kind': 'cont', 'col': mode['col'], 'cmap': cmap, 'norm': norm,
            'label': mode['label'], 'title': mode['title']}


def _draw_switch_fly_poses(ax, poses, coloring, target_colors, geom, unit,
                           xlabel=True, ylabel=True):
    """Draw one panel of fly poses (position + heading arrow) in the shared square
    y-up window, colored per `coloring` ('cat' discrete, or a continuous metric)."""
    cx, cy, half, arrow_len = geom
    qkw = dict(angles='xy', scale_units='xy', scale=1, width=0.006,
               headwidth=3, headlength=4, zorder=2)
    if coloring['kind'] == 'cat':
        col, fmt = coloring['col'], coloring['fmt']
        drew = False
        for val, color in coloring['value_colors'].items():
            p = poses[poses[col] == val]
            if p.empty:
                continue
            drew = True
            ax.scatter(p['fly_x'], p['fly_y_up'], s=24, color=color, alpha=0.8,
                       edgecolors='none', zorder=3, label=f'{fmt(val)} (n={len(p)})')
            # heading: cos in x, -sin in y (image y-down -> world y-up)
            ax.quiver(p['fly_x'], p['fly_y_up'], np.cos(p['fly_theta']) * arrow_len,
                      -np.sin(p['fly_theta']) * arrow_len, color=color, alpha=0.8, **qkw)
        if drew:
            ax.legend(fontsize=6, loc='upper right', title=coloring['title'],
                      title_fontsize=6)
    else:
        p = poses[np.isfinite(poses[coloring['col']].astype(float))]
        if not p.empty:
            c = p[coloring['col']].astype(float).to_numpy()
            ax.scatter(p['fly_x'], p['fly_y_up'], c=c, cmap=coloring['cmap'],
                       norm=coloring['norm'], s=24, alpha=0.85, edgecolors='none', zorder=3)
            ax.quiver(p['fly_x'], p['fly_y_up'], np.cos(p['fly_theta']) * arrow_len,
                      -np.sin(p['fly_theta']) * arrow_len, c, cmap=coloring['cmap'],
                      norm=coloring['norm'], alpha=0.85, **qkw)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    if xlabel:
        ax.set_xlabel(f'x ({unit})')
    if ylabel:
        ax.set_ylabel(f'y ({unit})')


def _pose_colorbar(fig, coloring, ax):
    """Add a shared colorbar for a continuous fly-pose coloring (no-op for 'cat')."""
    if coloring['kind'] != 'cont':
        return
    sm = plt.cm.ScalarMappable(cmap=coloring['cmap'], norm=coloring['norm'])
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=coloring['label'], fraction=0.046, pad=0.02)


def plot_switch_fly_pose(df, species=None, use_mm=True, color_by='target',
                         assay_type_dir=None, save_name=None):
    """Fly lab-frame pose at the switch (all switches, not split by case).

    color_by selects the coloring (see POSE_COLOR_MODES): 'target' (inner/outer dot
    switched to), 'old_theta'/'new_theta' (|theta error| to the old/new target),
    or 'old_fov'/'new_fov' (old/new target's retinal FOV bearing-rate, prog/reg).

    One square panel: every switch event's fly position in lab (allocentric)
    coordinates with a short heading arrow. Coordinates are world/y-up with positive
    labels. One figure per species. Coordinates in mm when available, else px.
    """
    prep = _prep_switch_fly_poses(df, species, use_mm)
    if prep is None:
        print(f"[warn] no switch fly poses for species={species}")
        return None, None
    poses, unit, target_colors, geom = prep
    coloring = _pose_coloring(poses, color_by, base_color=putil.species_color(species))

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    _draw_switch_fly_poses(ax, poses, coloring, target_colors, geom, unit)
    _pose_colorbar(fig, coloring, ax)
    ax.set_title(f'{species or "all species"} — fly pose at switch (lab frame, n={len(poses)})\n'
                 f'colored by {coloring["title"]}', fontsize=10)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name)
    return fig, ax


def plot_switch_fly_pose_by_case(df, species=None, use_mm=True, theta_thresh=30.0,
                                 assay_type_dir=None, save_name=None):
    """Fly lab-frame pose at the switch, gridded by old-target case, colored by new target.

    A 2x2 grid of panels -- rows = old target's |theta error| (< vs > `theta_thresh`
    deg), cols = old target's retinal FOV motion (progressive vs regressive) at the
    switch. Each panel draws every switch event's fly position in lab (allocentric)
    coordinates with a short heading arrow, colored by which dot the fly switched TO
    (inner vs outer). Coordinates are world/y-up (up is up) with positive labels --
    image y is reflected within its range, keeping the mm magnitudes. All panels
    share one square (equal-span, centered) window. One figure per species.
    Coordinates in mm when available, else px.
    """
    prep = _prep_switch_fly_poses(df, species, use_mm)
    if prep is None:
        print(f"[warn] no switch fly poses for species={species}")
        return None, None
    poses, unit, target_colors, geom = prep
    coloring = _pose_coloring(poses, 'target')

    row_specs = [(f'old |θ| < {theta_thresh:g}°', poses['old_abs_theta_deg'] < theta_thresh),
                 (f'old |θ| > {theta_thresh:g}°', poses['old_abs_theta_deg'] >= theta_thresh)]
    col_specs = [('old progressive', poses['old_motion_signed'] > 0),
                 ('old regressive', poses['old_motion_signed'] < 0)]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), squeeze=False, sharex=True, sharey=True)
    for i, (rlab, rmask) in enumerate(row_specs):
        for j, (clab, cmask) in enumerate(col_specs):
            ax = axes[i][j]
            cell = poses[rmask & cmask]
            _draw_switch_fly_poses(ax, cell, coloring, target_colors, geom, unit,
                                   xlabel=(i == 1), ylabel=(j == 0))
            ax.set_title(f'{rlab}, {clab} (n={len(cell)})', fontsize=9)
    fig.suptitle(f'{species or "all species"} — fly pose at switch (lab frame), '
                 f'colored by new target (inner={target_colors["inner"]}, '
                 f'outer={target_colors["outer"]})', fontsize=12)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name)
    return fig, axes


def plot_switch_fly_pose_by_newdot(df, species=None, use_mm=True, color_by='new_fov',
                                   assay_type_dir=None, save_name=None):
    """Fly lab-frame pose at the switch, one panel per destination dot (switched TO).

    Two side-by-side panels -- switches TO the inner dot vs TO the outer dot -- each
    drawing every switch event's fly position in lab (allocentric) coordinates with a
    short heading arrow, colored by `color_by` (default 'new_fov', the new target's
    retinal FOV motion; see POSE_COLOR_MODES). The coloring scale and the square
    world/y-up window are shared across both panels so they're directly comparable.
    One figure per species. Coordinates in mm when available, else px.
    """
    prep = _prep_switch_fly_poses(df, species, use_mm)
    if prep is None:
        print(f"[warn] no switch fly poses for species={species}")
        return None, None
    poses, unit, target_colors, geom = prep
    if 'switch_target' not in poses.columns:
        print(f"[warn] no switch_target column; skipping by-newdot pose for {species}")
        return None, None
    # shared coloring (norm/cmap) computed over all poses so the two panels match
    coloring = _pose_coloring(poses, color_by, base_color=putil.species_color(species))
    dot = poses['switch_target'].astype(str).str.lower()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), squeeze=False, sharex=True, sharey=True)
    axes = axes[0]
    for j, newdot in enumerate(('inner', 'outer')):
        ax = axes[j]
        cell = poses[dot == newdot]
        _draw_switch_fly_poses(ax, cell, coloring, target_colors, geom, unit,
                               xlabel=True, ylabel=(j == 0))
        ax.set_title(f'→ {newdot} dot (n={len(cell)})', fontsize=10,
                     color=target_colors[newdot])
    _pose_colorbar(fig, coloring, list(axes))
    fig.suptitle(f'{species or "all species"} — fly pose at switch by destination dot '
                 f'(lab frame), colored by {coloring["title"]}', fontsize=12)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name)
    return fig, axes


def plot_switch_fly_pose_by_intensity(df, species=None, use_mm=True, color_by='target',
                                      panels_per_row=5, assay_type_dir=None, save_name=None):
    """Fly lab-frame pose at the switch, one panel per LED intensity.

    Like plot_switch_fly_pose but split into a panel per LED intensity (the arousal
    level). color_by selects the coloring (see POSE_COLOR_MODES). Same world/y-up
    coords and shared square window; one figure per species. mm when available, else px.
    """
    prep = _prep_switch_fly_poses(df, species, use_mm)
    if prep is None:
        print(f"[warn] no switch fly poses for species={species}")
        return None, None
    poses, unit, target_colors, geom = prep
    coloring = _pose_coloring(poses, color_by, base_color=putil.species_color(species))

    intensities = sorted(poses['led_intensity'].dropna().unique())
    if not intensities:
        print(f"[warn] no LED intensities for species={species}")
        return None, None
    ncols = min(panels_per_row, len(intensities))
    nrows = int(np.ceil(len(intensities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    flat = axes.ravel()
    for ax in flat:
        ax.set_visible(False)
    for k, inten in enumerate(intensities):
        ax = flat[k]
        ax.set_visible(True)
        i, j = divmod(k, ncols)
        cell = poses[poses['led_intensity'] == inten]
        _draw_switch_fly_poses(ax, cell, coloring, target_colors, geom, unit,
                               xlabel=(i == nrows - 1), ylabel=(j == 0))
        ax.set_title(f'{inten:g}% (n={len(cell)})', fontsize=9)
    _pose_colorbar(fig, coloring, [flat[k] for k in range(len(intensities))])
    fig.suptitle(f'{species or "all species"} — fly pose at switch by LED intensity '
                 f'(lab frame), colored by {coloring["title"]}', fontsize=12)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name)
    return fig, axes


# ── peri-switch time-course + sampled trajectories ────────────────────────────

_TIMECOURSE_VARS = {
    'theta_error': ('new_absangle_deg', 'old_absangle_deg', '|θ error to dot| (deg)'),
    'headdist': ('new_headdist_mm', 'old_headdist_mm', 'head-to-dot distance (mm)'),
    'dist': ('new_dist_mm', 'old_dist_mm', 'centroid-to-dot distance (mm)'),
}


def plot_switch_peri_timecourse(df, yvar='theta_error', window_sec=4.0, groupby=('species',),
                                show_events=True, assay_type_dir=None, save_name=None):
    """Peri-switch time-course of a per-frame metric for the new vs old target dot,
    aligned at the switch (t=0; a few seconds either side), one panel per group.

    yvar -- 'theta_error' (|θ error to dot|, deg), 'headdist' or 'dist' (distance to
    dot, mm).
    Draws faint individual switch events plus the mean +/- SEM across events for the
    new target (blue) and old target (red). Distances are thorax-to-dot-center.
    """
    if yvar not in _TIMECOURSE_VARS:
        raise ValueError(f"yvar must be one of {list(_TIMECOURSE_VARS)}")
    ncol, ocol, ylabel = _TIMECOURSE_VARS[yvar]
    traj, _ = asw.switch_target_trajectories(df, window_sec=window_sec, groupby=groupby)
    if not traj:
        print('[warn] no switch trajectories to plot')
        return None, None

    labels = sorted(traj)
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4),
                             squeeze=False, sharex=True, sharey=True)
    for j, label in enumerate(labels):
        ax = axes[0][j]
        g = traj[label]
        # new target in the species/assay color, old target in grey
        new_c, old_c = putil.courtship_color(label), putil.OLD_TARGET_COLOR
        n_ev = g.drop_duplicates(['acquisition', 'switch_frame']).shape[0]
        for col, color, name in ((ncol, new_c, 'new target'), (ocol, old_c, 'old target')):
            if show_events:
                for _, ev in g.groupby(['acquisition', 'switch_frame']):
                    ev = ev.sort_values('t_rel')
                    ax.plot(ev['t_rel'], ev[col], color=color, alpha=0.12, lw=0.6, zorder=1)
            agg = (g.groupby('frame_offset')
                     .agg(t=('t_rel', 'first'), m=(col, 'mean'), se=(col, 'sem'))
                     .sort_values('t'))
            ax.fill_between(agg['t'], agg['m'] - agg['se'], agg['m'] + agg['se'],
                            color=color, alpha=0.25, lw=0, zorder=2)
            ax.plot(agg['t'], agg['m'], color=color, lw=2, label=name, zorder=3)
        ax.axvline(0, color='white', ls='--', lw=0.8, alpha=0.5)
        ax.set_title(f'{label} (n={n_ev})')
        ax.set_xlabel('time from switch (s)')
        if j == 0:
            ax.set_ylabel(ylabel)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f'peri-switch {ylabel}: new vs old target', fontsize=12)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name or f'switch_timecourse_{yvar}.png')
    return fig, axes


def plot_switch_trajectory_samples(df, groupby=('species',), n_samples=12, window_sec=2.0,
                                   paired=True, vlim_percentile=90,
                                   assay_type_dir=None, save_prefix='switch_trajectory_samples'):
    """Triad-style sampled switch trajectories, one panel per event, in BOTH the
    egocentric (focal-centered, range-ring) and allocentric (lab) frames.

    For each group (default per species) up to `n_samples` switch events are sampled
    across assays and laid out as a panel grid, drawn as time-gradient tracks with
    the new target (tomato) and old target (gray) -- plus, in the allo frame, the
    focal fly (blue) with a heading arrow. Reuses the triad renderer
    (analyses.triad.src.putil.switch_plots) on the projector's switch trajectories,
    so the look matches generate_switch_plots._plot_switch_target_trajectory_samples.
    One figure per group per frame is written; ego files plain, allo suffixed '_allo'.

    paired -- also write a partner figure ('..._paired') where each event's
    trajectory sits beside a copy with the targets colored by the retinal FOV
    bearing-rate (target_ang_vel_fov; magenta = regressive, green = progressive),
    with a shared colorbar -- the same sanity layout as the triad pairs plot.
    vlim_percentile -- |metric| percentile the paired color scale spans (default 90).
    """
    traj, ppm_dict = asw.switch_target_trajectories(df, window_sec=window_sec, groupby=groupby)
    if not traj:
        print('[warn] no switch trajectories to sample')
        return
    from analyses.triad.src.putil import switch_plots as tsp
    fps = float(df['fps'].iloc[0]) if 'fps' in df.columns else 60.0
    half = window_sec / 2
    for label, tdf in traj.items():
        ppm = ppm_dict.get(label, 1.0)
        # new target in the species/assay color, old target grey, focal male dark purple
        new_color = putil.courtship_color(label, default='tomato')
        for frame, suffix in (('ego', ''), ('allo', '_allo')):
            fig, _ = tsp.plot_switch_target_trajectory_samples_for_assay(
                tdf, ppm, label, frame=frame, n_samples=n_samples, fps=fps,
                new_color=new_color, old_color=putil.OLD_TARGET_COLOR,
                focal_color=putil.FOCAL_COLOR, half_window_sec=half)
            if fig is not None:
                _save(fig, assay_type_dir, f'{save_prefix}_{label}{suffix}.png')

            # partner panels colored by retinal FOV bearing-rate (prog/regressive)
            if paired:
                fig, _ = tsp.plot_switch_target_trajectory_pairs_for_assay(
                    tdf, ppm, label, frame=frame, n_samples=n_samples, fps=fps,
                    new_color=new_color, old_color=putil.OLD_TARGET_COLOR,
                    focal_color=putil.FOCAL_COLOR, metric_focal_color=putil.FOCAL_COLOR,
                    cmap=putil.prog_regr_cmap(), half_window_sec=half,
                    vlim_percentile=vlim_percentile)
                if fig is not None:
                    _save(fig, assay_type_dir, f'{save_prefix}_{label}{suffix}_paired.png')