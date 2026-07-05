"""Plots for the projector dot-assay outer-dot engagement analysis."""

import os
import numpy as np
import matplotlib.pyplot as plt

from .. import analyze_outerdot as ao
from .. import analyze_switches as asw
from .. import led_metadata as lm

# Ruta-lab courtship color scheme + dark (#262626) theme (libs.plotting); the
# shared palette/helpers are a hard dependency for the colors used throughout.
import libs.plotting as putil
putil.set_sns_style(style='courtship', min_fontsize=11)


def _hue_label(hue, v):
    """Legend label for one hue value."""
    if hue == 'stim_speed':
        return f'{v:g} mm/s'
    return str(v)


def _hue_colors(values, base):
    """Map ordered hue values (LED intensity, speed, direction) to shades along
    `base` -- a species/assay color -- light=low to dark=high."""
    grad = putil.color_gradient(base, len(values))
    return {v: grad[i] for i, v in enumerate(values)}


def plot_fraction_vs_intensity(df, behaviors=ao.DEFAULT_BEHAVIORS,
                               assay_type_dir=None, save_name='fraction_vs_intensity.png',
                               ylabel='fraction of frames', hue=None, suptitle=None):
    """Grid of fraction-engaged vs LED intensity: rows = behaviors, cols = species.

    Each species gets its own column (and its own intensity axis), because Dmel and
    Dyak were recorded at different LED-intensity ranges. The y-axis is shared so
    behavior levels are comparable across species. Mean (+/- SEM across assays) is
    drawn as a line with a shaded band. If assay_type_dir is given the figure is
    written to its figures/ dir.

    hue -- optional column to split each panel into one colored line per value
    (e.g. 'stim_speed' from build_assays --stim-speed, or 'target_direction' from
    --target-direction).
    """
    summary = ao.fraction_chasing_orienting(df, behaviors, hue=hue)
    per = ao.fraction_per_assay(df, behaviors, hue=hue)
    behaviors = [b for b in behaviors if b in df.columns]
    species = sorted(summary['species'].dropna().unique())

    species_colors = {sp: putil.species_color(sp) for sp in species}
    hue_vals = sorted(summary[hue].dropna().unique()) if hue else []
    # one gradient of shades per species (along that species' color) for the hue
    hue_colors = {sp: _hue_colors(hue_vals, species_colors[sp]) for sp in species} if hue else {}

    # n = number of assays contributing per species (for the column titles)
    n_by_species = per.groupby('species')['assay'].nunique().to_dict() if not per.empty else {}

    def _mean_sem(ax, s, color, label=None):
        """Mean line with +/- SEM shaded band (SEM=0 where a single assay)."""
        s = s.sort_values('led_intensity')
        sem = s['sem'].fillna(0.0)
        ax.fill_between(s['led_intensity'], s['mean'] - sem, s['mean'] + sem,
                        color=color, alpha=0.22, lw=0, zorder=1)
        ax.plot(s['led_intensity'], s['mean'], marker='o', color=color, label=label, zorder=2)

    nrows, ncols = len(behaviors), len(species)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False, sharey=True)

    for i, b in enumerate(behaviors):
        for j, sp in enumerate(species):
            ax = axes[i][j]
            sub = summary[(summary['behavior'] == b) & (summary['species'] == sp)]
            if hue:
                for v in hue_vals:
                    s = sub[sub[hue] == v]
                    if not s.empty:
                        _mean_sem(ax, s, hue_colors[sp][v], label=_hue_label(hue, v))
            else:
                _mean_sem(ax, sub, species_colors[sp])
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(f'{sp} (n={n_by_species.get(sp, 0)})')
            if i == nrows - 1:
                ax.set_xlabel('LED intensity (%)')
            if j == 0:
                ax.set_ylabel(f'{b}\n{ylabel}')
    if hue and hue_vals:
        # one legend per species column (top row); hue colors are species-specific so each
        # panel's legend reflects that species' own shades
        for j in range(ncols):
            axes[0][j].legend(title=hue, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()

    if assay_type_dir:
        fig_dir = os.path.join(assay_type_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, save_name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"saved {path}")
    return fig, axes


# fly + dot path colors (legible on the dark #262626 background)
TRACK_COLORS = {'fly': 'white', 'innerdot': '#E8830C', 'outerdot': '#2CA02C'}


def plot_assay_trajectories(df, species=None, direction=None, use_mm=True,
                            chasing_only=False, bridge_gap_sec=1.0,
                            assay_type_dir=None, save_name=None):
    """Grid of fly + dot trajectories: rows = assays, cols = LED intensity.

    Plots one figure for a single species (and optionally a single stimulus
    direction). Each cell draws the fly path plus each dot's path over the frames
    of that assay at that LED intensity. Axes share scale (fixed arena) and use
    world/y-up orientation (image y is negated, so a vertical flip of the raw
    camera view) -- the stimulus reads as commanded (ccw) and matches the switch
    ego-plots. Coordinates in mm when available, else px.

    direction -- if given (e.g. 'same'/'opposite'), filter to that target_direction
    (requires the column from build_assays --target-direction).
    chasing_only -- if True, restrict to a courtship proxy: frames where the fly is
    chasing either dot (chasing_innerdot OR chasing_outerdot), with chasing bouts
    less than `bridge_gap_sec` apart greedily bridged into one continuous segment.
    TEMP: stands in for a courtship/orienting classifier that does not exist yet.
    """
    sub = df
    if species is not None:
        sub = sub[sub['species'] == species]
    if direction is not None:
        if 'target_direction' not in sub.columns:
            raise KeyError("direction filter needs a 'target_direction' column "
                           "(build with build_assays --target-direction)")
        sub = sub[sub['target_direction'] == direction]
    sub = sub[lm.valid_led(sub)]
    if chasing_only:
        chase_cols = [c for c in ('chasing_innerdot', 'chasing_outerdot')
                      if c in sub.columns]
        if not chase_cols:
            print("[warn] chasing_only requested but no chasing_innerdot/"
                  "chasing_outerdot columns; skipping")
            return None, None
        # TEMP courtship proxy: chasing-either-dot with sub-`bridge_gap_sec` gaps
        # between bouts greedily filled, per assay (no orienting classifier yet).
        keep = []
        for _, g in sub.groupby('assay'):
            g = g.sort_values('frame')
            m = (g[chase_cols].astype(float).fillna(0) > 0).any(axis=1).to_numpy()
            fps = float(g['fps'].iloc[0]) if 'fps' in g.columns else 60.0
            m = ao._bridge_bouts(m, int(round(fps * bridge_gap_sec)))
            keep.extend(g.index[m].tolist())
        sub = sub.loc[keep]
    if sub.empty:
        print(f"[warn] no frames for species={species} direction={direction}"
              f"{' (chasing-only)' if chasing_only else ''}")
        return None, None

    assays = sorted(sub['assay'].unique())
    intensities = sorted(sub['led_intensity'].unique())
    suffix = '_mm' if (use_mm and 'fly_x_mm' in sub.columns) else ''
    unit = 'mm' if suffix else 'px'
    tracks = [t for t in ('fly', 'innerdot', 'outerdot') if f'{t}_x{suffix}' in sub.columns]

    nrows, ncols = len(assays), len(intensities)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    for i, assay in enumerate(assays):
        for j, inten in enumerate(intensities):
            ax = axes[i][j]
            cell = sub[(sub['assay'] == assay) & (sub['led_intensity'] == inten)]
            cell = cell.sort_values('frame')
            # break the line wherever frames aren't consecutive (e.g. gaps
            # between chasing bouts) by inserting NaNs at the discontinuities
            gaps = cell['frame'].to_numpy()
            breaks = np.nonzero(np.diff(gaps) != 1)[0] + 1 if len(gaps) else []
            for t in tracks:
                x = np.insert(cell[f'{t}_x{suffix}'].to_numpy().astype(float), breaks, np.nan)
                # negate y: image coords are y-down; plot in world/y-up space so the
                # stimulus rotation reads as commanded (ccw) and matches the switch
                # ego-plots. (This is a vertical flip of the raw camera view.)
                y = -np.insert(cell[f'{t}_y{suffix}'].to_numpy().astype(float), breaks, np.nan)
                ax.plot(x, y, color=TRACK_COLORS[t], lw=0.5, alpha=0.8)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f'{inten:g}%', fontsize=10)
            if j == 0:
                ax.set_ylabel(assay, fontsize=6, rotation=0, ha='right', va='center')

    title = species or 'all species'
    if direction is not None:
        title += f' — {direction} direction'
    if chasing_only:
        title += f' — courting [TEMP: chasing, ≤{bridge_gap_sec:g}s gaps bridged]'
    fig.suptitle(f'{title}  (fly={TRACK_COLORS["fly"]}, '
                 f'inner={TRACK_COLORS["innerdot"]}, outer={TRACK_COLORS["outerdot"]}; {unit})',
                 fontsize=11)
    fig.tight_layout()

    if assay_type_dir and save_name:
        fig_dir = os.path.join(assay_type_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, save_name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"saved {path}")
    return fig, axes


def _save(fig, assay_type_dir, save_name):
    if assay_type_dir and save_name:
        fig_dir = os.path.join(assay_type_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, save_name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"saved {path}")


# ── pursuit-parameter distributions (by species × arousal/LED) ──────────────
# Per-dot metrics are read "to the chased dot": for frames chasing the inner dot we
# take the *_innerdot value, for the outer dot the *_outerdot value. Distance prefers
# a head-based feature (headdist_to_*_mm) when present, else centroid distance.
# A metric with a 'global' key (e.g. fly speed) is not dot-specific: the same column
# is used regardless of dot, and 'chasing' = frames chasing *either* dot.
# 'scale' multiplies the raw column values (e.g. rad -> deg); defaults to 1.
# 'name' is the display name used in titles (defaults to the metric key).
# Optional flags applied (in this order) to 'global' columns, since the raw JAABA
# perframe features for the fly are per-*frame* in pixel/radian units:
#   'abs'           -- take |value| (magnitude, e.g. turning rate regardless of sign)
#   'scale'         -- constant multiplier (e.g. rad -> deg)
#   'rate_per_frame'-- multiply by fps (per-frame -> per-second)
#   'px_to_mm'      -- divide by px_per_mm (pixels -> mm)
# (verified empirically: lin_speed == px/frame, ang_vel == rad/frame.)
PURSUIT_METRICS = {
    'theta_error': {'tmpl': ['absangle_to_{dot}'], 'absfallback': 'angle_to_{dot}',
                    'scale': 180.0 / np.pi, 'name': 'θ error',
                    'label': '|θ error to chased dot| (deg)'},
    'dist':  {'tmpl': ['headdist_to_{dot}_mm', 'dist_to_{dot}_mm'],
              'label': 'distance to chased dot (mm)'},
    # lin_speed is px/frame -> mm/s via ×fps then ÷px_per_mm
    'speed': {'global': ['lin_speed'], 'rate_per_frame': True, 'px_to_mm': True,
              'name': 'speed', 'label': 'fly speed (mm/s)'},
    # ang_vel is rad/frame -> deg/s via ×fps then ×180/π; abs = turning magnitude
    'ang_vel': {'global': ['ang_vel'], 'abs': True, 'rate_per_frame': True,
                'scale': 180.0 / np.pi, 'name': 'angular velocity',
                'label': '|angular velocity| (deg/s)'},
}
_DOTS = ('innerdot', 'outerdot')


def _resolve_dot_metric(df, dot, spec):
    """Series of the metric for one dot (abs of a signed fallback if needed), or None."""
    for col in (t.format(dot=dot) for t in spec['tmpl']):
        if col in df.columns:
            return df[col], col
    fb = spec.get('absfallback')
    if fb:
        col = fb.format(dot=dot)
        if col in df.columns:
            return df[col].abs(), col + ' (abs)'
    return None, None


def _chasing_any_mask(df):
    """Boolean mask of frames chasing either dot, or None if no chasing columns."""
    cols = [c for c in ('chasing_innerdot', 'chasing_outerdot') if c in df.columns]
    if not cols:
        return None
    return (df[cols].astype(float).fillna(0) > 0).any(axis=1).to_numpy()


def build_pursuit_long(df, metric):
    """Long-form rows [species, led_intensity, dot, subset, value] for one metric.

    subset is 'all' (every frame) or 'chasing'. For per-dot metrics, inner- and
    outer-dot rows are pooled, each frame contributing the value *to that dot*, and
    'chasing' keeps frames where chasing_<dot>==1. For a 'global' metric (e.g. fly
    speed) the single column is used and 'chasing' = frames chasing either dot.
    Rows with NaN value/intensity/species are dropped.
    """
    import pandas as pd
    df = df[lm.valid_led(df)]            # gate LED intensity (incl. per-species cap)
    spec = PURSUIT_METRICS[metric]
    scale = spec.get('scale', 1.0)
    used, parts = {}, []

    if 'global' in spec:
        col = next((c for c in spec['global'] if c in df.columns), None)
        if col is None:
            print(f"[warn] none of {spec['global']} present; '{metric}' skipped")
            return None
        used['fly'] = col
        base = df[['species', 'led_intensity', 'assay']].copy()
        vals = df[col].to_numpy()
        if spec.get('abs'):
            vals = np.abs(vals)
        vals = vals * scale
        if spec.get('rate_per_frame'):                 # per-frame -> per-second
            vals = vals * df['fps'].to_numpy()
        if spec.get('px_to_mm'):                        # pixels -> mm
            vals = vals / df['px_per_mm'].to_numpy()
        base['value'] = vals
        base['dot'] = 'fly'
        base['subset'] = 'all'
        parts.append(base)
        mask = _chasing_any_mask(df)
        if mask is None:
            print("[warn] no chasing columns; only the 'all' subset for speed")
        else:
            ch = base[mask].copy()
            ch['subset'] = 'chasing'
            parts.append(ch)
    else:
        for dot in _DOTS:
            vals, src = _resolve_dot_metric(df, dot, spec)
            if vals is None:
                continue
            used[dot] = src
            base = df[['species', 'led_intensity', 'assay']].copy()
            base['value'] = vals.to_numpy() * scale
            base['dot'] = dot
            base['subset'] = 'all'
            parts.append(base)
            chase = f'chasing_{dot}'
            if chase in df.columns:
                ch = base[df[chase].to_numpy() == 1].copy()
                ch['subset'] = 'chasing'
                parts.append(ch)
            else:
                print(f"[warn] no '{chase}' column; '{dot}' contributes only the 'all' subset")

    if not parts:
        print(f"[warn] none of the {metric} columns present; nothing to plot")
        return None
    print(f"  {metric}: using {used}")
    long = pd.concat(parts, ignore_index=True)
    return long.dropna(subset=['value', 'led_intensity', 'species'])


def _pursuit_dot_phrase(metric):
    """'to chased dot' for per-dot metrics, '' for global ones (used in titles)."""
    return '' if 'global' in PURSUIT_METRICS[metric] else ' to chased dot'


def _pursuit_name(metric):
    """Display name for a metric in titles (defaults to the metric key)."""
    return PURSUIT_METRICS[metric].get('name', metric)


def plot_pursuit_distributions(df, metric='dist', species=None, bins=50,
                               violin=False, xlim_percentile=99,
                               assay_type_dir=None, save_name=None):
    """Per-species distribution of a pursuit metric across LED intensities.

    histograms (default): one panel per LED intensity (the arousal level), in each
    the 'chasing' distribution overlaid on 'all frames' as a sanity check.
    violin=True: one violin per LED intensity over *courtship (chasing) frames only*,
    colored by intensity -- so the metric's shift with arousal reads at a glance.
    """
    long = build_pursuit_long(df, metric)
    if long is None or long.empty:
        return None, None
    if species is not None:
        long = long[long['species'] == species]
    if long.empty:
        print(f"[warn] no {metric} data for species={species}")
        return None, None

    label = PURSUIT_METRICS[metric]['label']
    subsets = [s for s in ('all', 'chasing') if s in long['subset'].unique()]
    title_sp = species or 'all species'
    base = putil.species_color(species)                    # species color (grey if pooled)
    # chasing in the species color, 'all frames' baseline in grey
    subset_colors = {'all': putil.NEUTRAL_GREY, 'chasing': base}

    if violin:
        import seaborn as sns
        chasing = long[long['subset'] == 'chasing']
        if chasing.empty:
            print(f"[warn] no chasing frames for {metric}, species={species}; skipping violin")
            return None, None
        order = sorted(chasing['led_intensity'].unique())
        colors = _hue_colors(order, base)                  # shades along the species color
        fig, ax = plt.subplots(figsize=(1.4 * len(order) + 2, 4))
        sns.violinplot(data=chasing, x='led_intensity', y='value', order=order,
                       hue='led_intensity', hue_order=order, legend=False,
                       palette=colors, density_norm='width', cut=0, inner='quartile', ax=ax)
        ax.set_xlabel('LED intensity (%)')
        ax.set_ylabel(label)
        ax.set_title(f'{title_sp} — courtship-frame {_pursuit_name(metric)}{_pursuit_dot_phrase(metric)}')
        fig.tight_layout()
        _save(fig, assay_type_dir, save_name)
        return fig, ax

    intensities = sorted(long['led_intensity'].unique())
    ncols = len(intensities)
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 3.2),
                             squeeze=False, sharex=True, sharey=True)
    axes = axes[0]
    for j, inten in enumerate(intensities):
        ax = axes[j]
        panel = long[long['led_intensity'] == inten]
        # shared bin edges across subsets (clipped at percentile to tame tails)
        pv = panel['value']
        x_max = np.percentile(pv, xlim_percentile) if xlim_percentile else pv.max()
        edges = np.linspace(float(pv.min()), float(x_max), bins + 1)
        for s in subsets:
            v = panel[panel['subset'] == s]['value']
            if v.empty:
                continue
            ax.hist(v, bins=edges, density=True, histtype='stepfilled',
                    color=subset_colors[s], alpha=0.45, edgecolor=subset_colors[s],
                    lw=1.0, label=f'{s} (n={len(v)})')
        ax.set_title(f'{inten:g}%', fontsize=10)
        ax.set_xlabel(label)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('density')
    fig.suptitle(f'{title_sp} — {_pursuit_name(metric)}{_pursuit_dot_phrase(metric)} (chasing vs all frames)',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name)
    return fig, axes


def plot_pursuit_by_intensity(df, metric='dist', species=None, bins=100, kde=False,
                              cdf=False, xlim_percentile=99,
                              assay_type_dir=None, save_name=None):
    """Per species: chasing-frame distributions at each LED intensity overlaid in one axes.

    One distribution per LED intensity (with a faint matching fill), colored along a
    gradient of the species color (light = low intensity → dark = high), so the shift
    in the pursuit metric with arousal reads at a glance. Only chasing frames are used;
    the x-range is shared across intensities so the curves are directly comparable.

    kde -- if True, draw a smooth gaussian-KDE density line per intensity instead of
    step histograms (cleaner when low-n intensities make the bins noisy).
    cdf -- if True, draw the empirical cumulative distribution (ECDF) per intensity as
    a plain outline (no fill); y is cumulative density 0→1. A rightward shift of a
    curve means larger values at that intensity, and curve crossings show where the
    distributions overlap. Takes precedence over kde.
    """
    long = build_pursuit_long(df, metric)
    if long is None or long.empty:
        return None, None
    long = long[long['subset'] == 'chasing']
    if species is not None:
        long = long[long['species'] == species]
    if long.empty:
        print(f"[warn] no chasing {metric} data for species={species}")
        return None, None

    label = PURSUIT_METRICS[metric]['label']
    title_sp = species or 'all species'
    intensities = sorted(long['led_intensity'].unique())
    colors = _hue_colors(intensities, putil.species_color(species))   # shades of species color

    pv = long['value']
    x_min = float(pv.min())
    x_max = float(np.percentile(pv, xlim_percentile) if xlim_percentile else pv.max())
    edges = np.linspace(x_min, x_max, bins + 1)
    grid = np.linspace(x_min, x_max, 256)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if cdf:
        # ECDF evaluated on a grid spanning the full data range, so every curve
        # reaches 1; the view is clipped to the shared [x_min, x_max] window.
        cdf_grid = np.linspace(x_min, float(pv.max()), 512)
        for inten in intensities:
            vv = long[long['led_intensity'] == inten]['value'].to_numpy()
            vv = np.sort(vv[np.isfinite(vv)])
            if vv.size == 0:
                continue
            y = np.searchsorted(vv, cdf_grid, side='right') / vv.size
            ax.plot(cdf_grid, y, color=colors[inten], lw=1.6,
                    label=f'{inten:g}% (n={vv.size})')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel(label)
        ax.set_ylabel('cumulative density')
        ax.set_title(f'{title_sp} — chasing-frame {_pursuit_name(metric)}'
                     f'{_pursuit_dot_phrase(metric)} ECDF by LED intensity')
        ax.legend(title='LED intensity', fontsize=8)
        fig.tight_layout()
        _save(fig, assay_type_dir, save_name)
        return fig, ax

    for inten in intensities:
        v = long[long['led_intensity'] == inten]['value']
        c = colors[inten]
        if kde:
            from scipy.stats import gaussian_kde
            vv = v.to_numpy()
            vv = vv[np.isfinite(vv)]
            if vv.size < 2 or np.ptp(vv) == 0:
                continue
            dens = gaussian_kde(vv)(grid)
            ax.fill_between(grid, dens, color=c, alpha=0.10, lw=0)
            ax.plot(grid, dens, color=c, lw=1.8, label=f'{inten:g}% (n={vv.size})')
        else:
            if v.empty:
                continue
            # faint fill under a crisp outline (two passes so the lines stay readable)
            ax.hist(v, bins=edges, density=True, histtype='stepfilled', color=c, alpha=0.12, lw=0)
            ax.hist(v, bins=edges, density=True, histtype='step', color=c,
                    lw=1.4, label=f'{inten:g}% (n={len(v)})')
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(label)
    ax.set_ylabel('density')
    ax.set_title(f'{title_sp} — chasing-frame {_pursuit_name(metric)}{_pursuit_dot_phrase(metric)} by LED intensity')
    ax.legend(title='LED intensity', fontsize=8)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name)
    return fig, ax


def plot_pursuit_per_acquisition(df, metric='dist', agg='median', subset='chasing',
                                 assay_type_dir=None, save_name=None):
    """Per-acquisition summary of a pursuit metric: one panel per species, x = LED
    intensity. Each acquisition contributes one point (its `agg` over the `subset`
    frames at that intensity), and a horizontal line marks the across-acquisition
    `agg` per intensity (with a thin SEM whisker). Points/lines are shaded along the
    species color by intensity. `agg` is applied at *both* levels (within an
    acquisition and across acquisitions), so agg='median' is robust to per-frame and
    per-acquisition outliers.

    Makes the acquisition the unit of replication so species × arousal can be
    compared as points, complementing the pooled per-frame distributions.
    `subset` is 'chasing' (courtship frames, default) or 'all'; `agg` is
    'median' (default) or 'mean'.
    """
    long = build_pursuit_long(df, metric)
    if long is None or long.empty:
        return None, None
    long = long[long['subset'] == subset]
    if long.empty:
        print(f"[warn] no {subset} {metric} data for per-acquisition plot")
        return None, None

    aggfn = 'median' if agg == 'median' else 'mean'
    central = np.nanmedian if agg == 'median' else np.nanmean   # across-acquisition line
    per_acq = (long.groupby(['species', 'led_intensity', 'assay'])['value']
                   .agg(aggfn).reset_index())

    label = PURSUIT_METRICS[metric]['label']
    species = sorted(per_acq['species'].unique())
    # each species shows only the intensities it actually has data at (Dmel and Dyak
    # were run over different LED ranges), so empty intensity slots aren't drawn
    sp_intensities = {sp: sorted(per_acq[per_acq['species'] == sp]['led_intensity'].unique())
                      for sp in species}
    max_n = max((len(v) for v in sp_intensities.values()), default=1)

    fig, axes = plt.subplots(
        1, len(species), squeeze=False, sharey=True,
        figsize=(max(1.5, 0.6 * max_n) * len(species), 4))
    axes = axes[0]
    rng = np.random.default_rng(0)
    for ax, sp in zip(axes, species):
        intensities = sp_intensities[sp]
        xpos = {inten: i for i, inten in enumerate(intensities)}
        colors = _hue_colors(intensities, putil.species_color(sp))   # shade by intensity
        gs = per_acq[per_acq['species'] == sp]
        for inten in intensities:
            i = xpos[inten]
            vals = gs[gs['led_intensity'] == inten]['value'].to_numpy()
            if not len(vals):
                continue
            c = colors[inten]
            jit = (rng.random(len(vals)) - 0.5) * 0.3
            ax.scatter(i + jit, vals, s=40, color=c, alpha=0.85,
                       edgecolor='black', linewidths=0.4, zorder=3)
            m = float(central(vals))
            ax.plot([i - 0.33, i + 0.33], [m, m], color=c, lw=3, zorder=4)   # central line
            if len(vals) > 1:
                se = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
                ax.errorbar(i, m, yerr=se, color=c, capsize=4, lw=1.2, zorder=4)
        ax.set_xticks(range(len(intensities)))
        ax.set_xticklabels([f'{inten:g}' for inten in intensities])
        ax.set_xlabel('LED intensity (%)')
        ax.set_title(sp)
        ax.margins(x=0.12)
    axes[0].set_ylabel(f'{agg} {label} / acquisition')
    fig.suptitle(f'Per-acquisition {agg} {_pursuit_name(metric)}'
                 f'{_pursuit_dot_phrase(metric)} ({subset} frames)', fontsize=12)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name or f'pursuit_{metric}_per_acquisition.png')
    return fig, axes


def plot_pursuit_summary(df, metric='dist', subset='chasing',
                         center='median', spread='iqr',
                         species=None, assay_type_dir=None, save_name=None):
    """Pooled frame-level summary of a pursuit metric: one panel per species, x = LED
    intensity, y = the `center` over ALL `subset` frames with an error band.

    A compact alternative to the per-acquisition scatter: every `subset` frame at a
    given (species, LED) is pooled and reduced to one center + spread marker, so the
    shift in the metric with arousal reads as a single line per species.

    center -- 'median' (default; robust to the right-skewed tails of these metrics)
              or 'mean'.
    spread -- the error band:
        'iqr'  -- Q1..Q3 (asymmetric; robust, pairs with median; shows the bulk spread)
        'mad'  -- median +/- MAD (symmetric robust spread)
        'sd'   -- mean +/- SD (population spread across frames)
        'sem'  -- mean +/- std/sqrt(n_frames)
        'ci95' -- mean +/- 1.96*sem
      Note frames are autocorrelated, so sem/ci95 under-state biological uncertainty
      (the acquisition is the true replicate, see plot_pursuit_per_acquisition); they
      describe the pooled frames, not the animals. n is the frame count.
    """
    import pandas as pd
    long = build_pursuit_long(df, metric)
    if long is None or long.empty:
        return None, None
    long = long[long['subset'] == subset]
    if species is not None:
        long = long[long['species'] == species]
    if long.empty:
        print(f"[warn] no {subset} {metric} data for species={species}")
        return None, None

    def _stats(v):
        v = v.to_numpy(float)
        med, mean = np.median(v), np.mean(v)
        q1, q3 = np.percentile(v, [25, 75])
        sd = np.std(v, ddof=1) if v.size > 1 else 0.0
        sem = sd / np.sqrt(v.size)
        mad = np.median(np.abs(v - med))
        c = med if center == 'median' else mean
        # (lower, upper) absolute offsets from the center for the error bar
        band = {
            'iqr':  (c - q1, q3 - c),
            'mad':  (mad, mad),
            'sd':   (sd, sd),
            'sem':  (sem, sem),
            'ci95': (1.96 * sem, 1.96 * sem),
        }[spread]
        return pd.Series({'center': c, 'lo': band[0], 'hi': band[1], 'n': v.size})

    g = (long.groupby(['species', 'led_intensity'])['value']
             .apply(_stats).unstack().reset_index())

    label = PURSUIT_METRICS[metric]['label']
    species_order = sorted(g['species'].unique())
    sp_intensities = {sp: sorted(g[g['species'] == sp]['led_intensity'].unique())
                      for sp in species_order}
    max_n = max((len(v) for v in sp_intensities.values()), default=1)

    fig, axes = plt.subplots(
        1, len(species_order), squeeze=False, sharey=True,
        figsize=(max(1.6, 0.5 * max_n) * len(species_order), 4))
    axes = axes[0]
    for ax, sp in zip(axes, species_order):
        intensities = sp_intensities[sp]
        x = list(range(len(intensities)))
        gs = g[g['species'] == sp].set_index('led_intensity').reindex(intensities)
        color = putil.species_color(sp)
        yerr = np.vstack([gs['lo'].to_numpy(float), gs['hi'].to_numpy(float)])
        ax.errorbar(x, gs['center'].to_numpy(float), yerr=yerr,
                    color=color, marker='o', ms=6, lw=1.8, capsize=4, zorder=3)
        # frame count under each point (the pool size behind each estimate)
        for xi, (_, r) in zip(x, gs.iterrows()):
            if np.isfinite(r['center']):
                ax.annotate(f"{int(r['n'])}", (xi, r['center']), textcoords='offset points',
                            xytext=(0, -12), ha='center', fontsize=6, color='0.4')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{inten:g}' for inten in intensities])
        ax.set_xlabel('LED intensity (%)')
        ax.set_title(sp)
        ax.margins(x=0.12)
    axes[0].set_ylabel(f'{center} {label}')
    spread_lbl = {'iqr': 'IQR', 'mad': '±MAD', 'sd': '±SD',
                  'sem': '±SEM', 'ci95': '±95% CI'}[spread]
    fig.suptitle(f'{_pursuit_name(metric)}{_pursuit_dot_phrase(metric)} '
                 f'— {center} ({spread_lbl}) across {subset} frames', fontsize=12)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name or f'pursuit_{metric}_summary.png')
    return fig, axes


def plot_chasing_bout_durations(df, bridge_sec=1.0, bins=40, log_x=True,
                                short_sec=0.25, xmax_percentile=99,
                                assay_type_dir=None, save_name=None):
    """Distributions of chasing-either-dot bout durations, overlaid by LED intensity,
    one panel per species. Bouts are runs of chasing either dot with gaps shorter
    than `bridge_sec` linked into one bout (see analyze_outerdot.chasing_bouts).

    The y-axis is the fraction of *chasing frames* (chasing time), not the fraction
    of bouts: each bout is weighted by its chasing-frame count, so a block whose
    chasing time is really spent in long bouts reads as such even if it also has many
    tiny bouts. Lets you see how contiguous chasing is across LED blocks and spot a
    block (e.g. 0%) whose chasing time is mostly a short-duration tail of likely false
    positives. A dashed line marks `short_sec`; the legend reports each intensity's
    bout count and the fraction of chasing frames in bouts below short_sec. `log_x`
    (default) spreads the short bouts out; bins are shared across intensities and
    species for comparability.
    """
    bouts = ao.chasing_bouts(df, bridge_sec=bridge_sec)
    if bouts.empty:
        print("[warn] no chasing bouts to plot")
        return None, None

    species = sorted(bouts['species'].unique())
    durs = bouts['dur_sec']
    lo = max(float(durs.min()), 1.0 / 240)                  # ~1 frame floor for log scale
    hi = float(np.percentile(durs, xmax_percentile))
    hi = max(hi, lo * 1.1)
    edges = (np.logspace(np.log10(lo), np.log10(hi), bins + 1) if log_x
             else np.linspace(0.0, hi, bins + 1))

    fig, axes = plt.subplots(1, len(species), figsize=(6 * len(species), 4.2),
                             squeeze=False, sharey=True)
    axes = axes[0]
    for ax, sp in zip(axes, species):
        b = bouts[bouts['species'] == sp]
        intensities = sorted(b['led_intensity'].unique())
        colors = _hue_colors(intensities, putil.species_color(sp))
        for inten in intensities:
            bsub = b[b['led_intensity'] == inten]
            d = bsub['dur_sec']
            if d.empty:
                continue
            c = colors[inten]
            # weight each bout by its chasing-frame count so the y-axis is the fraction
            # of *chasing frames* (i.e. chasing time) in bouts of each duration -- what
            # matters here, not the fraction of bouts (which over-weights tiny ones)
            cframes = (bsub['n_frames'] * bsub['frac_chasing']).to_numpy()
            w = cframes / cframes.sum()
            short_frac = cframes[(d < short_sec).to_numpy()].sum() / cframes.sum()
            ax.hist(d, bins=edges, weights=w, histtype='stepfilled', color=c, alpha=0.10, lw=0)
            ax.hist(d, bins=edges, weights=w, histtype='step', color=c, lw=1.6,
                    label=f'{inten:g}% (n={len(d)} bouts, {short_frac:.0%} of frames <{short_sec:g}s)')
        if log_x:
            ax.set_xscale('log')
        ax.axvline(short_sec, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel('chasing bout duration (s)')
        ax.set_title(sp)
        ax.legend(title='LED intensity', fontsize=7)
    axes[0].set_ylabel('fraction of chasing frames')
    fig.suptitle(f'Chasing-either-dot bout durations (gaps <{bridge_sec:g}s linked)', fontsize=13)
    fig.tight_layout()
    _save(fig, assay_type_dir, save_name or 'chasing_bout_durations.png')
    return fig, axes
