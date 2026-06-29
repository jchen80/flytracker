"""
Matplotlib diagnostics for orientation-flip events.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .detect import detect_flip_events


def plot_ori_flip_diagnostics(df, fps=60, jump_threshold_rad=2.0,
                               stability_threshold_rad=0.3,
                               fly_ids=None, acq=None,
                               max_flies=3, figsize_per_fly=(16, 4)):
    """
    For each fly, plot orientation timeseries with flip events highlighted,
    overlaid with minimum distance to any other fly.

    Arguments:
        df  -- single-acquisition tracks df

    Keyword Arguments:
        fps                     -- frames per second (default: 60)
        jump_threshold_rad      -- threshold for jump detection (default: 2.0)
        stability_threshold_rad -- max std of d(ori)/dt in the flipped segment (default: 0.3)
        fly_ids                 -- list of fly ids to plot; if None, plots all (up to max_flies)
        acq                     -- acquisition name for title (default: None)
        max_flies               -- max number of flies to plot (default: 3)
        figsize_per_fly         -- (width, height) per fly panel (default: (16, 4))

    Returns:
        fig
    """
    events = detect_flip_events(df, fps=fps, jump_threshold_rad=jump_threshold_rad,
                                stability_threshold_rad=stability_threshold_rad)

    all_fly_ids = sorted(df['id'].unique())
    if fly_ids is None:
        fly_ids = all_fly_ids[:max_flies]

    n = len(fly_ids)
    fig, axes = plt.subplots(n, 1,
                             figsize=(figsize_per_fly[0], figsize_per_fly[1] * n),
                             sharex=False)
    if n == 1:
        axes = [axes]

    for ax, fly_id in zip(axes, fly_ids):
        fly_df = (df[df['id'] == fly_id]
                    .drop_duplicates('frame')
                    .sort_values('frame'))
        t = fly_df['frame'].values / fps
        ori_deg = np.rad2deg(fly_df['ori'].values)

        # minimum dist_to_other across pairs for this fly, per frame
        if 'dist_to_other' in df.columns:
            min_dist = (df[df['id'] == fly_id]
                          .groupby('frame')['dist_to_other']
                          .min()
                          .reindex(fly_df['frame'])
                          .values)
        else:
            min_dist = None

        ax2 = ax.twinx()

        ax.plot(t, ori_deg, color='white', lw=0.8, alpha=0.9, label='ori (°)')
        ax.axhline(0, color='white', lw=0.3, alpha=0.3)

        if min_dist is not None:
            ax2.plot(t, min_dist, color='cyan', lw=0.8, alpha=0.6, label='min dist (px)')
            ax2.set_ylabel('min dist to other (px)', color='cyan', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='cyan')

        # shade flip event segments for this fly
        fly_events = events[events['fly_id'] == fly_id] if len(events) > 0 else pd.DataFrame()
        for _, ev in fly_events.iterrows():
            t_start = ev['start_frame'] / fps
            t_end   = ev['end_frame'] / fps
            ax.axvspan(t_start, t_end, color='red', alpha=0.2)
            ax.axvline(t_start, color='red', lw=1.0, ls='--', alpha=0.7)
            ax.text(t_start, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 180,
                    f"{ev['duration_sec']:.1f}s", color='red', fontsize=7, va='top')

        ax.set_ylabel('orientation (°)')
        ax.set_xlabel('time (s)')
        title = f'{acq} — ' if acq else ''
        ax.set_title(f'{title}fly {fly_id}  |  {len(fly_events)} flip events detected')
        ax.set_ylim(-190, 190)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels() if min_dist is not None else ([], [])
        flip_patch = mpatches.Patch(color='red', alpha=0.3, label='flip segment')
        ax.legend(handles=lines1 + lines2 + [flip_patch], fontsize=8, loc='upper right')

    plt.tight_layout()
    return fig


def plot_flip_duration_distribution(events, fps=60, figsize=(10, 4)):
    """
    Plot distribution of flip event durations and jump magnitudes.

    Arguments:
        events -- DataFrame from detect_flip_events()
    """
    if len(events) == 0:
        print("No events to plot.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].hist(events['duration_sec'], bins=30, color='tomato', edgecolor='white', lw=0.5)
    axes[0].set_xlabel('duration (s)')
    axes[0].set_ylabel('count')
    axes[0].set_title('Flip segment durations')

    axes[1].hist(np.abs(events['jump_in_rad']), bins=20, color='dodgerblue', edgecolor='white', lw=0.5)
    axes[1].axvline(np.pi, color='yellow', lw=1.5, ls='--', label='π')
    axes[1].set_xlabel('|jump| (rad)')
    axes[1].set_title('Jump magnitude at flip onset')
    axes[1].legend(fontsize=8)

    if 'min_dist_at_flip' in events.columns:
        axes[2].scatter(events['min_dist_at_flip'], events['duration_sec'],
                        color='limegreen', alpha=0.7, s=30, edgecolors='white', lw=0.5)
        axes[2].set_xlabel('min dist to other during flip (px)')
        axes[2].set_ylabel('duration (s)')
        axes[2].set_title('Flip duration vs proximity')

    plt.tight_layout()
    return fig
