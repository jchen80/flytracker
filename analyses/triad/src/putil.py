import matplotlib.pyplot as plt
import cv2
from adjustText import adjust_text
import glob
import numpy as np
import os
import matplotlib.gridspec as gridspec

def plot_roi_overlay(avi_path, trk, calib, frame_idx=0, save_dir=None, acq=None,
               id_colors=None, arrow_len=30, figsize=(8, 8)):
    '''
    Plot a specified frame of .avi with fly positions, orientations, and ROI overlay.

    Arguments:
        avi_path  -- path to .avi file
        trk       -- FlyTracker tracks dataframe with pos_x, pos_y, ori, id, frame columns
        calib     -- FlyTracker calibration dict with 'rois' key

    Keyword Arguments:
        frame_idx  -- frame index to plot (default: 0)
        save_dir   -- directory to save figure (default: None, no save)
        acq        -- acquisition name for plot title (default: None)
        id_colors  -- dict mapping fly id to color (default: {0: dodgerblue, 1: tomato, 2: limegreen})
        arrow_len  -- length of orientation arrow in pixels (default: 30)
        figsize    -- figure size (default: (8, 8))

    Returns:
        fig, ax
    '''
    if id_colors is None:
        id_colors = {0: 'dodgerblue', 1: 'tomato', 2: 'limegreen'}

    # Load specified frame
    frame_rgb = get_frame_rgb(avi_path, frame_idx)
    frame_height, frame_width = frame_rgb.shape[:2]

    # ROI
    y0, x0, roi_h, roi_w = calib['rois']

    # Get tracking data for this frame
    trk_frame = trk[trk['frame'] == frame_idx].copy()
    assert len(trk_frame) > 0, f"No tracking data found for frame {frame_idx}"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(frame_rgb, origin='upper')

    rect = plt.Rectangle((x0, y0), roi_w, roi_h,
                          linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    texts = []
    for _, row in trk_frame.iterrows():
        fly_id = int(row['id'])
        px, py = row['pos_x'], row['pos_y']
        ori = row['ori']
        color = id_colors.get(fly_id, 'white')

        ax.scatter(px, py, s=120, color=color, zorder=5, label=f'Fly {fly_id}')
        ax.annotate('', xy=(px + arrow_len*np.cos(ori), py + arrow_len*np.sin(ori)),
                    xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        t = ax.text(px, py,
                    f"id={fly_id}\n({px:.0f}, {py:.0f})\nori={np.rad2deg(ori):.1f}°",
                    color=color, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
        texts.append(t)

    adjust_text(texts, ax=ax, expand=(1.5, 1.5),
                arrowprops=dict(arrowstyle='-', color='white', lw=0.8, alpha=0.6))

    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)
    ax.set_xticks(np.arange(0, frame_width, 100))
    ax.set_yticks(np.arange(0, frame_height, 100))
    ax.grid(True, color='white', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    title = f'{acq} — frame {frame_idx}' if acq else f'Frame {frame_idx}'
    ax.set_title(f'{title}\nfly positions & orientations')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'frame_{frame_idx}_sanity_check.png')
        fig.savefig(savepath, dpi=150)
        print(f"Saved to {savepath}")

    return fig, ax

def get_frame_rgb(avi_path, frame_idx):
    '''
    Extract a single frame from an .avi file.

    Arguments:
        avi_path  -- path to .avi file
        frame_idx -- frame index to extract

    Returns:
        frame_rgb -- HxWx3 numpy array in RGB
    '''
    cap = cv2.VideoCapture(avi_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Failed to read frame {frame_idx} from {avi_path}"
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def plot_transformation_sanity_check(df, calib, avi_path, frame_idx=None,
                                     flyid1=0, flyid2=1,
                                     save_dir=None, acq=None,
                                     id_colors=None, arrow_len=30,
                                     figsize=(18, 12)):
    '''
    Six-panel diagnostic plot showing coordinate transformations for a given frame.
    Panel 1: raw video frame with fly positions
    Panel 2: fly0-centered, no rotation
    Panel 3: fly0-centered, rotated (fly0 faces East)
    Panel 4: empty (placeholder)
    Panel 5: fly1-centered, no rotation
    Panel 6: fly1-centered, rotated (fly1 faces East)

    Arguments:
        df       -- output of do_transformations_on_df
        calib    -- FlyTracker calibration dict
        avi_path -- path to .avi file

    Keyword Arguments:
        frame_idx  -- frame index to plot; if None, picks a random frame (default: None)
        flyid1     -- id of focal fly (default: 0)
        flyid2     -- id of other fly (default: 1)
        save_dir   -- directory to save figure (default: None)
        acq        -- acquisition name for title (default: None)
        id_colors  -- dict mapping fly id to color (default: None)
        arrow_len  -- orientation arrow length in pixels (default: 30)
        figsize    -- figure size (default: (18, 12))

    Returns:
        fig
    '''
    if id_colors is None:
        id_colors = {0: 'dodgerblue', 1: 'tomato', 2: 'limegreen'}

    # Pick frame
    if frame_idx is None:
        rng = np.random.default_rng()
        frame_idx = int(rng.integers(0, df['frame'].max()))
    print(f"Using frame {frame_idx}")

    # Load frame
    frame_rgb = get_frame_rgb(avi_path, frame_idx)
    frame_h, frame_w = frame_rgb.shape[:2]

    # Get per-fly rows for this frame
    f0 = df[(df['id'] == flyid1) & (df['frame'] == frame_idx)].iloc[0]
    f1 = df[(df['id'] == flyid2) & (df['frame'] == frame_idx)].iloc[0]

    # Print summary
    print("\n--- Absolute positions (pixel space) ---")
    for label, f in [(f'Fly {flyid1}', f0), (f'Fly {flyid2}', f1)]:
        print(f"{label}: pos=({f['pos_x']:.1f}, {f['pos_y']:.1f})  "
              f"ori={np.rad2deg(f['ori']):.1f}°  "
              f"ctr=({f['ctr_x']:.1f}, {f['ctr_y']:.1f})")

    # ROI
    y0_roi, x0_roi, roi_h, roi_w = calib['rois']

    # Shared axis limit based on fly0's relative position
    rel_x = f0['targ_centered_to_focal_x']
    rel_y = f0['targ_centered_to_focal_y']
    lim = max(abs(rel_x), abs(rel_y)) * 1.5 + 50

    rel_x_1 = f1['targ_centered_to_focal_x']
    rel_y_1 = f1['targ_centered_to_focal_y']
    lim1 = max(abs(rel_x_1), abs(rel_y_1)) * 1.5 + 50

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4)

    # ---- Panel 1: Raw video frame ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(frame_rgb, origin='upper')
    rect = plt.Rectangle((x0_roi, y0_roi), roi_w, roi_h,
                          lw=2, edgecolor='cyan', facecolor='none', ls='--')
    ax1.add_patch(rect)
    for fly_id, f in [(flyid1, f0), (flyid2, f1)]:
        color = id_colors[fly_id]
        px, py, ori = f['pos_x'], f['pos_y'], f['ori']
        ax1.scatter(px, py, s=120, color=color, zorder=5)
        ax1.annotate('', xy=(px + arrow_len*np.cos(ori), py + arrow_len*np.sin(ori)),
                     xytext=(px, py),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax1.text(px+12, py-12, f"id={fly_id}\n({px:.0f},{py:.0f})\n{np.rad2deg(ori):.1f}°",
                 color=color, fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
    ax1.set_xlim(0, frame_w); ax1.set_ylim(frame_h, 0)
    ax1.set_xticks(np.arange(0, frame_w, 100))
    ax1.set_yticks(np.arange(0, frame_h, 100))
    ax1.grid(True, color='white', lw=0.5, alpha=0.3)
    ax1.set_title(f'Panel 1: Raw frame {frame_idx}\n(pixel coords, y-down)')
    ax1.set_xlabel('x (pixels)'); ax1.set_ylabel('y (pixels)')

    def _plot_centered_panel(ax, focal_id, focal_f, targ_id, targ_f,
                             rel_x, rel_y, lim, rotated=False):
        '''Helper to draw panels 2/3 and 5/6.'''
        focal_color = id_colors[focal_id]
        targ_color = id_colors[targ_id]

        # Focal fly at origin
        ax.scatter(0, 0, s=120, color=focal_color, zorder=5, label=f'Fly {focal_id} (focal)')
        if rotated:
            ax.annotate('', xy=(arrow_len, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=focal_color, lw=2))
            ax.text(5, -5, f"id={focal_id}\n(0,0)\nori→0° (rotated)",
                    color=focal_color, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))
        else:
            ori = focal_f['ori']
            ax.annotate('', xy=(arrow_len*np.cos(ori), arrow_len*np.sin(ori)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=focal_color, lw=2))
            ax.text(5, -5, f"id={focal_id}\n(0, 0)\n{np.rad2deg(ori):.1f}°",
                    color=focal_color, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

        # Target fly
        ax.scatter(rel_x, rel_y, s=120, color=targ_color, zorder=5, label=f'Fly {targ_id}')
        targ_ori = targ_f['rot_ori'] if rotated else targ_f['ori']
        ax.annotate('', xy=(rel_x + arrow_len*np.cos(targ_ori), rel_y + arrow_len*np.sin(targ_ori)),
                    xytext=(rel_x, rel_y),
                    arrowprops=dict(arrowstyle='->', color=targ_color, lw=2))
        theta_str = f"\nθ={np.rad2deg(focal_f['targ_pos_theta']):.1f}°" if rotated else ''
        ax.text(rel_x+5, rel_y-5,
                f"id={targ_id}\n({rel_x:.1f},{rel_y:.1f})\n{np.rad2deg(targ_ori):.1f}°{theta_str}",
                color=targ_color, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

        ax.axhline(0, color='white', lw=0.5, alpha=0.4)
        ax.axvline(0, color='white', lw=0.5, alpha=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(lim, -lim)
        ax.set_xticks(np.arange(-int(lim), int(lim), 100))
        ax.set_yticks(np.arange(-int(lim), int(lim), 100))
        ax.grid(True, color='white', lw=0.5, alpha=0.3)
        ax.legend(fontsize=8)

    # ---- Panel 2: Fly0-centered, no rotation ----
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_centered_panel(ax2, flyid1, f0, flyid2, f1,
                         f0['targ_centered_to_focal_x'], f0['targ_centered_to_focal_y'],
                         lim, rotated=False)
    ax2.set_title('Panel 2: Fly0-centered, no rotation\n(y-down pixel convention)')
    ax2.set_xlabel('x (pixels rel.)'); ax2.set_ylabel('y (pixels rel.)')

    # ---- Panel 3: Fly0-centered, rotated ----
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_centered_panel(ax3, flyid1, f0, flyid2, f1,
                         f0['targ_rel_pos_x'], f0['targ_rel_pos_y'],
                         lim, rotated=True)
    ax3.set_title('Panel 3: Fly0-centered, rotated\n(fly0 faces East)')
    ax3.set_xlabel('x (egocentric)'); ax3.set_ylabel('y (egocentric)')

    # ---- Panel 4: placeholder ----
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')

    # ---- Panel 5: Fly1-centered, no rotation ----
    ax5 = fig.add_subplot(gs[1, 1])
    _plot_centered_panel(ax5, flyid2, f1, flyid1, f0,
                         f1['targ_centered_to_focal_x'], f1['targ_centered_to_focal_y'],
                         lim1, rotated=False)
    ax5.set_title('Panel 5: Fly1-centered, no rotation\n(y-down pixel convention)')
    ax5.set_xlabel('x (pixels rel.)'); ax5.set_ylabel('y (pixels rel.)')

    # ---- Panel 6: Fly1-centered, rotated ----
    ax6 = fig.add_subplot(gs[1, 2])
    _plot_centered_panel(ax6, flyid2, f1, flyid1, f0,
                         f1['targ_rel_pos_x'], f1['targ_rel_pos_y'],
                         lim1, rotated=True)
    ax6.set_title('Panel 6: Fly1-centered, rotated\n(fly1 faces East)')
    ax6.set_xlabel('x (egocentric)'); ax6.set_ylabel('y (egocentric)')

    title = f'{acq} — ' if acq else ''
    fig.suptitle(f'{title}Y-axis diagnostic — frame {frame_idx}', fontsize=13)
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'yaxis_diagnostic_frame{frame_idx}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig