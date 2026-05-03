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

def extract_bout_clips(df, avi_path, save_dir, max_frames=600, mark_flies=False, 
                       id_colors=None, n_clips=1):
    '''
    Extract video clips per action type in df.
    Clips are at most max_frames long, centered on the bout if the bout is shorter.

    Arguments:
        df       -- tracks dataframe with action and boutnum columns
        avi_path -- path to source .avi file
        save_dir -- directory to save clips

    Keyword Arguments:
        max_frames -- maximum clip length in frames; use -1 for full bout (default: 600)
        mark_flies -- if True, draw actor/target markers on each frame (default: False)
        id_colors  -- dict mapping fly id to BGR color tuple (default: None)
        n_clips    -- number of clips to extract per action; use -1 for all (default: 1)

    Returns:
        clip_paths -- list of saved clip paths
    '''
    if id_colors is None:
        id_colors = {
            0: (255, 144, 30),   # dodgerblue in BGR
            1: (71, 99, 255),    # tomato in BGR
            2: (50, 205, 50),    # limegreen in BGR
        }

    action_cols = [c for c in df.columns if f'{c}_boutnum' in df.columns]
    if len(action_cols) == 0:
        print("No action columns found in df.")
        return []

    cap = cv2.VideoCapture(avi_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clips_dir = os.path.join(save_dir, 'bout_clips')
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []

    # loop over all possible actions 
    for action in action_cols:
        boutnum_col = f'{action}_boutnum'
        target_col = f'{action}_target'
        has_target = target_col in df.columns

        total_bouts = sorted(df[df[action] != -1][boutnum_col].dropna().unique())
        if len(total_bouts) == 0:
            print(f"No bouts found for action: {action}")
            continue

        bouts_to_extract = total_bouts if n_clips == -1 else total_bouts[:n_clips]
        print(f"Extracting {len(bouts_to_extract)}/{len(total_bouts)} bouts for action: {action}")

        # iterate over bouts_to_extract bouts per action 
        for bout_num in bouts_to_extract:
            bout_rows = df[df[boutnum_col] == bout_num]
            bout_start = int(bout_rows['frame'].min())
            bout_end = int(bout_rows['frame'].max())
            bout_len = bout_end - bout_start + 1

            # Compute clip window
            clip_start = max(0, bout_start)
            clip_end = min(total_frames - 1, bout_end)

            if max_frames != -1 and (clip_end - clip_start + 1) > max_frames:
                bout_center = (bout_start + bout_end) // 2
                clip_start = max(0, bout_center - max_frames // 2)
                clip_end = min(total_frames - 1, clip_start + max_frames - 1)

            clip_len = clip_end - clip_start + 1
            clip_name = f'{action}_bout{int(bout_num):03d}_frames{clip_start}-{clip_end}.mp4'
            clip_path = os.path.join(clips_dir, clip_name)

            print(f"  Writing {clip_name} ({clip_len} frames, bout length={bout_len})")

            # Pre-build frame lookup dictionary for marking
            if mark_flies:
                frame_lookup = {}
                for frame_idx in range(clip_start, clip_end + 1):
                    frame_df = df[df['frame'] == frame_idx]
                    fly_positions = {}
                    for _, row in frame_df.drop_duplicates('id').iterrows():
                        fly_id = int(row['id'])
                        fly_positions[fly_id] = (int(row['pos_x']), int(row['pos_y']))

                    actor_id = None
                    target_id = None
                    actor_rows = frame_df[frame_df[action] == frame_df['id']]
                    if len(actor_rows) > 0:
                        actor_id = int(actor_rows.iloc[0]['id'])
                        if has_target:
                            targ_val = actor_rows.iloc[0][target_col]
                            if targ_val is not None and targ_val != -1:
                                target_id = int(targ_val)

                    frame_lookup[frame_idx] = (fly_positions, actor_id, target_id)

            # Write clip
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(clip_path, fourcc, fps, (frame_w, frame_h))

            for frame_idx in range(clip_start, clip_end + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                if mark_flies:
                    fly_positions, actor_id, target_id = frame_lookup[frame_idx]

                    # Small dot for all flies
                    for fly_id, (px, py) in fly_positions.items():
                        color = id_colors.get(fly_id, (255, 255, 255))
                        cv2.circle(frame, (px, py), radius=6, color=color, thickness=-1)

                    # Larger ring + 'A' for actor
                    if actor_id is not None and actor_id in fly_positions:
                        px, py = fly_positions[actor_id]
                        color = id_colors.get(actor_id, (255, 255, 255))
                        cv2.circle(frame, (px, py), radius=12, color=color, thickness=2)
                        cv2.putText(frame, 'A', (px + 14, py - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Larger ring + 'T' for target
                    if target_id is not None and target_id in fly_positions:
                        px, py = fly_positions[target_id]
                        color = id_colors.get(target_id, (255, 255, 255))
                        cv2.circle(frame, (px, py), radius=12, color=color, thickness=2)
                        cv2.putText(frame, 'T', (px + 14, py - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Action label and frame number — dimmed outside bout
                    is_bout_frame = bout_start <= frame_idx <= bout_end
                    label_color = (255, 255, 255) if is_bout_frame else (120, 120, 120)
                    cv2.putText(frame, f'{action} frame {frame_idx}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

                out.write(frame)

            out.release()
            clip_paths.append(clip_path)

    cap.release()
    print(f"Saved {len(clip_paths)} clips to {clips_dir}")
    return clip_paths