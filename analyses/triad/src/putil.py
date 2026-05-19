import matplotlib.pyplot as plt
import cv2
from adjustText import adjust_text
import glob
import numpy as np
import os
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns


from analyses.triad.src import util as tutil

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
        df       -- single-acquisition output of do_transformations_on_df
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

def _build_frame_lookup(df, clip_start, clip_end, action_col):
    '''Build per-frame dict of (fly_positions, fly_orientations, actor_target_pairs).
    df must be a single-acquisition tracks df.'''
    target_col = f'{action_col}_target'
    has_target = target_col in df.columns
    frame_lookup = {}
    clip_df = df[(df['frame'] >= clip_start) & (df['frame'] <= clip_end)]
    for frame_idx, frame_df in clip_df.groupby('frame'):
        fly_positions, fly_orientations = {}, {}
        for _, row in frame_df.drop_duplicates('id').iterrows():
            fid = int(row['id'])
            if pd.isna(row['pos_x']) or pd.isna(row['pos_y']):
                continue
            fly_positions[fid] = (int(row['pos_x']), int(row['pos_y']))
            if not pd.isna(row['ori']):
                fly_orientations[fid] = float(row['ori'])
        actor_target_pairs = []
        for _, actor_row in frame_df[frame_df[action_col] == frame_df['id']].iterrows():
            a_id = int(actor_row['id'])
            t_id = None
            if has_target:
                targ_val = actor_row[target_col]
                if targ_val is not None and targ_val != -1:
                    t_id = int(targ_val)
            actor_target_pairs.append((a_id, t_id))
        frame_lookup[frame_idx] = (fly_positions, fly_orientations, actor_target_pairs)
    return frame_lookup


def _write_annotated_clip(cap, clip_path, clip_start, clip_end, fps, frame_w, frame_h,
                           id_colors, frame_lookup=None, label_fn=None):
    '''
    Write a clip from cap to clip_path.
    frame_lookup  -- dict from _build_frame_lookup; if None, no fly annotation.
    label_fn(frame_idx) -- callable returning (text, bgr_color) for the frame label;
                           if None, no text is drawn.
    '''
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(clip_path, fourcc, fps, (frame_w, frame_h))
    arrow_len = 20

    for frame_idx in range(clip_start, clip_end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_lookup and frame_idx in frame_lookup:
            fly_positions, fly_orientations, actor_target_pairs = frame_lookup[frame_idx]

            for fid, (px, py) in fly_positions.items():
                color = id_colors.get(fid, (255, 255, 255))
                cv2.circle(frame, (px, py), radius=6, color=color, thickness=-1)
                if fid in fly_orientations:
                    ori = fly_orientations[fid]
                    ex = int(px + arrow_len * np.cos(ori))
                    ey = int(py + arrow_len * np.sin(ori))
                    cv2.arrowedLine(frame, (px, py), (ex, ey),
                                    color=color, thickness=2, tipLength=0.3)

            for actor_id, target_id in actor_target_pairs:
                if actor_id in fly_positions:
                    px, py = fly_positions[actor_id]
                    color = id_colors.get(actor_id, (255, 255, 255))
                    cv2.circle(frame, (px, py), radius=12, color=color, thickness=2)
                    cv2.putText(frame, 'A', (px + 14, py - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if target_id is not None and target_id in fly_positions:
                    px, py = fly_positions[target_id]
                    color = id_colors.get(target_id, (255, 255, 255))
                    cv2.circle(frame, (px, py), radius=12, color=color, thickness=2)
                    cv2.putText(frame, 'T', (px - 22, py - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label_fn is not None:
            text, color = label_fn(frame_idx)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

    out.release()


def extract_bout_clips(df, avi_path, save_dir, max_frames=600, mark_flies=False,
                       id_colors=None, n_clips=1, action_cols=None):
    '''
    Extract video clips per action type in df.
    Clips are at most max_frames long, centered on the bout if the bout is longer.

    Arguments:
        df       -- single-acquisition tracks dataframe with action and boutnum columns
        avi_path -- path to source .avi file
        save_dir -- directory to save clips

    Keyword Arguments:
        max_frames   -- maximum clip length in frames; use -1 for full bout (default: 600)
        mark_flies   -- if True, draw actor/target markers on each frame (default: False)
        id_colors    -- dict mapping fly id to BGR color tuple (default: None)
        n_clips      -- number of clips to extract per action; use -1 for all (default: 1)
        action_cols  -- list of action column names to extract clips for; if None, uses
                        all action columns detected in df (default: None)

    Returns:
        clip_paths -- list of saved clip paths
    '''
    if id_colors is None:
        id_colors = {
            0: (255, 144, 30),
            1: (71, 99, 255),
            2: (50, 205, 50),
            3: (0, 215, 255),
        }

    if action_cols is None:
        action_cols = [c for c in df.columns if f'{c}_boutnum' in df.columns]
    else:
        action_cols = [c for c in action_cols if f'{c}_boutnum' in df.columns]
    if len(action_cols) == 0:
        print("No action columns found in df.")
        return []

    cap = cv2.VideoCapture(avi_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clips_dir = os.path.join(save_dir, 'bout_clips')
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []

    for action in action_cols:
        boutnum_col = f'{action}_boutnum'
        total_bouts = sorted(df[df[action] != -1][boutnum_col].dropna().unique())
        if len(total_bouts) == 0:
            print(f"No bouts found for action: {action}")
            continue

        bouts_to_extract = total_bouts if n_clips == -1 else total_bouts[:n_clips]
        print(f"Extracting {len(bouts_to_extract)}/{len(total_bouts)} bouts for action: {action}")

        for bout_num in bouts_to_extract:
            bout_rows  = df[df[boutnum_col] == bout_num]
            bout_start = int(bout_rows['frame'].min())
            bout_end   = int(bout_rows['frame'].max())
            bout_len   = bout_end - bout_start + 1

            clip_start = max(0, bout_start)
            clip_end   = min(total_frames - 1, bout_end)
            if max_frames != -1 and (clip_end - clip_start + 1) > max_frames:
                bout_center = (bout_start + bout_end) // 2
                clip_start  = max(0, bout_center - max_frames // 2)
                clip_end    = min(total_frames - 1, clip_start + max_frames - 1)

            clip_name = f'{action}_bout{int(bout_num):03d}_frames{clip_start}-{clip_end}.mp4'
            clip_path = os.path.join(clips_dir, clip_name)
            print(f"  Writing {clip_name} ({clip_end - clip_start + 1} frames, bout={bout_len})")

            frame_lookup = _build_frame_lookup(df, clip_start, clip_end, action) if mark_flies else None

            def _label(fi, action=action, bs=bout_start, be=bout_end):
                color = (255, 255, 255) if bs <= fi <= be else (120, 120, 120)
                return f'{action} frame {fi}', color

            _write_annotated_clip(cap, clip_path, clip_start, clip_end, fps, frame_w, frame_h,
                                  id_colors, frame_lookup=frame_lookup,
                                  label_fn=_label if mark_flies else None)
            clip_paths.append(clip_path)

    cap.release()
    print(f"Saved {len(clip_paths)} clips to {clips_dir}")
    return clip_paths


def extract_switch_clips(df, avi_path, save_dir, action_col='courtship',
                          padding_frames=120, mark_flies=True, id_colors=None):
    '''
    Extract a short clip around each target-switch event in a courtship bout.
    Requires '{action_col}_switch' column (added by assign_target_orientation).

    Arguments:
        df           -- single-acquisition tracks df with switch and target columns
        avi_path     -- path to source .avi file
        save_dir     -- directory to save clips (saved into a 'switch_clips' subdir)

    Keyword Arguments:
        action_col     -- action column (default: 'courtship')
        padding_frames -- frames before/after switch frame to include (default: 120)
        mark_flies     -- if True, draw actor/target markers (default: True)
        id_colors      -- dict mapping fly id → BGR color tuple (default: None)

    Returns:
        clip_paths -- list of saved clip paths
    '''
    if id_colors is None:
        id_colors = {
            0: (255, 144, 30),
            1: (71, 99, 255),
            2: (50, 205, 50),
            3: (0, 215, 255),
        }

    switch_col = f'{action_col}_switch'
    target_col = f'{action_col}_target'

    if switch_col not in df.columns:
        print(f"No '{switch_col}' column found. Run assign_target_orientation first.")
        return []

    # one event per (frame, actor) — df has multiple rows per frame
    switch_events = (df[df[switch_col] == 1]
                     .drop_duplicates(subset=['frame', 'id'])
                     [['frame', 'id', target_col]]
                     .copy())

    if len(switch_events) == 0:
        print("No switch events found.")
        return []

    cap          = cv2.VideoCapture(avi_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clips_dir = os.path.join(save_dir, 'switch_clips')
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []

    print(f"Extracting {len(switch_events)} switch clip(s) for '{action_col}'")

    for _, ev in switch_events.iterrows():
        switch_frame = int(ev['frame'])
        actor_id     = int(ev['id'])
        to_target    = int(ev[target_col]) if ev[target_col] != -1 else None

        # from_target: last assigned target for this actor before the switch frame
        prior = df[(df['id'] == actor_id) &
                   (df['frame'] < switch_frame) &
                   (df[switch_col] != -1)]
        from_target = None
        if len(prior) > 0:
            last_val = prior.sort_values('frame').iloc[-1][target_col]
            from_target = int(last_val) if last_val != -1 else None

        clip_start = max(0, switch_frame - padding_frames)
        clip_end   = min(total_frames - 1, switch_frame + padding_frames)

        from_str  = str(from_target) if from_target is not None else 'NA'
        to_str    = str(to_target)   if to_target   is not None else 'NA'
        clip_name = (f'{action_col}_switch_fr{switch_frame:06d}'
                     f'_act{actor_id}_from{from_str}_to{to_str}.mp4')
        clip_path = os.path.join(clips_dir, clip_name)
        print(f"  {clip_name}  ({clip_end - clip_start + 1} frames)")

        frame_lookup = _build_frame_lookup(df, clip_start, clip_end, action_col) if mark_flies else None

        def _label(fi, sf=switch_frame, fs=from_str, ts=to_str):
            if fi == sf:
                return f'fr {fi}   << SWITCH {fs} -> {ts}', (0, 0, 255)
            return f'fr {fi}', (255, 255, 255)

        _write_annotated_clip(cap, clip_path, clip_start, clip_end, fps, frame_w, frame_h,
                              id_colors, frame_lookup=frame_lookup, label_fn=_label)
        clip_paths.append(clip_path)

    cap.release()
    print(f"Saved {len(clip_paths)} switch clips to {clips_dir}")
    return clip_paths


def sample_frames_by_metric(df, avi_path, save_dir, metric, acq=None,
                             action_col='courtship',
                             step=1.0, n_samples=5,
                             metric_range=None,
                             id_colors=None):
    '''
    Sample example frames across evenly-spaced bins of a metric during action bouts,
    restricted to actor→target pairs. Saves one annotated PNG per sampled frame.

    Arguments:
        df       -- single-acquisition tracks df with action, target, and metric columns
        avi_path -- path to source .avi file
        save_dir -- parent output directory; bin subdirs (e.g. '0.0-1.0/') created inside
        metric   -- column to bin on (e.g. 'dist_to_other', 'abs_theta_error_deg')

    Keyword Arguments:
        acq          -- acquisition name used as filename prefix (default: None)
        action_col   -- action column to filter on (default: 'courtship')
        step         -- bin width in metric units (default: 1.0)
        n_samples    -- max frames to sample per bin (default: 5)
        metric_range -- (min, max) tuple; if None uses data range
        id_colors    -- dict mapping fly id → BGR color tuple

    Returns:
        saved_paths -- list of saved image paths
    '''
    if id_colors is None:
        id_colors = {
            0: (255, 144, 30),
            1: (71, 99, 255),
            2: (50, 205, 50),
            3: (0, 215, 255),
        }

    target_col = f'{action_col}_target'
    required = [action_col, target_col, metric, 'pair', 'id', 'frame', 'pos_x', 'pos_y', 'ori']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Missing columns {missing}, skipping.")
        return []

    # Filter to actor→target rows only
    pair_fly0    = df['pair'].str.split('_').str[0].astype(int)
    pair_fly1    = df['pair'].str.split('_').str[-1].astype(int)
    fly_id       = df['id'].astype(int)
    targ_id      = pd.to_numeric(df[target_col], errors='coerce').fillna(-1).astype(int)
    is_acting    = (df[action_col] == df['id'])
    has_target   = (targ_id != -1)
    pair_matches = (
        ((pair_fly0 == fly_id) & (pair_fly1 == targ_id)) |
        ((pair_fly1 == fly_id) & (pair_fly0 == targ_id))
    )
    plot_df = df[is_acting & has_target & pair_matches].copy()
    plot_df['_target_id'] = targ_id[plot_df.index]

    vals = plot_df[metric].dropna()
    if len(plot_df) == 0 or len(vals) == 0:
        print(f"  No actor→target rows with valid '{metric}', skipping.")
        return []

    v_min = metric_range[0] if metric_range is not None else float(np.floor(vals.min()))
    v_max = metric_range[1] if metric_range is not None else float(np.ceil(vals.max()))
    bin_edges = np.arange(v_min, v_max + step, step)

    cap = cv2.VideoCapture(avi_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_paths = []
    acq_prefix  = f'{acq}_' if acq else ''

    for i in range(len(bin_edges) - 1):
        b_lo, b_hi = bin_edges[i], bin_edges[i + 1]
        bin_rows = plot_df[(plot_df[metric] >= b_lo) & (plot_df[metric] < b_hi)]
        if len(bin_rows) == 0:
            continue

        sample_rows = bin_rows.sample(min(n_samples, len(bin_rows)), random_state=42)
        for _, row in sample_rows.iterrows():
            frame_idx    = int(row['frame'])
            actor_id     = int(row['id'])
            target_id_v  = int(row['_target_id'])
            metric_val   = row[metric]

            if frame_idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_img = cap.read()
            if not ret:
                continue

            # Build per-fly position / orientation for this frame
            frame_df         = df[df['frame'] == frame_idx].drop_duplicates('id')
            fly_positions    = {}
            fly_orientations = {}
            for _, frow in frame_df.iterrows():
                fid = int(frow['id'])
                if pd.isna(frow['pos_x']) or pd.isna(frow['pos_y']):
                    continue
                fly_positions[fid] = (int(frow['pos_x']), int(frow['pos_y']))
                if not pd.isna(frow['ori']):
                    fly_orientations[fid] = float(frow['ori'])

            arrow_len = 20
            for fid, (px, py) in fly_positions.items():
                color = id_colors.get(fid, (255, 255, 255))
                cv2.circle(frame_img, (px, py), radius=6, color=color, thickness=-1)
                if fid in fly_orientations:
                    ori = fly_orientations[fid]
                    ex = int(px + arrow_len * np.cos(ori))
                    ey = int(py + arrow_len * np.sin(ori))
                    cv2.arrowedLine(frame_img, (px, py), (ex, ey),
                                    color=color, thickness=2, tipLength=0.3)

            for fid_mark, label in [(actor_id, 'A'), (target_id_v, 'T')]:
                if fid_mark in fly_positions:
                    px, py = fly_positions[fid_mark]
                    color = id_colors.get(fid_mark, (255, 255, 255))
                    cv2.circle(frame_img, (px, py), radius=14, color=color, thickness=2)
                    cv2.putText(frame_img, label, (px + 16, py - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            overlay = f'{metric}={metric_val:.2f}  fr={frame_idx}'
            if acq:
                overlay += f'  {acq}'
            cv2.putText(frame_img, overlay, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            fname  = f'{b_lo:.1f}-{b_hi:.1f}_{acq_prefix}frame{frame_idx:06d}_act{actor_id}_targ{target_id_v}.png'
            fpath  = os.path.join(save_dir, fname)
            cv2.imwrite(fpath, frame_img)
            saved_paths.append(fpath)

    cap.release()
    return saved_paths


def sample_frames_by_metric_across_assays(assay_dfs, rootdir, metric, figdir,
                                           action_col='courtship',
                                           step=1.0, n_samples=5,
                                           metric_range=None,
                                           id_colors=None):
    '''
    Wrapper around sample_frames_by_metric that iterates over all assay types
    and acquisitions, saving to:
        figdir/{metric}_interval_examples/{assay_type}/{bin_lo}-{bin_hi}/

    Filenames include the acquisition name and frame number.
    Bins are defined globally across all assay data so the intervals are comparable.

    Arguments:
        assay_dfs -- dict mapping assay_type → df (from data_io.get_assay_dfs)
        rootdir   -- root data directory (expects raw_videos/{acq}/{acq}.avi)
        metric    -- metric column to bin on
        figdir    -- root figures directory

    Keyword Arguments:
        action_col   -- action column to filter on (default: 'courtship')
        step         -- bin width in metric units (default: 1.0)
        n_samples    -- max frames to sample per bin per acquisition (default: 5)
        metric_range -- (min, max) tuple; if None computed from all acting rows
        id_colors    -- dict mapping fly id → BGR color tuple

    Returns:
        all_paths -- list of all saved image paths
    '''
    # Compute global metric range so bins are consistent across assays
    if metric_range is None:
        all_vals = pd.concat([
            d.loc[d[action_col] == d['id'], metric].dropna()
            for d in assay_dfs.values()
            if action_col in d.columns and metric in d.columns
        ])
        if len(all_vals) == 0:
            print(f"No values for '{metric}' during '{action_col}'. Aborting.")
            return []
        metric_range = (float(np.floor(all_vals.min())), float(np.ceil(all_vals.max())))
        print(f"Auto metric_range for {metric}: {metric_range}")

    out_root = os.path.join(figdir, f'{metric}_interval_examples')
    os.makedirs(out_root, exist_ok=True)

    all_paths = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns:
            print(f"\n{assay_type}: '{action_col}' column missing, skipping.")
            continue

        assay_out_dir = os.path.join(out_root, assay_type)
        os.makedirs(assay_out_dir, exist_ok=True)

        acqs = sorted(assay_df['acquisition'].unique())
        print(f"\n{assay_type}: {len(acqs)} acquisitions")

        for acq in acqs:
            avi_path = os.path.join(rootdir, 'raw_videos', acq, f'{acq}.avi')
            if not os.path.exists(avi_path):
                print(f"  {acq}: .avi not found, skipping.")
                continue

            acq_df = assay_df[assay_df['acquisition'] == acq].copy()
            paths  = sample_frames_by_metric(
                acq_df, avi_path, assay_out_dir, metric, acq=acq,
                action_col=action_col, step=step, n_samples=n_samples,
                metric_range=metric_range, id_colors=id_colors)
            print(f"  {acq}: {len(paths)} frames saved")
            all_paths.extend(paths)

    print(f"\nTotal: {len(all_paths)} frames → {out_root}")
    return all_paths


def _plot_bout_signals_on_axes(axes, actor_rows, candidate_ids, id_colors, lookback_frames, fps):
    '''
    Plot all signals on provided axes. Shared by plot_bout_signals and plot_switch_signals.

    Arguments:
        axes          -- list of 4 axes [te, dte_smooth, dist, relvel_smooth]
        actor_rows    -- prepped actor rows df with _cand_id, _t, and signal columns
        candidate_ids -- sorted list of candidate fly ids
        id_colors     -- dict mapping fly id to color
        lookback_frames -- smoothing window in frames
        fps           -- frames per second
    '''
    actor_rows = actor_rows.sort_values(['_cand_id', 'frame'])

    # compute signals
    actor_rows['_abs_theta'] = actor_rows['theta_error'].abs()
    actor_rows['_abs_theta_dt'] = (actor_rows
                                    .groupby('_cand_id')['_abs_theta']
                                    .transform(lambda x: x.diff() / (1/fps)))
    actor_rows['_abs_theta_dt_smoothed'] = (actor_rows
                                             .groupby('_cand_id')['_abs_theta_dt']
                                             .transform(lambda x: x.rolling(
                                                 window=lookback_frames, min_periods=1).mean()))
    actor_rows['_rel_vel_smoothed'] = (actor_rows
                                        .groupby('_cand_id')['rel_vel']
                                        .transform(lambda x: x.rolling(
                                            window=lookback_frames, min_periods=1).mean()))

    for cand_id in candidate_ids:
        cand_rows = actor_rows[actor_rows['_cand_id'] == cand_id].sort_values('frame')
        color = id_colors.get(cand_id, 'white')
        t = cand_rows['_t'].values
        label = f'fly {cand_id}'

        axes[0].plot(t, np.rad2deg(cand_rows['_abs_theta']),
                     color=color, lw=1.5, label=label)
        axes[1].plot(t, np.rad2deg(cand_rows['_abs_theta_dt_smoothed']),
                     color=color, lw=1.5, label=label)
        axes[2].plot(t, cand_rows['dist_to_other'],
                     color=color, lw=1.5, label=label)
        axes[3].plot(t, cand_rows['_rel_vel_smoothed'],
                     color=color, lw=1.5, label=label)

    axes[0].set_ylabel('|θ error| (°)')
    axes[1].set_ylabel(f'smoothed d|θ error|/dt\n(°/s)')
    axes[2].set_ylabel('dist to other (px)')
    axes[3].set_ylabel(f'smoothed rel vel\n(px/s)')
    axes[0].legend(fontsize=9, loc='upper right')

    return actor_rows  # return with computed signal columns

def _mark_switch_frames(axes, switch_times_t):
    '''
    Draw a vertical dashed line at each switch time on all axes.

    Arguments:
        axes            -- list of axes
        switch_times_t  -- list of float time values (seconds relative to bout start)
    '''
    label = 'switch'
    for t in switch_times_t:
        for ax in axes:
            ax.axvline(t, color='yellow', lw=1.5, ls='--', alpha=0.9, label=label)
        label = None  # only label first line


def plot_bout_signals(df, bout_num, action_col='courtship',
                      fps=60, lookback_sec=1.0,
                      mark_switches=True,
                      save_dir=None, acq=None,
                      id_colors=None, figsize=(14, 12)):
    '''
    Plot all signals for a full bout: |TE|, smoothed d|TE|/dt, dist_to_other, smoothed rel_vel.

    Arguments:
        df       -- single-acquisition transformed tracks df
        bout_num -- bout number to plot

    Keyword Arguments:
        action_col    -- name of action column (default: 'courtship')
        fps           -- frames per second (default: 60)
        lookback_sec  -- smoothing window in seconds (default: 1.0)
        mark_switches -- if True, draw vertical lines at switch frames from
                         '{action_col}_switch' column (default: True)
        save_dir      -- directory to save figure (default: None)
        acq           -- acquisition name for title (default: None)
        id_colors     -- dict mapping fly id to color (default: None)
        figsize       -- figure size (default: (14, 12))

    Returns:
        fig, axes
    '''
    if id_colors is None:
        id_colors = {0: 'dodgerblue', 1: 'tomato', 2: 'limegreen'}

    actor_rows, actor_id, bout_start, bout_end = _get_bout_actor_rows(
        df, bout_num, action_col, fps)
    if actor_rows is None:
        return None, None

    lookback_frames = int(lookback_sec * fps)
    candidate_ids = sorted(actor_rows['_cand_id'].unique())
    bout_start_t = 0
    bout_end_t = (bout_end - bout_start) / fps

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    _plot_bout_signals_on_axes(axes, actor_rows, candidate_ids,
                                id_colors, lookback_frames, fps)
    _shade_bout_axes(axes, bout_start_t, bout_end_t)

    if mark_switches:
        switch_col = f'{action_col}_switch'
        boutnum_col = f'{action_col}_boutnum'
        if switch_col in df.columns:
            switch_frames = (df[(df['id'] == actor_id) &
                               (df[switch_col] == 1) &
                               (df[boutnum_col] == bout_num)]['frame'].unique())
            switch_times_t = sorted((f - bout_start) / fps for f in switch_frames)
            if switch_times_t:
                _mark_switch_frames(axes, switch_times_t)

    axes[3].set_xlabel('time relative to bout start (s)')
    fig.suptitle(_bout_title(acq, action_col, bout_num, actor_id, bout_start, bout_end, fps),
                 fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir,
                                f'{action_col}_bout{int(bout_num):03d}_signals.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, axes



def _get_bout_actor_rows(df, bout_num, action_col, fps):
    '''Extract and prep actor rows for a given bout. Used by plotting functions.'''
    boutnum_col = f'{action_col}_boutnum'
    bout_rows = df[df[boutnum_col] == bout_num]
    if len(bout_rows) == 0:
        print(f"No rows found for bout {bout_num}")
        return None, None, None, None

    actor_id = int(bout_rows[bout_rows[action_col] == bout_rows['id']].iloc[0]['id'])
    bout_start = int(bout_rows['frame'].min())
    bout_end = int(bout_rows['frame'].max())

    pad = int(fps)
    frame_min = max(0, bout_start - pad)
    frame_max = bout_end + pad

    actor_rows = df[(df['id'] == actor_id) &
                    (df['frame'] >= frame_min) &
                    (df['frame'] <= frame_max)].copy()

    actor_rows['_cand_id'] = (actor_rows['pair']
                               .str.split('_')
                               .apply(lambda x: [int(i) for i in x])
                               .apply(lambda x: [i for i in x if i != actor_id][0]))
    actor_rows['_t'] = (actor_rows['frame'] - bout_start) / fps

    return actor_rows, actor_id, bout_start, bout_end


def _shade_bout_axes(axes, bout_start_t, bout_end_t):
    '''Shade bout region and add reference lines. Used by plotting functions.'''
    for ax in axes:
        ax.axvspan(bout_start_t, bout_end_t, color='white', alpha=0.08)
        ax.axvline(bout_start_t, color='white', lw=1, ls='--', alpha=0.5)
        ax.axvline(bout_end_t, color='white', lw=1, ls='--', alpha=0.5)
        ax.axhline(0, color='white', lw=0.5, alpha=0.3)

def _bout_title(acq, action_col, bout_num, actor_id, bout_start, bout_end, fps):
    '''Generate consistent bout title string. Used by plotting functions.'''
    base = f'{acq} — ' if acq else ''
    return (f'{base}{action_col} bout {bout_num} | actor: fly {actor_id}\n'
            f'frames {bout_start}–{bout_end} ({(bout_end-bout_start)/fps:.1f}s)')

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


def _add_focal_fly_marker(ax, ppm):
    '''Add focal fly position marker and heading arrow to an axis.'''
    arrow_len = 2 if ppm is not None else 40
    ax.scatter(0, 0, s=150, color='white', zorder=5, marker='o')
    ax.annotate('', xy=(arrow_len, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)
    ax.axvline(0, color='white', lw=0.5, alpha=0.3)

def plot_metric_distribution(df, metric='dist_to_other',
                              action_cols=None, focal_fly_ids=None,
                              save_dir=None, acq=None,
                              figsize=(10, 5), bins=50,
                              xlim_percentile=99):
    plot_df = _filter_by_focal_fly(df, focal_fly_ids)
    if len(plot_df) == 0:
        print("No rows found after focal fly filtering.")
        return None, None
    if metric not in plot_df.columns:
        print(f"Metric '{metric}' not found.")
        return None, None

    panels = _build_condition_panels(plot_df, action_cols)
    palette = sns.color_palette('husl', len(panels))

    fig, ax = plt.subplots(figsize=figsize)
    for (label, panel_df), color in zip(panels, palette):
        # per-condition bin edges, clipped at xlim_percentile to suppress long tails
        panel_vals = panel_df.drop_duplicates(['frame', 'pair'])[metric].dropna()
        x_max = np.percentile(panel_vals, xlim_percentile) if xlim_percentile is not None else panel_vals.max()
        bin_edges = np.linspace(panel_vals.min(), x_max, bins + 1)
        counts, n = _compute_metric_histogram(panel_df, metric, bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.step(bin_centers, counts, color=color, alpha=0.8,
                label=f'{label} (n={n})', where='mid')
        ax.fill_between(bin_centers, counts, alpha=0.3, color=color, step='mid')

    ax.set_xlabel(metric)
    ax.set_ylabel('density')
    ax.legend(fontsize=9)
    title = f'{acq} — ' if acq else ''
    focal_str = f'focal fly: {focal_fly_ids}' if focal_fly_ids is not None else 'all pairs'
    ax.set_title(f'{title}{metric} distribution\n{focal_str}')
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        savepath = os.path.join(save_dir, f'{metric}_distribution_{action_str}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, ax

def plot_metric_distribution_across_assays(assay_dfs, metric='dist_to_other',
                                            action_cols=None, focal_flies_map=None,
                                            assay_colors=None, bins=50,
                                            include_non_action=False,
                                            target_action_col=None,
                                            xlim_percentile=99,
                                            save_dir=None, figsize=None):
    """
    Plot distribution of a metric across multiple assay types, optionally split by action conditions.
    Arguments:
    - assay_dfs: dict mapping assay type to its dataframe which is processed to contain per-frame, per-pair of fly metrics such as dist_to_other, theta_error, etc.
    - metric: name of the metric column to plot (e.g. 'dist_to_other)
    - action_cols: list of action column names to split by (e.g. ['courtship', 'aggression']). If None, only 'all frames' panel is plotted.
    - focal_flies_map: dict mapping triad_type to list of focal fly ids to include in the plot when considering pairs. If None, all flies are included.
    - assay_colors: dict mapping assay type to color for plotting. If None, a default color palette is used.
    - bins: number of bins for the histogram
    - include_non_action: if True, add a non-action panel for each action, labeled
                          'non-<action> (all pairs)' to indicate it covers all fly pairs (default: False)
    - target_action_col: if set, the regular action panel is labeled '<action> (all pairs)' and
                         an additional '<action> (target only)' panel is added showing only the
                         focal fly → assigned target pair. When not set, the action panel keeps
                         its plain label. (e.g. 'courtship'). (default: None)
    - xlim_percentile: clip bin range to this percentile of the data to suppress long tails;
                       set to None to use the full data range (default: 99)
    - save_dir: directory to save the figure. If None, figure is not saved.
    - figsize: tuple specifying figure size. If None, size is determined by number of panels.
    """
    # Build condition labels
    condition_labels = ['all frames']
    if action_cols is not None:
        for a in action_cols:
            if any(a in df.columns for df in assay_dfs.values()):
                condition_labels.append(f'{a} (all pairs)' if target_action_col == a else a)
                if target_action_col == a:
                    condition_labels.append(f'{a} (target only)')
                if include_non_action:
                    condition_labels.append(f'non-{a} (all pairs)')

    n_panels = len(condition_labels)
    if figsize is None:
        figsize = (6 * n_panels, 5)

    if assay_colors is None:
        palette = sns.color_palette('husl', len(assay_dfs))
        assay_colors = {k: palette[i] for i, k in enumerate(sorted(assay_dfs.keys()))}

    def _apply_condition_filter(plot_df, cond_label):
        '''Filter plot_df to the rows for a given condition label. Returns None to skip.'''
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

    # Pass 1: filter data per (cond_label, assay_type)
    panel_dfs = {}
    for cond_label in condition_labels:
        for assay_type in sorted(assay_dfs.keys()):
            assay_df = assay_dfs[assay_type]
            triad_type = assay_df['triad_type'].iloc[0]
            focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
            filtered = _apply_condition_filter(
                _filter_by_focal_fly(assay_df, focal_flies), cond_label)
            panel_dfs[(cond_label, assay_type)] = filtered

    # Pass 2: compute per-condition bin edges from data pooled across assays
    condition_bin_edges = {}
    for cond_label in condition_labels:
        cond_vals = [
            panel_dfs[(cond_label, at)].drop_duplicates(['frame', 'pair', 'acquisition'])[metric].dropna()
            for at in sorted(assay_dfs.keys())
            if panel_dfs[(cond_label, at)] is not None and len(panel_dfs[(cond_label, at)]) > 0
        ]
        if not cond_vals:
            continue
        combined = pd.concat(cond_vals)
        x_max = np.percentile(combined, xlim_percentile) if xlim_percentile is not None else combined.max()
        condition_bin_edges[cond_label] = np.linspace(combined.min(), x_max, bins + 1)

    # Pass 3: plot
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, cond_label in zip(axes, condition_labels):
        bin_edges = condition_bin_edges.get(cond_label)
        if bin_edges is None:
            continue
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for assay_type in sorted(assay_dfs.keys()):
            plot_df = panel_dfs.get((cond_label, assay_type))
            if plot_df is None or len(plot_df) == 0:
                continue
            counts, n = _compute_metric_histogram(
                plot_df, metric, bin_edges,
                dedup_cols=['frame', 'pair', 'acquisition'])
            num_acq = plot_df['acquisition'].nunique()
            color = assay_colors.get(assay_type, 'white')
            ax.step(bin_centers, counts, color=color, alpha=0.8,
                    label=f'{assay_type} (n={num_acq})', where='mid')
            ax.fill_between(bin_centers, counts, alpha=0.1, color=color, step='mid')

        ax.set_xlabel(metric)
        ax.set_title(cond_label)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('density')
    fig.suptitle(f'{metric} distribution across assay types', fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        savepath = os.path.join(save_dir,
                                f'{metric}_distribution_across_assays_{action_str}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, axes

def plot_relative_position_density(df, action_cols=None, focal_fly_ids=None,
                                    save_dir=None, acq=None,
                                    figsize=None, n_grid=100, ppm=None,
                                    vmax_percentile=99):
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
        h, x_edges, y_edges, n = _compute_position_histogram(
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
        im = ax.imshow(h.T, origin='lower', aspect='equal', cmap='inferno',
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
    fig.suptitle(f'{title}relative position density\n{focal_str}', fontsize=12)
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        savepath = os.path.join(save_dir, f'relative_position_density_{action_str}.png')
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
                                                  save_dir=None, figsize=None):
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

    n_rows = len(condition_labels)
    n_cols = len(assay_types)
    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    # global axis limits
    all_x, all_y = [], []
    for assay_type, assay_df in assay_dfs.items():
        ppm = ppm_dict.get(assay_type, 1)
        triad_type = assay_df['triad_type'].iloc[0]
        focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
        plot_df = _filter_by_focal_fly(assay_df, focal_flies)
        dedup = plot_df.drop_duplicates(['frame', 'pair', 'acquisition'])
        all_x.append(dedup['targ_rel_pos_x'].dropna() / ppm)
        all_y.append(dedup['targ_rel_pos_y'].dropna() / ppm)
    x_lim = (pd.concat(all_x).min(), pd.concat(all_x).max())
    y_lim = (pd.concat(all_y).min(), pd.concat(all_y).max())

    # pre-compute all histograms
    histograms = {}
    n_points = {}
    for cond_label in condition_labels:
        for assay_type in assay_types:
            assay_df = assay_dfs[assay_type]
            ppm = ppm_dict.get(assay_type, 1)
            triad_type = assay_df['triad_type'].iloc[0]
            focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
            plot_df = _filter_by_focal_fly(assay_df, focal_flies)

            if cond_label == 'all frames':
                pass
            elif cond_label.startswith('non-'):
                action = cond_label[4:].replace(' (all pairs)', '')
                if action not in plot_df.columns:
                    histograms[(cond_label, assay_type)] = None
                    continue
                plot_df = _exclude_action_frames(plot_df, action)
            elif cond_label.endswith(' (target only)'):
                action = cond_label[:-len(' (target only)')]
                if action not in plot_df.columns:
                    histograms[(cond_label, assay_type)] = None
                    continue
                plot_df = _select_action_frames(plot_df, action)
                plot_df = _filter_to_target_pairs(plot_df, action)
            else:
                action = cond_label.replace(' (all pairs)', '')
                if action not in plot_df.columns:
                    histograms[(cond_label, assay_type)] = None
                    continue
                plot_df = _select_action_frames(plot_df, action)

            if len(plot_df) == 0:
                histograms[(cond_label, assay_type)] = None
                continue

            h, x_edges, y_edges, n = _compute_position_histogram(
                plot_df, x_lim, y_lim, n_grid, ppm,
                dedup_cols=['frame', 'pair', 'acquisition'])
            histograms[(cond_label, assay_type)] = (h, x_edges, y_edges)
            n_points[(cond_label, assay_type)] = n

    # global color scale with percentile clip across ALL panels
    valid = [v for v in histograms.values() if v is not None]
    all_nonzero = np.concatenate([h[h > 0] for h, _, _ in valid])
    vmin = 0
    vmax = np.percentile(all_nonzero, vmax_percentile)
    first_ppm = next(iter(ppm_dict.values()))
    unit_str = '% / mm²' if first_ppm is not None else 'prob. density'
    density_label = f'{unit_str} (>{vmax:.3f} clipped at {vmax_percentile}th pct)'
    print(f"Global vmax ({vmax_percentile}th percentile): {vmax:.4f} {unit_str}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              sharex=True, sharey=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, cond_label in enumerate(condition_labels):
        for col_idx, assay_type in enumerate(assay_types):
            ax = axes[row_idx, col_idx]
            result = histograms.get((cond_label, assay_type))
            if result is None:
                ax.set_visible(False)
                continue

            h, x_edges, y_edges = result
            n = n_points.get((cond_label, assay_type), 0)
            ppm = ppm_dict.get(assay_type, 1)

            im = ax.imshow(h.T, origin='lower', aspect='equal', cmap='inferno',
                           extent=[x_edges[0], x_edges[-1],
                                   y_edges[0], y_edges[-1]],
                           vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label=density_label)
            _add_focal_fly_marker(ax, ppm)
            ax.set_xlim(x_lim); ax.set_ylim(y_lim)
            ax.set_aspect('equal')
            ax.text(0.02, 0.02, f'n={n}', transform=ax.transAxes,
                    color='white', fontsize=7, va='bottom')

            if row_idx == 0:
                ax.set_title(assay_type, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f'{cond_label}\ny (mm)', fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel('x (mm)', fontsize=9)

    fig.suptitle('Relative position density across assay types', fontsize=13)
    plt.tight_layout()

    if save_dir is not None:
        action_str = '_'.join(action_cols) if action_cols else 'all'
        savepath = os.path.join(save_dir,
                                f'relative_position_density_across_assays_{action_str}.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")
    return fig, axes


def plot_action_fraction_across_assays(assay_dfs, action_col,
                                        assay_colors=None,
                                        save_dir=None, figsize=(6, 5)):
    '''
    Plot fraction of frames in which action_col occurs, compared across assay types.
    Fraction is computed per acquisition as unique action frames / total frames.
    Each point is one acquisition; bar shows the mean +/- SE.

    Arguments:
        assay_dfs  -- dict mapping assay type to its dataframe
        action_col -- name of action column

    Keyword Arguments:
        assay_colors -- dict mapping assay type to color (default: None)
        save_dir     -- directory to save figure (default: None)
        figsize      -- figure size (default: (6, 5))

    Returns:
        fig, ax
    '''
    if assay_colors is None:
        palette = sns.color_palette('husl', len(assay_dfs))
        assay_colors = {k: palette[i] for i, k in enumerate(sorted(assay_dfs.keys()))}

    records = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns:
            print(f"Action '{action_col}' not found in assay '{assay_type}', skipping.")
            continue
        for acq_name, acq_df in assay_df.groupby('acquisition'):
            total_frames = acq_df['frame'].nunique()
            if total_frames == 0:
                continue
            action_frames = acq_df[acq_df[action_col] != -1]['frame'].nunique()
            records.append({
                'assay_type': assay_type,
                'acquisition': acq_name,
                'fraction': action_frames / total_frames,
            })

    if len(records) == 0:
        print("No data to plot.")
        return None, None

    plot_data = pd.DataFrame(records)
    assay_order = sorted(assay_dfs.keys())
    palette = [assay_colors.get(a, 'white') for a in assay_order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_data, x='assay_type', y='fraction',
                order=assay_order, palette=palette,
                errorbar='se', ax=ax, alpha=0.7)
    sns.stripplot(data=plot_data, x='assay_type', y='fraction',
                  order=assay_order, palette=palette,
                  size=7, jitter=True, ax=ax, alpha=0.9,
                  linewidth=0.5, edgecolor='white')

    n_per_assay = plot_data.groupby('assay_type')['acquisition'].nunique()
    ax.set_xticklabels([f'{a}\n(n={n_per_assay.get(a, 0)})' for a in assay_order])
    ax.set_xlabel('assay type')
    ax.set_ylabel('fraction of frames')
    ax.set_title(f'{action_col}: fraction of frames')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'{action_col}_fraction_across_assays.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, ax


def plot_action_rate_across_assays(assay_dfs, action_col, fps=None,
                                    norm_minutes=30,
                                    assay_colors=None,
                                    save_dir=None, figsize=(6, 5)):
    '''
    Plot number of action bouts compared across assay types.
    If norm_minutes is set, normalizes by recording duration (bouts per norm_minutes).
    If norm_minutes is None, plots raw bout count — useful for one-off events like copulation.
    Each point is one acquisition; bar shows the mean +/- SE.

    Arguments:
        assay_dfs  -- dict mapping assay type to its dataframe
        action_col -- name of action column

    Keyword Arguments:
        fps          -- frames per second; required when norm_minutes is set (default: None)
        norm_minutes -- normalize to this many minutes; if None, plot raw count (default: 30)
        assay_colors -- dict mapping assay type to color (default: None)
        save_dir     -- directory to save figure (default: None)
        figsize      -- figure size (default: (6, 5))

    Returns:
        fig, ax
    '''
    if norm_minutes is not None and fps is None:
        raise ValueError("fps is required when norm_minutes is set.")

    if assay_colors is None:
        palette = sns.color_palette('husl', len(assay_dfs))
        assay_colors = {k: palette[i] for i, k in enumerate(sorted(assay_dfs.keys()))}

    boutnum_col = f'{action_col}_boutnum'
    records = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns:
            print(f"Action '{action_col}' not found in assay '{assay_type}', skipping.")
            continue
        if boutnum_col not in assay_df.columns:
            print(f"Boutnum column '{boutnum_col}' not found in assay '{assay_type}', skipping.")
            continue
        for acq_name, acq_df in assay_df.groupby('acquisition'):
            total_frames = acq_df['frame'].nunique()
            if total_frames == 0:
                continue
            n_bouts = acq_df[acq_df[action_col] != -1][boutnum_col].nunique()
            if norm_minutes is not None:
                duration_min = (total_frames / fps) / 60
                value = n_bouts / duration_min * norm_minutes
            else:
                value = n_bouts
            records.append({
                'assay_type': assay_type,
                'acquisition': acq_name,
                'value': value,
                'n_bouts': n_bouts,
            })

    if len(records) == 0:
        print("No data to plot.")
        return None, None

    plot_data = pd.DataFrame(records)
    assay_order = sorted(assay_dfs.keys())
    palette = [assay_colors.get(a, 'white') for a in assay_order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_data, x='assay_type', y='value',
                order=assay_order, palette=palette,
                errorbar='se', ax=ax, alpha=0.7)
    sns.stripplot(data=plot_data, x='assay_type', y='value',
                  order=assay_order, palette=palette,
                  size=7, jitter=True, ax=ax, alpha=0.9,
                  linewidth=0.5, edgecolor='white')

    n_per_assay = plot_data.groupby('assay_type')['acquisition'].nunique()
    ax.set_xticklabels([f'{a}\n(n={n_per_assay.get(a, 0)})' for a in assay_order])
    ax.set_xlabel('assay type')
    ylabel = f'bouts per {norm_minutes} min' if norm_minutes is not None else 'bout count'
    ax.set_ylabel(ylabel)
    ax.set_title(f'{action_col}: {ylabel}')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'{action_col}_rate_across_assays.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, ax


def plot_target_sex_fraction_across_assays(assay_dfs, action_col, sex_map,
                                            focal_flies_map=None,
                                            assay_colors=None,
                                            save_dir=None, figsize=(8, 5)):
    '''
    Plot fraction of action frames in which the focal fly targeted a male vs female,
    compared across assay types.

    Fraction is computed per acquisition as:
        frames targeting sex X / total frames with an assigned target

    Each point is one acquisition; bar shows mean +/- SE.
    Male and female fractions are shown side by side as a grouped bar.

    Arguments:
        assay_dfs  -- dict mapping assay type to its dataframe
        action_col -- name of action column (e.g. 'courtship')
        sex_map    -- dict mapping triad_type to {fly_id: 'M' or 'F'},
                      e.g. {'MMF': {0: 'M', 1: 'M', 2: 'F'},
                             'MFF': {0: 'M', 1: 'F', 2: 'F'}}

    Keyword Arguments:
        focal_flies_map -- dict mapping triad_type to list of focal fly ids (default: None)
        assay_colors    -- dict mapping assay type to color (default: None)
        save_dir        -- directory to save figure (default: None)
        figsize         -- figure size (default: (8, 5))

    Returns:
        fig, ax
    '''
    target_col = f'{action_col}_target'

    if assay_colors is None:
        palette = sns.color_palette('husl', len(assay_dfs))
        assay_colors = {k: palette[i] for i, k in enumerate(sorted(assay_dfs.keys()))}

    records = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns or target_col not in assay_df.columns:
            print(f"Missing '{action_col}' or '{target_col}' in assay '{assay_type}', skipping.")
            continue

        triad_type = assay_df['triad_type'].iloc[0]
        fly_sex = sex_map.get(triad_type, {})
        if not fly_sex:
            print(f"No sex_map entry for triad_type '{triad_type}', skipping assay '{assay_type}'.")
            continue

        focal_flies = focal_flies_map.get(triad_type) if focal_flies_map else None
        plot_df = _filter_by_focal_fly(assay_df, focal_flies)

        # rows where focal fly is acting and a target is assigned
        acting_df = plot_df[
            (plot_df[action_col] == plot_df['id']) &
            (pd.to_numeric(plot_df[target_col], errors='coerce').fillna(-1).astype(int) != -1)
        ].drop_duplicates(['acquisition', 'frame', 'id'])

        for acq_name, acq_df in acting_df.groupby('acquisition'):
            total = len(acq_df)
            if total == 0:
                continue

            target_sex = acq_df[target_col].astype(int).map(fly_sex)
            for sex in ['M', 'F']:
                count = (target_sex == sex).sum()
                records.append({
                    'assay_type': assay_type,
                    'acquisition': acq_name,
                    'sex': sex,
                    'fraction': count / total,
                })

    if len(records) == 0:
        print("No data to plot.")
        return None, None

    plot_data = pd.DataFrame(records)
    assay_order = sorted(assay_dfs.keys())

    sex_palette = {'M': '#5599ff', 'F': '#ff6666'}

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_data, x='assay_type', y='fraction', hue='sex',
                order=assay_order, hue_order=['M', 'F'],
                palette=sex_palette, errorbar='se', ax=ax, alpha=0.7)
    sns.stripplot(data=plot_data, x='assay_type', y='fraction', hue='sex',
                  order=assay_order, hue_order=['M', 'F'],
                  palette=sex_palette, size=6, jitter=True, ax=ax, alpha=0.85,
                  linewidth=0.5, edgecolor='white', dodge=True,
                  legend=False)

    n_per_assay = plot_data.groupby('assay_type')['acquisition'].nunique()
    ax.set_xticklabels([f'{a}\n(n={n_per_assay.get(a, 0)})' for a in assay_order])
    ax.set_xlabel('assay type')
    ax.set_ylabel('fraction of action frames')
    ax.set_title(f'{action_col}: fraction of frames targeting male vs female')
    ax.legend(title='target sex')
    plt.tight_layout()

    if save_dir is not None:
        savepath = os.path.join(save_dir, f'{action_col}_target_sex_fraction_across_assays.png')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {savepath}")

    return fig, ax