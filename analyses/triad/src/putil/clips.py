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
import libs.plotting as putil
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
        color = id_colors.get(fly_id, putil.fg_color())

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
                arrowprops=dict(arrowstyle='-', color=putil.fg_color(), lw=0.8, alpha=0.6))

    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)
    ax.set_xticks(np.arange(0, frame_width, 100))
    ax.set_yticks(np.arange(0, frame_height, 100))
    ax.grid(True, color='#CCCCCC', linewidth=0.5, alpha=0.3)
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
    ax1.grid(True, color='#CCCCCC', lw=0.5, alpha=0.3)
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

        ax.axhline(0, color=putil.fg_color(), lw=0.5, alpha=0.4)
        ax.axvline(0, color=putil.fg_color(), lw=0.5, alpha=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(lim, -lim)
        ax.set_xticks(np.arange(-int(lim), int(lim), 100))
        ax.set_yticks(np.arange(-int(lim), int(lim), 100))
        ax.grid(True, color='#CCCCCC', lw=0.5, alpha=0.3)
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
                           id_colors, frame_lookup=None, label_fn=None, frame_repeat=1):
    '''
    Write a clip from cap to clip_path.
    frame_lookup  -- dict from _build_frame_lookup; if None, no fly annotation.
    label_fn(frame_idx) -- callable returning (text, bgr_color) for the frame label;
                           if None, no text is drawn.
    frame_repeat  -- write each source frame this many times (>=1). Slows playback by
                     this factor regardless of whether the player honors the fps header,
                     since it changes the actual frame count (default: 1, real time).
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

        for _ in range(frame_repeat):
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
                          padding_frames=120, mark_flies=True, id_colors=None,
                          switch_source='prefer_manual', slowdown=1.0):
    '''
    Extract a short clip around each target-switch event in a courtship bout.

    Arguments:
        df           -- single-acquisition tracks df with switch and target columns
        avi_path     -- path to source .avi file
        save_dir     -- directory to save clips (saved into a 'switch_clips' subdir)

    Keyword Arguments:
        action_col     -- action column (default: 'courtship')
        padding_frames -- frames before/after switch frame to include (default: 120)
        mark_flies     -- if True, draw actor/target markers (default: True)
        id_colors      -- dict mapping fly id → BGR color tuple (default: None)
        switch_source  -- which switch annotation to use: 'auto' ({action_col}_auto_switch),
                          'manual' (switching column), or 'prefer_manual' (manual if present,
                          else auto). Default: 'prefer_manual'.
        slowdown       -- integer playback slow-down factor. Each source frame is written
                          this many times at the native fps, so e.g. slowdown=2 plays the
                          clip back at half speed (2x slower). Frame duplication (rather
                          than lowering the fps header) is used so the slow-down shows in
                          any player, even ones that ignore the declared fps. Default: 1.

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

    target_col = f'{action_col}_target'

    resolved = tutil._resolve_switch_source(df, action_col, switch_source)
    if resolved is None:
        print(f"No switch column available for switch_source={switch_source!r}.")
        return []

    if resolved == 'manual':
        switch_rows = df[(df['switching'] == df['id']) & (df['switching'] != -1)]
        switch_events = (switch_rows
                         .drop_duplicates(subset=['frame', 'id'])
                         [['frame', 'id', target_col]]
                         .copy())
    else:
        switch_col = f'{action_col}_auto_switch'
        switch_events = (df[df[switch_col] == 1]
                         .drop_duplicates(subset=['frame', 'id'])
                         [['frame', 'id', target_col]]
                         .copy())
    source_label = resolved

    if len(switch_events) == 0:
        print("No switch events found.")
        return []

    cap          = cv2.VideoCapture(avi_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_repeat = max(1, int(round(slowdown)))   # duplicate frames to slow playback
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clips_dir = os.path.join(save_dir, 'switch_clips')
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []

    speed_note = f' at {1/frame_repeat:g}x speed' if frame_repeat != 1 else ''
    print(f"Extracting {len(switch_events)} {source_label} switch clip(s) for '{action_col}'{speed_note}")

    for _, ev in switch_events.iterrows():
        switch_frame = int(ev['frame'])
        actor_id     = int(ev['id'])
        to_target    = int(ev[target_col]) if pd.notna(ev[target_col]) and ev[target_col] != -1 else None

        # from_target: last assigned target for this actor before the switch frame
        prior = df[(df['id'] == actor_id) &
                   (df['frame'] < switch_frame) &
                   (df[target_col].notna()) &
                   (df[target_col] != -1)]
        from_target = None
        if len(prior) > 0:
            last_val = prior.sort_values('frame').iloc[-1][target_col]
            from_target = int(last_val) if pd.notna(last_val) and last_val != -1 else None

        clip_start = max(0, switch_frame - padding_frames)
        clip_end   = min(total_frames - 1, switch_frame + padding_frames)

        from_str  = str(from_target) if from_target is not None else 'NA'
        to_str    = str(to_target)   if to_target   is not None else 'NA'
        clip_name = (f'{action_col}_{source_label}_switch_fr{switch_frame:06d}'
                     f'_act{actor_id}_from{from_str}_to{to_str}.mp4')
        clip_path = os.path.join(clips_dir, clip_name)
        print(f"  {clip_name}  ({clip_end - clip_start + 1} frames)")

        frame_lookup = _build_frame_lookup(df, clip_start, clip_end, action_col) if mark_flies else None

        def _label(fi, sf=switch_frame, fs=from_str, ts=to_str):
            if fi == sf:
                return f'fr {fi}   << SWITCH {fs} -> {ts}', (0, 0, 255)
            return f'fr {fi}', (255, 255, 255)

        _write_annotated_clip(cap, clip_path, clip_start, clip_end, fps, frame_w, frame_h,
                              id_colors, frame_lookup=frame_lookup, label_fn=_label,
                              frame_repeat=frame_repeat)
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
            acq_base = re.sub(r'_ch\d+$', '', acq)
            avi_path = os.path.join(rootdir, 'raw_videos', acq_base, f'{acq_base}.avi')
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


def sample_clips_by_metric(df, avi_path, save_dir, metric, fps, acq=None,
                            action_col='courtship',
                            step=1.0, n_clips=3,
                            clip_pre_sec=1.0, clip_post_sec=1.0,
                            metric_range=None,
                            id_colors=None):
    '''
    Sample short video clips across evenly-spaced bins of a metric during action bouts,
    restricted to actor→target pair rows.  One clip per sampled frame, centered on it.

    Arguments:
        df       -- single-acquisition tracks df with action, target, and metric columns
        avi_path -- path to source .avi file
        save_dir -- parent output directory; bin subdirs (e.g. '0.0-1.0/') created inside
        metric   -- column to bin on (e.g. 'target_vel', 'dist_to_other')
        fps      -- frames per second of the video

    Keyword Arguments:
        acq          -- acquisition name used as filename prefix (default: None)
        action_col   -- action column to filter on (default: 'courtship')
        step         -- bin width in metric units (default: 1.0)
        n_clips      -- max clips to extract per bin (default: 3)
        clip_pre_sec -- seconds of video before the sampled frame (default: 1.0)
        clip_post_sec-- seconds of video after the sampled frame (default: 1.0)
        metric_range -- (min, max) tuple; if None uses data range
        id_colors    -- dict mapping fly id → BGR color tuple

    Returns:
        saved_paths -- list of saved .mp4 paths
    '''
    if id_colors is None:
        id_colors = {
            0: (255, 144, 30),
            1: (71, 99, 255),
            2: (50, 205, 50),
            3: (0, 215, 255),
        }

    target_col = f'{action_col}_target'
    required = [action_col, target_col, metric, 'pair', 'id', 'frame']
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
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pre_frames  = int(round(clip_pre_sec * fps))
    post_frames = int(round(clip_post_sec * fps))

    saved_paths = []
    acq_prefix  = f'{acq}_' if acq else ''

    for i in range(len(bin_edges) - 1):
        b_lo, b_hi = bin_edges[i], bin_edges[i + 1]
        bin_rows = plot_df[(plot_df[metric] >= b_lo) & (plot_df[metric] < b_hi)]
        if len(bin_rows) == 0:
            continue

        bin_label = f'{b_lo:.1f}-{b_hi:.1f}'
        bin_dir   = os.path.join(save_dir, bin_label)
        os.makedirs(bin_dir, exist_ok=True)

        sample_rows = bin_rows.sample(min(n_clips, len(bin_rows)), random_state=42)
        for _, row in sample_rows.iterrows():
            center_frame = int(row['frame'])
            actor_id     = int(row['id'])
            target_id_v  = int(row['_target_id'])
            metric_val   = row[metric]

            clip_start = max(0, center_frame - pre_frames)
            clip_end   = min(total_frames - 1, center_frame + post_frames)

            frame_lookup = _build_frame_lookup(df, clip_start, clip_end, action_col)

            def _label(fi, cv=metric_val, cf=center_frame, m=metric, a=acq):
                marker = '>>>' if fi == cf else '   '
                txt = f'{marker} {m}={cv:.2f}  fr={fi}'
                if a:
                    txt += f'  {a}'
                return txt, (255, 255, 255)

            fname = (f'{bin_label}_{acq_prefix}fr{center_frame:06d}'
                     f'_act{actor_id}_targ{target_id_v}.mp4')
            clip_path = os.path.join(bin_dir, fname)
            _write_annotated_clip(cap, clip_path, clip_start, clip_end, fps,
                                  frame_w, frame_h, id_colors,
                                  frame_lookup=frame_lookup, label_fn=_label)
            saved_paths.append(clip_path)

    cap.release()
    return saved_paths


def sample_clips_by_metric_across_assays(assay_dfs, rootdir, metric, figdir,
                                          action_col='courtship',
                                          step=1.0, n_clips=3,
                                          clip_pre_sec=1.0, clip_post_sec=1.0,
                                          metric_range=None,
                                          id_colors=None):
    '''
    Wrapper around sample_clips_by_metric that iterates over all assay types
    and acquisitions.  Saves to:
        figdir/{metric}_clip_examples/{assay_type}/{bin_lo}-{bin_hi}/

    Bins are defined globally across all assay data for comparable intervals.

    Arguments:
        assay_dfs -- dict mapping assay_type → df (from data_io.get_assay_dfs)
        rootdir   -- root data directory (expects raw_videos/{acq}/*.avi or {acq}.avi)
        metric    -- metric column to bin on (e.g. 'target_vel')
        figdir    -- root figures directory

    Keyword Arguments:
        action_col   -- action column to filter on (default: 'courtship')
        step         -- bin width in metric units (default: 1.0)
        n_clips      -- max clips to extract per bin per acquisition (default: 3)
        clip_pre_sec -- seconds before the sampled frame (default: 1.0)
        clip_post_sec-- seconds after the sampled frame (default: 1.0)
        metric_range -- (min, max) tuple; if None computed globally from acting rows
        id_colors    -- dict mapping fly id → BGR color tuple

    Returns:
        all_paths -- list of all saved .mp4 paths
    '''
    # Compute global metric range so bins are consistent across assays
    if metric_range is None:
        target_col = f'{action_col}_target'
        all_vals_list = []
        for d in assay_dfs.values():
            if action_col not in d.columns or metric not in d.columns:
                continue
            if target_col not in d.columns:
                continue
            acting = d[(d[action_col] == d['id']) & (pd.to_numeric(d[target_col], errors='coerce').fillna(-1) != -1)]
            all_vals_list.append(acting[metric].dropna())
        if not all_vals_list or len(pd.concat(all_vals_list)) == 0:
            print(f"No values for '{metric}' during '{action_col}'. Aborting.")
            return []
        all_vals = pd.concat(all_vals_list)
        metric_range = (float(np.floor(all_vals.min())), float(np.ceil(all_vals.max())))
        print(f"Auto metric_range for {metric}: {metric_range}")

    out_root = os.path.join(figdir, f'{metric}_clip_examples')
    os.makedirs(out_root, exist_ok=True)

    all_paths = []
    for assay_type, assay_df in assay_dfs.items():
        if action_col not in assay_df.columns:
            print(f"\n{assay_type}: '{action_col}' column missing, skipping.")
            continue

        assay_out_dir = os.path.join(out_root, assay_type)
        os.makedirs(assay_out_dir, exist_ok=True)

        acqs = sorted(assay_df['acquisition'].unique())
        FPS  = int(assay_df['FPS'].iloc[0]) if 'FPS' in assay_df.columns else 60
        print(f"\n{assay_type}: {len(acqs)} acquisitions  FPS={FPS}")

        for acq in acqs:
            # Support multichamber parquet names (strip _ch{N} suffix for video lookup)
            import re as _re
            base_acq = _re.sub(r'_ch\d+$', '', acq)
            avi_matches = glob.glob(os.path.join(rootdir, 'raw_videos', base_acq, '*.avi'))
            if not avi_matches:
                print(f"  {acq}: .avi not found, skipping.")
                continue
            avi_path = avi_matches[0]

            acq_df = assay_df[assay_df['acquisition'] == acq].copy()
            paths  = sample_clips_by_metric(
                acq_df, avi_path, assay_out_dir, metric, FPS, acq=acq,
                action_col=action_col, step=step, n_clips=n_clips,
                clip_pre_sec=clip_pre_sec, clip_post_sec=clip_post_sec,
                metric_range=metric_range, id_colors=id_colors)
            print(f"  {acq}: {len(paths)} clips saved")
            all_paths.extend(paths)

    print(f"\nTotal: {len(all_paths)} clips → {out_root}")
    return all_paths
