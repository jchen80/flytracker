import numpy as np
import libs.utils as util


def split_by_chamber(trk, feat, calib):
    '''
    Split multi-chamber tracking data into per-chamber subsets.

    For single-chamber data (n_chambers == 1 or unspecified), returns
    [(trk, feat, calib)] unchanged.

    For multi-chamber data, assigns each fly to its chamber using ROI
    bounding boxes from calib['rois'], remaps fly IDs to 0-based within
    each chamber, and returns one (trk_ch, feat_ch, calib_ch) tuple
    per chamber.  The per-chamber calib has the same scalar layout as a
    single-chamber calib (centroids shape (2,), rois shape (4,)).

    Arguments:
        trk   -- fly tracks dataframe with 'id', 'pos_x', 'pos_y' columns
        feat  -- fly features dataframe with 'id' column
        calib -- calibration dict with 'n_chambers', 'rois', 'centroids'

    Returns:
        list of (trk_ch, feat_ch, calib_ch) tuples, one per chamber
    '''
    n_chambers = int(calib.get('n_chambers', 1))
    rois = calib.get('rois')

    # Normalize rois to (n_chambers, 4) — calibration.mat sometimes stores it
    # as a 1D object array of (1, 4) subarrays rather than a clean 2D array.
    if isinstance(rois, np.ndarray) and rois.dtype == object:
        try:
            rois = np.vstack([np.asarray(r).flatten() for r in rois])
        except Exception:
            pass

    if n_chambers <= 1 or rois is None or (isinstance(rois, np.ndarray) and rois.ndim < 2):
        return [(trk, feat, calib, None)]

    # rois: (n_chambers, 4) -> [y, x, h, w] per chamber
    # centroids: (n_chambers, 2) -> [x, y] per chamber
    centroids = calib.get('centroids')
    if centroids is not None:
        centroids = np.asarray(centroids)
        if centroids.dtype == object:
            try:
                centroids = np.vstack([np.asarray(c).flatten() for c in centroids])
            except Exception:
                centroids = None
        if centroids is not None:
            try:
                centroids = centroids.reshape(n_chambers, 2)
            except ValueError:
                print(f"  WARNING: could not reshape centroids {centroids.shape} to "
                      f"({n_chambers}, 2) — falling back to ROI centres.")
                centroids = None

    fly_ids = sorted(trk['id'].unique())
    fly_to_chamber = {}

    for fid in fly_ids:
        fly_trk = trk[trk['id'] == fid]
        med_x = float(fly_trk['pos_x'].median())
        med_y = float(fly_trk['pos_y'].median())

        assigned = False
        for ch_idx in range(n_chambers):
            roi_y, roi_x, roi_h, roi_w = rois[ch_idx]
            if roi_x <= med_x <= roi_x + roi_w and roi_y <= med_y <= roi_y + roi_h:
                fly_to_chamber[fid] = ch_idx
                assigned = True
                break

        if not assigned:
            if centroids is not None:
                dists = [np.hypot(med_x - centroids[ci, 0], med_y - centroids[ci, 1])
                         for ci in range(n_chambers)]
                ch_idx = int(np.argmin(dists))
                fly_to_chamber[fid] = ch_idx
                print(f"  WARNING: fly {fid} (pos {med_x:.1f},{med_y:.1f}) outside all "
                      f"ROI bounding boxes — assigned to nearest chamber {ch_idx}.")
            else:
                fly_to_chamber[fid] = 0
                print(f"  WARNING: fly {fid} outside all ROI bounding boxes and "
                      f"no centroids available — defaulting to chamber 0.")

    # Diagnostic: show which global IDs landed in each chamber
    for ch_idx in range(n_chambers):
        ch_flies_diag = sorted([fid for fid, ch in fly_to_chamber.items() if ch == ch_idx])
        print(f"  Chamber {ch_idx}: global fly IDs {ch_flies_diag} "
              f"-> local IDs {list(range(len(ch_flies_diag)))}")

    chambers = []
    for ch_idx in range(n_chambers):
        ch_flies = sorted([fid for fid, ch in fly_to_chamber.items() if ch == ch_idx])
        if not ch_flies:
            print(f"  WARNING: no flies assigned to chamber {ch_idx}, skipping.")
            continue

        trk_ch = trk[trk['id'].isin(ch_flies)].copy()
        feat_ch = feat[feat['id'].isin(ch_flies)].copy()

        # Remap global IDs to 0-based local IDs (sorted global order)
        id_remap = {old_id: new_id for new_id, old_id in enumerate(ch_flies)}
        trk_ch['id'] = trk_ch['id'].map(id_remap)
        feat_ch['id'] = feat_ch['id'].map(id_remap)

        calib_ch = {k: v for k, v in calib.items()
                    if k not in ('rois', 'centroids', 'n_chambers')}
        calib_ch['n_chambers'] = 1
        calib_ch['rois'] = np.array(rois[ch_idx])
        if centroids is not None:
            calib_ch['centroids'] = centroids[ch_idx].copy()

        chambers.append((trk_ch, feat_ch, calib_ch, ch_flies))

    return chambers


# Transform data
def compute_dist_body_adj(pos_x, pos_y, ori, major_ax, minor_ax,
                          other_pos_x, other_pos_y, other_ori,
                          other_major_ax, other_minor_ax,
                          pix_per_mm=1):
    """
    Minimum gap between two fly ellipses, in mm (or pixels if pix_per_mm=1).

    Each fly is modeled as an ellipse whose major axis is aligned with its
    orientation. For each fly, the radius in the direction toward the other
    fly is computed as:
        r = (a * b) / sqrt((a * sin(alpha))^2 + (b * cos(alpha))^2)
    where a = major_ax/2, b = minor_ax/2, alpha = angle between fly's major
    axis and the line connecting the two centroids.

    Returns max(0, centroid_dist - r_focal - r_other) / pix_per_mm.
    NaN is propagated wherever axis lengths or orientations are NaN.
    """
    dx = other_pos_x - pos_x
    dy = other_pos_y - pos_y
    d  = np.sqrt(dx**2 + dy**2)

    a1, b1 = major_ax / 2.0, minor_ax / 2.0
    a2, b2 = other_major_ax / 2.0, other_minor_ax / 2.0

    angle_to_other = np.arctan2(dy, dx)

    alpha1 = angle_to_other - ori
    denom1 = np.sqrt((a1 * np.sin(alpha1))**2 + (b1 * np.cos(alpha1))**2)
    r1 = np.where(denom1 > 0, (a1 * b1) / denom1, np.nan)

    alpha2 = angle_to_other + np.pi - other_ori
    denom2 = np.sqrt((a2 * np.sin(alpha2))**2 + (b2 * np.cos(alpha2))**2)
    r2 = np.where(denom2 > 0, (a2 * b2) / denom2, np.nan)

    return np.maximum(0.0, d - r1 - r2) / pix_per_mm


def add_pairwise_metrics(trk_, feat_, calib, flyid1=0, flyid2=1):
    """
    Adds the following pairwise metrics to feat_:
    - dist_to_other: distance to the other fly (centroid-to-centroid, mm)
    - dist_to_other_body_adj: ellipse-edge-to-ellipse-edge distance (mm);
      accounts for fly body size by subtracting each fly's ellipse radius
      in the direction toward the other fly
    - facing_angle: angle between fly's orientation and the line connecting
      the two flies

    Assumes that trk_ and feat_ are already filtered to only contain the
    two fly ids of interest (flyid1 and flyid2).
    """
    ppm = calib.get('PPM', 1)
    has_ellipse = all(c in trk_.columns for c in ('major_axis_len', 'minor_axis_len'))

    for fid, oid in [(flyid1, flyid2), (flyid2, flyid1)]:
        f_trk = trk_[trk_['id']==fid]
        o_trk = trk_[trk_['id']==oid]
        dist = util.compute_dist_to_other(
            f_trk['pos_x'].values, f_trk['pos_y'].values,
            o_trk['pos_x'].values, o_trk['pos_y'].values,
            pix_per_mm=ppm)
        feat_.loc[feat_['id']==fid, 'dist_to_other'] = dist

        if has_ellipse:
            dist_adj = compute_dist_body_adj(
                f_trk['pos_x'].values,      f_trk['pos_y'].values,
                f_trk['ori'].values,        f_trk['major_axis_len'].values,
                f_trk['minor_axis_len'].values,
                o_trk['pos_x'].values,      o_trk['pos_y'].values,
                o_trk['ori'].values,        o_trk['major_axis_len'].values,
                o_trk['minor_axis_len'].values,
                pix_per_mm=ppm)
            feat_.loc[feat_['id']==fid, 'dist_to_other_body_adj'] = dist_adj

        facing_angle = util.compute_facing_angle(
            f_trk['ori'].values, f_trk['pos_x'].values, f_trk['pos_y'].values,
            o_trk['pos_x'].values, o_trk['pos_y'].values)
        feat_.loc[feat_['id']==fid, 'facing_angle'] = facing_angle

    return trk_, feat_

