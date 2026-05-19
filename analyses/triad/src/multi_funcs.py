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

    if n_chambers <= 1 or rois is None or (isinstance(rois, np.ndarray) and rois.ndim < 2):
        return [(trk, feat, calib)]

    # rois: (n_chambers, 4) -> [y, x, h, w] per chamber
    # centroids: (n_chambers, 2) -> [x, y] per chamber
    centroids = calib.get('centroids')

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
                print(f"  WARNING: fly {fid} outside all ROI bounding boxes — "
                      f"assigned to nearest chamber {ch_idx}.")
            else:
                fly_to_chamber[fid] = 0
                print(f"  WARNING: fly {fid} outside all ROI bounding boxes and "
                      f"no centroids available — defaulting to chamber 0.")

    chambers = []
    for ch_idx in range(n_chambers):
        ch_flies = sorted([fid for fid, ch in fly_to_chamber.items() if ch == ch_idx])
        if not ch_flies:
            print(f"  WARNING: no flies assigned to chamber {ch_idx}, skipping.")
            continue

        trk_ch = trk[trk['id'].isin(ch_flies)].copy()
        feat_ch = feat[feat['id'].isin(ch_flies)].copy()

        id_remap = {old_id: new_id for new_id, old_id in enumerate(ch_flies)}
        trk_ch['id'] = trk_ch['id'].map(id_remap)
        feat_ch['id'] = feat_ch['id'].map(id_remap)

        calib_ch = {k: v for k, v in calib.items()
                    if k not in ('rois', 'centroids', 'n_chambers')}
        calib_ch['n_chambers'] = 1
        calib_ch['rois'] = np.array(rois[ch_idx])
        if centroids is not None:
            calib_ch['centroids'] = np.array(centroids[ch_idx])

        chambers.append((trk_ch, feat_ch, calib_ch))

    return chambers


# Transform data
def add_pairwise_metrics(trk_, feat_, calib, flyid1=0, flyid2=1):
    """
    Adds the following pairwise metrics to feat_:
    - dist_to_other: distance to the other fly
    - facing_angle: angle between fly's orientation and the line connecting the two flies

    Assumes that trk_ and feat_ are already filtered to only contain the two fly ids of interest (flyid1 and flyid2).
    """
    ppm = calib.get('PPM', 1)
    for fid, oid in [(flyid1, flyid2), (flyid2, flyid1)]:
        f_trk = trk_[trk_['id']==fid]
        o_trk = trk_[trk_['id']==oid]
        dist = util.compute_dist_to_other(
            f_trk['pos_x'].values, f_trk['pos_y'].values,
            o_trk['pos_x'].values, o_trk['pos_y'].values,
            pix_per_mm=ppm)
        feat_.loc[feat_['id']==fid, 'dist_to_other'] = dist
        facing_angle = util.compute_facing_angle(
            f_trk['ori'].values, f_trk['pos_x'].values, f_trk['pos_y'].values,
            o_trk['pos_x'].values, o_trk['pos_y'].values)
        feat_.loc[feat_['id']==fid, 'facing_angle'] = facing_angle

    return trk_, feat_

