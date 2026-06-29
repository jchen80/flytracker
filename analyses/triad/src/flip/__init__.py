"""
flip: orientation-flip detection, correction, interactive review, and plotting
(split from the former diagnose_ori_flips.py).

Submodules:
    detect      -- pure numpy/pandas detection + correction (no cv2/matplotlib)
    review_gui  -- OpenCV interactive review GUI (cv2 imported lazily)
    plots       -- matplotlib diagnostics

Public functions are re-exported here so callers can do
`from analyses.triad.src.flip import detect_flip_events, ...`.
"""
from .detect import (
    print_jump_frames,
    correct_mounting_nearest_ori,
    detect_flip_events,
    correct_flip_events,
)
from .plots import (
    plot_ori_flip_diagnostics,
    plot_flip_duration_distribution,
)