"""Central configuration for Member 1's pipeline (candidate generation + features).

Values here are chosen from TRAIN-partition analysis only (see eda/06_candidate_radius.py).
The evaluator-testing partition is never touched.
"""
import json
import os

DEFAULT_CFG = {
    # --- candidate generation ---
    "radius_m": 50.0,            # keep arcs whose perpendicular distance to the GPS point <= radius
    "fallback_radius_m": 150.0,  # used ONLY for points that have no candidate within radius_m
    "max_candidates": 15,        # keep the K nearest arcs per GPS point
    "sample_spacing_m": 15.0,    # spacing of KD-tree sample points along arcs
    "bbox_margin_m": 500.0,      # arcs farther than this from the track bbox are ignored
    # --- feature engineering ---
    "heading_min_move_m": 3.0,   # below this displacement the GPS heading is treated as undefined
    "gap_threshold_s": 10.0,     # dt above this is flagged as a GPS gap
    # --- label alignment (training / evaluation only, never at inference) ---
    "label_jump_penalty_m": 10.0,
    "label_dist_clip_m": 200.0,
}


def save_cfg(cfg, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_cfg(path=None):
    cfg = dict(DEFAULT_CFG)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg
