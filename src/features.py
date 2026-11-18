"""Feature engineering: one row per (GPS observation, candidate arc).

Everything here is computable at inference time WITHOUT the reference route. The route is only used
(optionally) to attach labels for training/evaluation. No statistics are learned here, so nothing
is fitted on evaluator data; scaling, if any, is left to the model pipeline (fit on train folds only).
"""
import numpy as np
import pandas as pd

from .candidates import CandidateGenerator
from .geo import LocalProjection, angle_diff_deg, bearing_deg, point_segment_dist
from .labels import align_labels
from .network import RoadNetwork
from .preprocess import clean_track

FEATURE_COLS = [
    "dist", "frac", "dist_rank", "dist_over_min", "n_cand", "is_fallback",
    "dist_prev", "dist_next", "dist_neigh_mean",
    "arc_len", "arc_bearing", "bearing_diff", "cos_bearing", "heading_valid",
    "speed_prev", "speed_next", "speed_change", "step_prev", "step_next",
    "dt_prev", "dt_next", "gap_flag",
    "head_out_deg", "tail_in_deg", "is_dup_arc",
]
ID_COLS = ["trip_id", "obs_idx", "t", "arc_id"]
LABEL_COLS = ["label_arc", "label_valid", "y"]


def point_features(xy, t, cfg):
    """Kinematic features of each GPS point from its neighbours (edges replicate the nearest)."""
    n = len(xy)
    prv = np.vstack([xy[:1], xy[:-1]])
    nxt = np.vstack([xy[1:], xy[-1:]])
    step_prev = np.hypot(*(xy - prv).T)
    step_next = np.hypot(*(nxt - xy).T)
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    if n > 1:
        dt_prev[0], dt_next[-1] = dt_next[0], dt_prev[-1]
    else:
        dt_prev[:] = dt_next[:] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_prev = np.where(dt_prev > 0, step_prev / dt_prev, 0.0)
        speed_next = np.where(dt_next > 0, step_next / dt_next, 0.0)
    cd = nxt - prv                                      # centred displacement -> heading
    move = np.hypot(cd[:, 0], cd[:, 1])
    heading = bearing_deg(cd[:, 0], cd[:, 1])
    heading_valid = move >= cfg["heading_min_move_m"]
    return dict(prv=prv, nxt=nxt, step_prev=step_prev, step_next=step_next, dt_prev=dt_prev,
                dt_next=dt_next, speed_prev=speed_prev, speed_next=speed_next,
                heading=heading, heading_valid=heading_valid)


def build_trip_tables(trip, cfg, with_labels=True):
    """Full per-trip pipeline.

    Returns (cand_df, point_df, meta):
      cand_df  - candidate/feature rows (ID_COLS + FEATURE_COLS [+ LABEL_COLS])
      point_df - one row per GPS observation (+ label info when with_labels)
      meta     - dict of diagnostics (projection centre, candidate counts, miss rate, ...)
    """
    track, rep = clean_track(trip.track)
    lon, lat, t = track["lon"].to_numpy(), track["lat"].to_numpy(), track["t"].to_numpy()
    proj = LocalProjection.from_track(lon, lat)
    xy = np.column_stack(proj.to_xy(lon, lat))
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    n = len(track)

    label_arc = np.full(n, -1, dtype=np.int64)
    label_dist = np.full(n, np.nan)
    if with_labels and trip.route is not None:
        _, label_arc, label_dist = align_labels(net, trip.route, xy, cfg["label_jump_penalty_m"],
                                                cfg["label_dist_clip_m"])

    gen = CandidateGenerator(net, xy, cfg["sample_spacing_m"], cfg["bbox_margin_m"])
    cand = gen.generate(xy, cfg["radius_m"], cfg["max_candidates"], cfg["fallback_radius_m"])

    pf = point_features(xy, t, cfg)
    pt = cand["pt"].to_numpy()
    arc = cand["arc"].to_numpy()
    dist = cand["dist"].to_numpy()
    d_min = cand.groupby("pt")["dist"].transform("min").to_numpy() if len(cand) else dist
    n_cand = cand.groupby("pt")["arc"].transform("size").to_numpy() if len(cand) else dist

    prv_i, nxt_i = np.maximum(pt - 1, 0), np.minimum(pt + 1, n - 1)
    ax, ay, bx, by = net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc]
    dist_prev = point_segment_dist(xy[prv_i, 0], xy[prv_i, 1], ax, ay, bx, by)[0]
    dist_next = point_segment_dist(xy[nxt_i, 0], xy[nxt_i, 1], ax, ay, bx, by)[0]
    hv = pf["heading_valid"][pt]
    bdiff = np.where(hv, angle_diff_deg(pf["heading"][pt], net.bearing[arc]), 90.0)
    cosb = np.where(hv, np.cos(np.radians(bdiff)), 0.0)

    f = {
        "trip_id": trip.trip_id, "obs_idx": track["obs_idx"].to_numpy()[pt], "t": t[pt], "arc_id": arc,
        "dist": dist, "frac": cand["frac"].to_numpy(), "dist_rank": cand["rank"].to_numpy(),
        "dist_over_min": dist - d_min, "n_cand": n_cand, "is_fallback": cand["is_fallback"].to_numpy(),
        "dist_prev": dist_prev, "dist_next": dist_next, "dist_neigh_mean": (dist_prev + dist + dist_next) / 3,
        "arc_len": net.length[arc], "arc_bearing": net.bearing[arc], "bearing_diff": bdiff,
        "cos_bearing": cosb, "heading_valid": hv.astype(np.int8),
        "speed_prev": pf["speed_prev"][pt], "speed_next": pf["speed_next"][pt],
        "speed_change": pf["speed_next"][pt] - pf["speed_prev"][pt],
        "step_prev": pf["step_prev"][pt], "step_next": pf["step_next"][pt],
        "dt_prev": pf["dt_prev"][pt], "dt_next": pf["dt_next"][pt],
        "gap_flag": ((pf["dt_prev"][pt] > cfg["gap_threshold_s"])
                     | (pf["dt_next"][pt] > cfg["gap_threshold_s"])).astype(np.int8),
        "head_out_deg": net.head_out_deg[arc], "tail_in_deg": net.tail_in_deg[arc],
        "is_dup_arc": net.is_dup[arc].astype(np.int8),
    }
    cand_df = pd.DataFrame(f)
    # A label is only trustworthy when the aligned reference arc is actually close to the GPS point.
    # The supplied routes can be incomplete (route ends before the track, or skips a stretch), so
    # points farther than radius_m from their aligned arc are flagged label_valid = 0.
    label_valid = (label_dist <= cfg["radius_m"]) & (label_arc >= 0)
    if with_labels:
        cand_df["label_arc"] = label_arc[pt]
        cand_df["label_valid"] = label_valid[pt].astype(np.int8)
        cand_df["y"] = (arc == label_arc[pt]).astype(np.int8)

    # point-level table
    n_per_pt = np.bincount(pt, minlength=n)
    point_df = pd.DataFrame({"trip_id": trip.trip_id, "obs_idx": track["obs_idx"].to_numpy(), "t": t,
                             "lon": lon, "lat": lat, "n_cand": n_per_pt})
    miss = np.nan
    if with_labels and trip.route is not None:
        hit = np.zeros(n, dtype=bool)
        if len(cand):
            hit[pt[arc == label_arc[pt]]] = True
        top1 = np.zeros(n, dtype=bool)
        if len(cand):
            nn = cand["rank"].to_numpy() == 1
            top1[pt[nn][arc[nn] == label_arc[pt[nn]]]] = True
        point_df["label_arc"] = label_arc
        point_df["label_dist"] = label_dist
        point_df["label_valid"] = label_valid.astype(np.int8)
        point_df["label_in_candidates"] = hit.astype(np.int8)
        point_df["nearest_is_label"] = top1.astype(np.int8)
        miss = 1.0 - hit.mean()
    meta = {"trip_id": trip.trip_id, "partition": trip.partition, "n_points": n,
            "n_candidate_rows": len(cand_df), "mean_candidates": float(n_per_pt.mean()),
            "points_without_candidates": int((n_per_pt == 0).sum()),
            "fallback_points": int(cand.loc[cand["is_fallback"] == 1, "pt"].nunique()) if len(cand) else 0,
            "candidate_miss_rate": miss, "lon0": proj.lon0, "lat0": proj.lat0, **rep}
    if with_labels and trip.route is not None:
        meta["label_dist_median_m"] = float(np.nanmedian(label_dist))
        meta["label_dist_p95_m"] = float(np.nanpercentile(label_dist, 95))
        meta["nearest_arc_accuracy"] = float(point_df["nearest_is_label"].mean())
        meta["label_valid_frac"] = float(label_valid.mean())
        if label_valid.any():
            meta["candidate_miss_rate_valid"] = float(1.0 - hit[label_valid].mean())
            meta["nearest_arc_accuracy_valid"] = float(top1[label_valid].mean())
    return cand_df, point_df, meta
